"""Event types emitted by Sentinel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from token_sentinel.policy_client import Policy


@dataclass
class CallRecord:
    """A single LLM API call captured by the tracer.

    Token-priced providers (Anthropic, OpenAI, Gemini, Bedrock) populate
    ``prompt_tokens`` / ``completion_tokens`` and leave ``usage_extra`` as
    the default empty dict. Providers that bill on dimensions other than
    tokens (+: Replicate per-image / per-pixel / per-second, future
    Runway / Luma / Cartesia) leave the token counts at 0 and stash the
    pricing dimension under ``usage_extra``.

    ``usage_extra``: For providers that bill on dimensions other than tokens
        (per-second of audio, per-pixel of image, per-image of generation).
        Schema convention::

            {
                "dimension_kind": "per_image" | "per_pixel" | "per_second"
                                  | "per_character" | "per_token",
                "dimension_value": float,
                "model_specific_meta": {...},   # optional, free-form
            }

        Empty dict for token-priced providers — those use the regular
        ``prompt_tokens`` / ``completion_tokens`` fields. Both populated
        and empty shapes are valid. No enum enforcement at the SDK
        boundary: the schema is structural only so + can add new
        ``dimension_kind`` values (e.g. ``"per_megapixel"``) without an
        SDK release. The cloud cost estimator's ``NON_TOKEN_PRICES``
        table dispatches on the string and falls back to
        ``model_unknown_fallback`` on an unrecognised key.
    """

    session_id: str
    timestamp: datetime
    provider: str
    model: str
    method: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    request_hash: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    user_facing_output: bool = False
    raw_request: dict[str, Any] = field(default_factory=dict)
    raw_response_meta: dict[str, Any] = field(default_factory=dict)
    # non-token pricing dimension (per_image / per_pixel / per_second).
    # See docstring above for the schema convention. Empty dict is the
    # backwards-compatible default — existing wrappers don't need to be
    # updated to populate it.
    usage_extra: dict[str, Any] = field(default_factory=dict)
    # Tag-based Chargeback: per-team / per-feature / per-customer cost
    # attribution. Populated by ``Sentinel.session(tags={...})`` and propagated
    # to every CallRecord (and downstream LeakEvent) emitted while that
    # session is active. Allowed keys are restricted at the Sentinel.session()
    # boundary to the allowlist (team / feature / customer / environment
    # / version); values are validated as URL-safe strings. Empty dict is the
    # backwards-compatible default — pre- customers who don't call
    # session() get the same CallRecord shape as before.
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LeakEvent:
    """A leak signal emitted by a rule.

    The optional :attr:`metadata` field is the V1 cloud-side judge verdict
    trail. When the cloud-side LLM-as-judge ratifies (or vetoes) a gray-zone
    event (raw confidence in [0.5, 0.8]), it writes a ``metadata["judge"]``
    sub-dict carrying:

        {
            "verdict": "ratify" | "veto" | "abstain",
            "raw_confidence": 0.62,
            "calibrated_confidence": 0.91,
            "primary_model": "claude-haiku-4-5",
            "shadow_model": "gemini-2.5-flash",   # if the 10% shadow ran
            "shadow_verdict": "ratify",            # null otherwise
            "reasoning": "...",
        }

    Customers reading the event in their ``on_leak`` handler can render the
    verdict in their dashboards. The field defaults to an empty dict so
    existing rule code (which never sets metadata) continues to round-trip
    unchanged.
    """

    type: str
    confidence: float
    project: str
    session_id: str
    rule: str
    evidence: dict[str, Any]
    estimated_burn: float
    suggested_action: str
    raised_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # LLM-as-judge ratification trail. Populated cloud-side when an event
    # lands in the gray zone (raw confidence in [judge_threshold_low,
    # judge_threshold_high]). Empty dict for events that never went through
    # the judge (sub-low, super-high, or judge disabled).
    metadata: dict[str, Any] = field(default_factory=dict)
    # Tag-based Chargeback: propagated from the originating CallRecord so
    # leak handlers (Slack, Datadog, internal cost dashboards) can route /
    # attribute the event by team/feature/customer/environment/version.
    # Empty dict for sessions that weren't created with tags — preserves
    # back-compat for every pre- leak-handler call site.
    tags: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        # NIT-3: include session_id and a key-only evidence summary so
        # error reports / log lines have enough info to triage. Evidence
        # values are deliberately omitted to avoid the redaction-leak
        # vector that the sample_args fix closed.
        evidence_keys = ",".join(sorted(self.evidence.keys())) if self.evidence else ""
        return (
            f"LeakEvent(type={self.type}, confidence={self.confidence:.2f}, "
            f"burn=${self.estimated_burn:.4f}, rule={self.rule}, "
            f"session_id={self.session_id}, evidence_keys=[{evidence_keys}])"
        )


class LeakDetected(Exception):
    """Raised by the wrapper when sentinel.mode == 'block' and a leak fires."""

    def __init__(self, event: LeakEvent):
        self.event = event
        super().__init__(str(event))


# ---------------------------------------------------------------------------
# Intervention Pack — policy enforcement exceptions
# ---------------------------------------------------------------------------
#
# The exceptions below are raised UNCONDITIONALLY by ``Sentinel.record_call``
# when the cloud-pushed policy says the call must be halted. They subclass
# :class:`LeakDetected` so existing ``mode='block'`` handling treats them
# correctly, but they are NOT gated on ``mode``: the customer opts into
# enforcement by configuring a policy endpoint, and that opt-in by definition
# means "halt the call when the policy says so." Logging mode is still
# meaningful for the rule engine (rules emit signals, no halt); the policy
# plane is enforcement and is a separate axis.
#
# Each exception carries the :class:`Policy` instance it acted on and the
# specific metric that tripped, so handlers can render meaningful errors
# without re-fetching the policy.


class BudgetExceeded(LeakDetected):
    """Raised when a session's cumulative ``estimated_burn`` would exceed the
    cloud-pushed ``policy.budget_usd_per_session``.

    The exception fires at the call boundary BEFORE the rule loop runs, so
    the offending call is rejected rather than recorded — the cloud's running
    aggregate stays consistent with the SDK's local view.
    """

    def __init__(
        self,
        event: LeakEvent,
        *,
        policy: Policy,
        session_id: str,
        current_usd: float,
        next_call_usd: float,
        budget_usd: float,
    ):
        # Stash the structured fields so handlers can render rich messages
        # (Slack, dashboards, log lines) without re-parsing the str(event).
        self.policy = policy
        self.session_id = session_id
        self.current_usd = current_usd
        self.next_call_usd = next_call_usd
        self.budget_usd = budget_usd
        super().__init__(event)


class VelocityExceeded(LeakDetected):
    """Raised when project-wide tokens/min exceeds ``policy.max_tokens_per_min``.

    The "absolute ceiling" velocity layer per the  founder decision —
    no per-customer baseline; a single hard cap on tokens-per-minute. False
    positives at low traffic are accepted for  and documented in the
    release notes; per-customer baselines land in .1.
    """

    def __init__(
        self,
        event: LeakEvent,
        *,
        policy: Policy,
        current_tokens_per_min: int,
        max_tokens_per_min: int,
    ):
        self.policy = policy
        self.current_tokens_per_min = current_tokens_per_min
        self.max_tokens_per_min = max_tokens_per_min
        super().__init__(event)


class KillSwitchActive(LeakDetected):
    """Raised when the operator hit the dashboard kill-switch.

    Halts ALL in-flight calls for the project at the next call boundary. The
    operator's intent is "stop this agent right now"; the SDK obeys at the
    next ``record_call``. There is no grace period or per-session opt-out —
    kill-switch is total.
    """

    def __init__(self, event: LeakEvent, *, policy: Policy):
        self.policy = policy
        super().__init__(event)


# ---------------------------------------------------------------------------
# Rebrand aliases — "token leak" → "token waste"
# ---------------------------------------------------------------------------
#
# The marketing surface (docs, blog, landing pages) renamed "token leak" to
# "token waste" in 2026-05. The internal Python API symbols were intentionally
# preserved to avoid breaking installed customer code; these aliases let new
# users adopt the rebranded names while existing call sites keep working.
#
# These are TRANSPARENT aliases — ``WasteEvent is LeakEvent`` evaluates to
# ``True``, ``isinstance(x, WasteEvent)`` succeeds for any ``LeakEvent``
# instance, and ``except WasteDetected:`` catches ``LeakDetected`` (and
# vice-versa). No ``DeprecationWarning`` is emitted: the old names remain
# first-class.
WasteEvent = LeakEvent
WasteDetected = LeakDetected
