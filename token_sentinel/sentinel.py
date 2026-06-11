"""Main Sentinel class — public API."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
import warnings
from collections import OrderedDict, deque
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, TypeVar

from token_sentinel.events import (
    BudgetExceeded,
    CallRecord,
    KillSwitchActive,
    LeakDetected,
    LeakEvent,
    VelocityExceeded,
)
from token_sentinel.rules import Rule, default_rules
from token_sentinel.tracer import Tracer

T = TypeVar("T")

Mode = Literal["log", "alert", "block"]
PolicyFailureMode = Literal["open", "closed"]
LeakHandler = Callable[[LeakEvent], None]


# ---------------------------------------------------------------------------
# Intervention Pack — module constants
# ---------------------------------------------------------------------------

# Rolling window for the absolute-ceiling velocity check (tokens/min). The
# spec is "tokens per *minute*" so we hard-code 60 seconds — making this
# configurable would just create misalignment between the SDK's local view
# and the cloud's velocity_max_tokens_per_min semantics.
_VELOCITY_WINDOW_SECONDS = 60.0

# Per-call USD burn estimate. This is the same heuristic that
# ``rules/tool_loop.py``'s ``_estimate_burn`` uses for its 3-cycle
# extrapolation: average per-token cost across the major frontier models
# rounds to about $9e-6 per token. We use it here at the per-call scale to
# decide "would this call push us over the session budget?" before the rule
# loop runs. Identical constant on purpose — deviation would create
# confusion when a customer sees the rule-side and policy-side burn numbers
# disagree.
_BURN_USD_PER_TOKEN = 9e-6


# Sentinel value distinguishing "kwarg not passed (use cloud_endpoint
# default)" from "explicitly passed None (disable policy plane)". Keeping
# this private to the module — customers should pass either a string URL
# or ``None`` to opt out; they should never see ``_POLICY_DEFAULT`` itself.
_POLICY_DEFAULT = object()


# ---------------------------------------------------------------------------
# Tag-based Chargeback — validation constants
# ---------------------------------------------------------------------------
#
# Tag keys are gated to a small, stable allowlist. Customer discovery
# converged on five dimensions: team / feature / customer / environment /
# version. We reject anything else at the SDK boundary (ValueError) and at
# the cloud boundary (400) so the dashboards have a known finite tag-key
# universe to render. Future tag keys land via SDK release + cloud release
# (not a runtime config knob) — keeping the surface stable matters more
# than letting customers ship arbitrary keys today.
_ALLOWED_TAG_KEYS: frozenset[str] = frozenset(
    {"team", "feature", "customer", "environment", "version"}
)

# URL-safe value regex. Same restrictions as the cloud-side validator —
# we keep the SDK and cloud regex literally identical so a tag that
# passes the SDK never gets rejected by the cloud (or vice-versa). No
# spaces, no quotes, no shell metacharacters: keeps `?tag=team&value=...`
# safe to round-trip through query strings without escaping.
_TAG_VALUE_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

# Max per-value chars. 64 is generous for human-meaningful values
# (team names, feature flags, customer IDs) while bounding the audit-log
# / dashboard rendering cost. Aligns with the cloud-side cap.
_MAX_TAG_VALUE_LENGTH = 64

# Max dict entries per session. We never want a single agent shipping
# 100 tag keys — that would explode the by-tag aggregation cardinality
# cloud-side. Eight is enough for the allowlist of five plus
# headroom for  additions.
_MAX_TAGS_PER_SESSION = 8


def _validate_session_tags(tags: dict[str, str] | None) -> dict[str, str]:
    """Validate and normalize a ``tags`` dict for ``Sentinel.session()``.

    Raises ValueError with a specific reason on:
      - key not in ``_ALLOWED_TAG_KEYS``
      - value not a string
      - value > ``_MAX_TAG_VALUE_LENGTH`` chars
      - value doesn't match ``_TAG_VALUE_PATTERN`` (URL-safe)
      - dict length > ``_MAX_TAGS_PER_SESSION``

    Returns a fresh ``dict[str, str]`` (defensive copy) so a customer
    mutating the original after ``session(tags=...)`` doesn't retroactively
    change the session's tag set.
    """
    if tags is None:
        return {}
    if not isinstance(tags, dict):
        raise ValueError(
            f"TokenSentinel: session(tags=...) must be a dict, got {type(tags).__name__}"
        )
    if len(tags) > _MAX_TAGS_PER_SESSION:
        raise ValueError(
            f"TokenSentinel: session tags must have ≤ "
            f"{_MAX_TAGS_PER_SESSION} entries, got {len(tags)}"
        )
    out: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            raise ValueError(
                f"TokenSentinel: session tag key must be a string, got {type(key).__name__}"
            )
        if key not in _ALLOWED_TAG_KEYS:
            raise ValueError(
                f"TokenSentinel: session tag key {key!r} not in the  "
                f"allowlist {sorted(_ALLOWED_TAG_KEYS)}"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"TokenSentinel: session tag value for {key!r} must be a "
                f"string, got {type(value).__name__}"
            )
        if len(value) > _MAX_TAG_VALUE_LENGTH:
            raise ValueError(
                f"TokenSentinel: session tag value for {key!r} exceeds "
                f"{_MAX_TAG_VALUE_LENGTH} chars (got {len(value)})"
            )
        if not _TAG_VALUE_PATTERN.match(value):
            raise ValueError(
                f"TokenSentinel: session tag value for {key!r} must match "
                f"^[a-zA-Z0-9._-]+$ (URL-safe; no spaces or special chars); "
                f"got {value!r}"
            )
        out[key] = value
    return out


class Session:
    """A logical agent session — the  chargeback boundary.

    Returned by :meth:`Sentinel.session`. The customer's wrapped LLM client
    reads the session's id + tags from this object when building each
    :class:`CallRecord`; tags propagate to every downstream
    :class:`LeakEvent` so leak handlers and the cloud chargeback API can
    attribute spend by team / feature / customer / environment / version.

    The session object is intentionally cheap: just a session_id + frozen
    tag dict. It does NOT take ownership of the wrapped client, install
    handlers, or open any cloud connections — the Sentinel instance still
    owns all of that. Multiple Session objects coexist freely on a single
    Sentinel; closing one doesn't affect others.

    A Session has no ``close()`` method by design — sessions naturally
    age out of the Sentinel's per-session burn dict and the Tracer's
    LRU on the existing ``max_sessions`` cap. Pinning lifecycle to a
    Python object would create a footgun where forgetting to ``close()``
    leaked memory.
    """

    __slots__ = ("session_id", "tags", "_sentinel")

    def __init__(
        self,
        sentinel: Sentinel,
        session_id: str,
        tags: dict[str, str],
    ) -> None:
        self.session_id: str = session_id
        # Store a shallow copy so external mutation of the caller's dict
        # doesn't retroactively change this session's tag set. The values
        # are strings (immutable) so no deep copy is needed.
        self.tags: dict[str, str] = dict(tags)
        self._sentinel: Sentinel = sentinel

    def record_call(self, call: CallRecord) -> list[LeakEvent]:
        """Convenience: stamp this session's tags onto the call before
        recording.

        The wrapper layer reads ``session.tags`` directly when building
        CallRecords, but customers who construct CallRecord by hand (the
        rare case — e.g., custom enrichers, or test code) can route
        through here for the same tag-propagation behaviour. The
        underlying ``Sentinel.record_call`` is unchanged.
        """
        # Defensive: ensure the call's session_id matches this session's
        # id. If a caller hand-builds a CallRecord and threads it through
        # the wrong Session.record_call, silently overwriting the
        # session_id would be a footgun. We DO overwrite when the field
        # is empty (the common case for ad-hoc CallRecords), but raise
        # on a real mismatch so the bug surfaces at the call site.
        if call.session_id and call.session_id != self.session_id:
            raise ValueError(
                f"Session.record_call: call.session_id "
                f"{call.session_id!r} != session.session_id "
                f"{self.session_id!r}"
            )
        if not call.session_id:
            call.session_id = self.session_id
        # Apply tags only if the caller didn't already set them — never
        # clobber a hand-set CallRecord.tags dict. The wrapper layer
        # builds fresh CallRecords with the empty default, so the merge
        # branch covers the customer-built-CallRecord case.
        if not call.tags:
            call.tags = dict(self.tags)
        return self._sentinel.record_call(call)

    def __repr__(self) -> str:
        tag_summary = ",".join(sorted(self.tags.keys())) if self.tags else ""
        return f"Session(session_id={self.session_id!r}, tags=[{tag_summary}])"


def _estimate_call_burn_usd(call: CallRecord) -> float:
    """Estimate the USD burn for a single call.

    Uses the same per-token coefficient as :func:`tool_loop._estimate_burn`
    so the policy plane and the rule engine agree on units. Negative token
    counts (which would only arrive from a buggy provider parser) are
    clamped to zero so we never produce a negative burn that could mask a
    real overage.
    """
    prompt = max(0, call.prompt_tokens)
    completion = max(0, call.completion_tokens)
    return (prompt + completion) * _BURN_USD_PER_TOKEN


class Sentinel:
    """Wrap LLM clients and watch for token leaks.

    Example:
        sentinel = Sentinel(project="my-agent")
        client = sentinel.wrap(anthropic.Anthropic())

        @sentinel.on_leak
        def handler(event):
            print(event)
    """

    def __init__(
        self,
        *,
        project: str,
        mode: Mode = "log",
        rules: list[str] | Literal["all"] = "all",
        config: dict[str, Any] | None = None,
        cloud_endpoint: str | None = None,
        api_key: str | None = None,
        min_confidence: float = 0.5,
        max_records_per_session: int = 200,
        max_sessions: int | None = 1000,
        dedup_window_seconds: float = 5.0,
        cloud_flush_interval_seconds: float = 5.0,
        cloud_batch_size: int = 50,
        cloud_queue_max: int = 1000,
        # --- Intervention Pack ---------------------------------------
        # Pull-based policy plane. The default sentinel ``_POLICY_DEFAULT``
        # means "use cloud_endpoint when it is set" — existing cloud
        # customers get the policy plane automatically. Pass
        # ``policy_endpoint=None`` to explicitly disable; pass a URL to
        # override (e.g., when policy lives on a different host than
        # event ingestion).
        policy_endpoint: str | None = _POLICY_DEFAULT,  # type: ignore[assignment]
        policy_active_poll_seconds: float = 2.0,
        policy_idle_poll_seconds: float = 5.0,
        policy_failure_mode: PolicyFailureMode = "open",
        # --- LLM-as-judge ratification (Pro tier) ----------------------
        # The judge runs cloud-side; the SDK only forwards these as request
        # headers on every ``POST /v1/events`` call so the cloud knows the
        # customer's gray-zone window and monthly cap. The cloud is the
        # source of truth for cost/billing — these are hints. V1.1 will add
        # a sticky ``PUT /api/judge-config`` endpoint; for V1 they round-trip
        # per-batch via headers.
        #
        # ``judge_threshold_low``/``judge_threshold_high`` define the gray
        # zone where ratification kicks in. Defaults match
        # ``pro_tier_blueprint.md`` Feature 1's spec (0.5 → 0.8).
        # ``judge_calls_per_month_max`` is the customer's hard cap on judge
        # invocations for the project; defaults to 1.8M (the 18%-of-events
        # baseline at 10M ingested events/mo, per ``llm_judge_finetuning.md``
        # Part 2.8).
        judge_threshold_low: float = 0.5,
        judge_threshold_high: float = 0.8,
        judge_calls_per_month_max: int = 1_800_000,
    ):
        self.project = project
        self.mode = mode
        self.config = config or {}
        self.cloud_endpoint = cloud_endpoint
        self.api_key = api_key
        # Clamp min_confidence to [0.0, 1.0] (LOW-4). A customer who passes
        # 2.0 would otherwise see no events fire ever; -1.0 would let every
        # event through. Both are surprising silent failures — clamp instead.
        if not 0.0 <= min_confidence <= 1.0:
            import warnings as _warnings

            _warnings.warn(
                f"TokenSentinel: min_confidence={min_confidence!r} clamped to [0.0, 1.0]",
                UserWarning,
                stacklevel=2,
            )
            min_confidence = max(0.0, min(1.0, min_confidence))
        self.min_confidence = min_confidence

        # judge config — stored as instance attributes so the customer can
        # introspect, but forwarded to the cloud sink (see below) so the
        # cloud knows the customer's gray-zone preferences on every batch.
        # Defensive clamping: thresholds must be in [0, 1] and low <= high
        # for the gray zone to be non-empty. We warn rather than raise so a
        # mis-typed threshold doesn't crash the customer's startup.
        if not 0.0 <= judge_threshold_low <= 1.0:
            warnings.warn(
                f"TokenSentinel: judge_threshold_low={judge_threshold_low!r} clamped to [0.0, 1.0]",
                UserWarning,
                stacklevel=2,
            )
            judge_threshold_low = max(0.0, min(1.0, judge_threshold_low))
        if not 0.0 <= judge_threshold_high <= 1.0:
            warnings.warn(
                f"TokenSentinel: judge_threshold_high={judge_threshold_high!r} "
                "clamped to [0.0, 1.0]",
                UserWarning,
                stacklevel=2,
            )
            judge_threshold_high = max(0.0, min(1.0, judge_threshold_high))
        if judge_threshold_low > judge_threshold_high:
            warnings.warn(
                f"TokenSentinel: judge_threshold_low ({judge_threshold_low}) > "
                f"judge_threshold_high ({judge_threshold_high}); swapping.",
                UserWarning,
                stacklevel=2,
            )
            judge_threshold_low, judge_threshold_high = (
                judge_threshold_high,
                judge_threshold_low,
            )
        if judge_calls_per_month_max < 0:
            judge_calls_per_month_max = 0
        self.judge_threshold_low = float(judge_threshold_low)
        self.judge_threshold_high = float(judge_threshold_high)
        self.judge_calls_per_month_max = int(judge_calls_per_month_max)
        self.tracer = Tracer(
            max_records_per_session=max_records_per_session,
            max_sessions=max_sessions,
        )
        self._rules: list[Rule] = self._load_rules(rules)
        self._handlers: list[LeakHandler] = []
        # Guards mutation/iteration of ``_handlers`` so that registering a new
        # handler while a dispatch is in flight (in another thread) cannot see
        # partial list state. The lock is *only* held while snapshotting the
        # list — handler invocation runs outside the lock so a slow / re-entrant
        # handler cannot deadlock dispatch.
        self._handler_lock = Lock()

        # --- Event de-duplication state (MED-4) ---------------------------
        # Two threads concurrently `record_call`-ing the same session can each
        # see overlapping rule windows and fire the same event with identical
        # evidence. To avoid duplicate signals at customer leak handlers, we
        # track per-session "recently-fired" event keys and suppress repeats
        # within ``dedup_window_seconds``. A value of 0 disables dedup
        # entirely (no state mutation, no lookup — zero cost).
        self._dedup_window_seconds = float(dedup_window_seconds)
        # session_id -> (event_key -> timestamp_seconds)
        self._recent_event_keys: dict[str, dict[str, float]] = {}
        self._dedup_lock = threading.Lock()

        # --- Cloud sink  -------------------------------------------
        # Lazily instantiated only when the customer wires up cloud delivery
        # by passing both ``cloud_endpoint`` and ``api_key``. Either alone is
        # treated as a misconfiguration silently — no sink, no thread spawn.
        # We import lazily so ``CloudSink`` is never loaded by customers who
        # don't use the cloud feature, keeping the cold-import cost minimal.
        self._cloud_sink: Any = None
        if cloud_endpoint and api_key:
            from token_sentinel import __version__ as _sdk_version
            from token_sentinel.cloud_client import CloudSink

            self._cloud_sink = CloudSink(
                endpoint=cloud_endpoint,
                api_key=api_key,
                project=project,
                sdk_version=_sdk_version,
                flush_interval_seconds=cloud_flush_interval_seconds,
                batch_size=cloud_batch_size,
                queue_max=cloud_queue_max,
                # forward the Sentinel's mode so every outbound event
                # carries it on the wire. Pre-V1.3 clouds discard unknown
                # fields (Pydantic ``extra='ignore'``), so this is safe to
                # ship to any cloud version. V1.3+ clouds use the field to
                # drive the per-mode savings credit (block 1.0×, alert 0.5×,
                # log 0.1×).
                mode=self.mode,
                # Pro-tier judge knobs — sent as request headers on every
                # POST /v1/events. The cloud reads them to drive per-call
                # gray-zone logic. None of the values are sensitive; keeping
                # them in headers (vs. body) lets the cloud key on them
                # without re-parsing the JSON.
                extra_headers={
                    "X-Judge-Threshold-Low": f"{self.judge_threshold_low:.4f}",
                    "X-Judge-Threshold-High": f"{self.judge_threshold_high:.4f}",
                    "X-Judge-Calls-Cap": str(self.judge_calls_per_month_max),
                },
            )

        # --- Policy client (Intervention Pack) ----------------------
        # Spawn the daemon-thread policy poller alongside the cloud sink,
        # gated on the same (cloud_endpoint AND api_key) condition so
        # existing customers automatically get the policy plane. The
        # customer can explicitly disable the policy plane while keeping
        # the cloud sink on by passing ``policy_endpoint=None`` after the
        # default-from-cloud-endpoint assignment below.
        #
        # Policy plane vs. cloud sink: they share the api_key and project
        # but talk to different endpoints (``/v1/policy`` vs. ``/v1/events``).
        # A future deployment may host them separately, hence the
        # ``policy_endpoint`` knob — defaults to ``cloud_endpoint`` for the
        # single-host case.
        self._policy_client: Any = None
        # Track the per-session running burn (USD) for the budget check, and
        # a project-wide rolling 1-minute token deque for the velocity check.
        # OrderedDict for the LRU-evict-when-too-many-sessions pattern that
        # mirrors :class:`Tracer`. We reuse ``max_sessions`` since the burn
        # dict shares the same lifecycle as the tracer's session entries.
        self._session_burn: OrderedDict[str, float] = OrderedDict()
        self._tokens_minute_window: deque[tuple[float, int]] = deque()
        self._policy_lock = threading.Lock()
        self._max_sessions_for_burn = max_sessions

        # Resolve policy_endpoint:
        #   - default sentinel ``_POLICY_DEFAULT`` → fall back to
        #     ``cloud_endpoint`` (existing cloud customers get the
        #     policy plane automatically).
        #   - explicit ``None`` → opt out; no policy plane runs.
        #   - explicit URL → use it as the policy endpoint.
        if policy_endpoint is _POLICY_DEFAULT:
            resolved_policy_endpoint = cloud_endpoint
        else:
            resolved_policy_endpoint = policy_endpoint

        if resolved_policy_endpoint and api_key:
            from token_sentinel import __version__ as _sdk_version
            from token_sentinel.policy_client import PolicyClient

            self._policy_client = PolicyClient(
                endpoint=resolved_policy_endpoint,
                api_key=api_key,
                project=project,
                active_poll_seconds=policy_active_poll_seconds,
                idle_poll_seconds=policy_idle_poll_seconds,
                failure_mode=policy_failure_mode,
                sdk_version=_sdk_version,
            )
            self._policy_client.start()

    # ---------------------------------------------------------------------
    # Tag-based Chargeback — session factory
    # ---------------------------------------------------------------------

    def session(
        self,
        session_id: str | None = None,
        *,
        tags: dict[str, str] | None = None,
    ) -> Session:
        """Open a logical session with optional chargeback tags.

        Returns a :class:`Session` whose ``session_id`` and ``tags`` the
        customer threads through their wrapped LLM calls. Tags propagate
        to every :class:`CallRecord` and downstream :class:`LeakEvent`
        emitted within the session so the cloud chargeback API can
        aggregate spend by team / feature / customer / environment /
        version.

        Args:
            session_id: Optional explicit session id. When ``None``, a
                fresh UUID4 is generated — matching the wrapper-level
                default. Pass an explicit id when your agent already
                threads its own session/run identifier (the common
                pattern for LangChain / LlamaIndex integrations).
            tags: Optional ``dict[str, str]`` of chargeback tags. Keys
                are restricted to the allowlist
                (``team`` / ``feature`` / ``customer`` / ``environment``
                / ``version``). Values must be URL-safe
                (``^[a-zA-Z0-9._-]+$``), ≤64 chars. Max 8 entries.
                ``None`` / omitted is equivalent to ``{}`` — full
                back-compat with pre- callers.

        Raises:
            ValueError: when a tag key isn't in the allowlist, a
                value isn't URL-safe, a value exceeds the length cap, or
                the dict exceeds the per-session entry cap. The error
                message names the offending key + value so the customer
                can fix the call site without grep'ing this file.

        Example:
            >>> sentinel = Sentinel(project="my-agent")
            >>> growth_team = sentinel.session(tags={"team": "growth"})
            >>> # Wrapped client calls inside this session pick up the tags.
        """
        # Validate FIRST so we raise BEFORE allocating the Session
        # object — failed validation should leave no state behind.
        validated = _validate_session_tags(tags)
        sid = session_id if session_id else str(uuid.uuid4())
        return Session(self, sid, validated)

    def _load_rules(self, requested: list[str] | str) -> list[Rule]:
        all_rules = default_rules(self.config)
        if requested == "all":
            return all_rules
        # MED-2: warn (don't raise — that would break customers relying on the
        # historical silent-drop) when the customer requested rule names that
        # don't exist. Typos like ``rules=["tool_loop", "tool_lop"]`` previously
        # silently loaded only ``tool_loop`` and the customer never noticed.
        all_rule_names = {r.name for r in all_rules}
        # ``requested`` is a list[str] in this branch (the "all" literal was
        # handled above); guard for non-iterable inputs by coercing to a list.
        try:
            requested_set = set(requested)
        except TypeError:
            requested_set = set()
        unknown = requested_set - all_rule_names
        if unknown:
            warnings.warn(
                f"TokenSentinel: unknown rule names ignored: {sorted(unknown)}. "
                f"Known rules: {sorted(all_rule_names)}.",
                UserWarning,
                stacklevel=3,
            )
        return [r for r in all_rules if r.name in requested_set]

    def wrap(self, client: T) -> T:
        """Wrap an LLM client. Returns the client with instrumented methods.

        Supports:
        - Anthropic: ``anthropic.Anthropic``, ``anthropic.AsyncAnthropic``
        - OpenAI: ``openai.OpenAI``, ``openai.AsyncOpenAI`` (and any
          OpenAI-compatible base_url: DeepSeek, Together, Fireworks, Groq, vLLM, …)
        - Google Gemini: ``google.genai.Client`` (covers Vertex AI when
          constructed with ``vertexai=True``)
        - AWS Bedrock: boto3 ``bedrock-runtime`` client
        - Voyage AI: ``voyageai.Client``, ``voyageai.AsyncClient``
          (embeddings + reranking — Anthropic's recommended embeddings provider)
        - Cohere V2: ``cohere.ClientV2``, ``cohere.AsyncClientV2``
          (chat + embeddings + rerank — enterprise-RAG focus per )
        - Replicate: ``replicate.Client`` (image/video models — non-token
          pricing via ``usage_extra`` dimension dispatch)
        - Deepgram: ``deepgram.DeepgramClient``, ``deepgram.AsyncDeepgramClient``
          (STT — pre-recorded file/URL + live websocket transcription)
        - ElevenLabs: ``elevenlabs.client.ElevenLabs``, ``elevenlabs.client.AsyncElevenLabs``
          (TTS — text-to-speech, billed per-character of input text)

        Raises:
            TypeError: if the client is unsupported, OR if the client is a
                supported provider but the expected accessor (e.g.
                ``client.messages.create``) cannot be reached. The raised
                ``TypeError`` chains the original exception via ``__cause__``
                so the customer can debug their setup. We do NOT swallow the
                error — failure isolation is for the hot path; setup-time
                misuse should be loud.
        """
        module = type(client).__module__
        cls_name = type(client).__name__

        if module.startswith("anthropic"):
            self._validate_accessors(client, cls_name, [("messages", "create")])
            from token_sentinel.wrappers.anthropic import wrap_anthropic

            return wrap_anthropic(client, self)  # type: ignore[arg-type, return-value]

        if module.startswith("openai"):
            self._validate_accessors(
                client,
                cls_name,
                [
                    ("chat", "completions", "create"),
                    ("embeddings", "create"),
                ],
            )
            from token_sentinel.wrappers.openai import wrap_openai

            return wrap_openai(client, self)  # type: ignore[no-any-return]

        if module.startswith("google.genai") or module.startswith("google_genai"):
            # Gemini's ``models.generate_content`` / ``generate_content_stream``
            # may not be present on minimal mocks; the wrapper itself is
            # defensive about each method (``getattr(..., None)``). We only
            # require ``client.models`` to be reachable here.
            self._validate_accessors(client, cls_name, [("models",)])
            from token_sentinel.wrappers.gemini import wrap_gemini

            return wrap_gemini(client, self)  # type: ignore[no-any-return]

        # Note: legacy `vertexai` SDK is deprecated in favor of google-genai
        # with `vertexai=True`. Customers on the legacy SDK should migrate;
        # we don't ship a separate wrapper for it.

        # Cohere V2 : enterprise-RAG chat + embedding + rerank. We
        # support the ``cohere.ClientV2`` / ``cohere.AsyncClientV2``
        # surface only — the legacy ``cohere.Client`` (V1) is deprecated
        # in cohere>=5 per the V1.6 provider research and falls through
        # to the "Unsupported" branch below. As with voyageai we require
        # AT LEAST one of ``chat`` / ``embed`` / ``rerank`` to be callable
        # before delegating, so a partially-initialised mock / subclass
        # produces a clear TypeError instead of a silent no-op.
        if module.startswith("cohere") and cls_name in {"ClientV2", "AsyncClientV2"}:
            try:
                has_chat = callable(getattr(client, "chat", None))
                has_embed = callable(getattr(client, "embed", None))
                has_rerank = callable(getattr(client, "rerank", None))
            except Exception as e:
                raise TypeError(
                    f"Sentinel could not instrument {cls_name}: probing "
                    f"chat/embed/rerank raised {type(e).__name__}: {e}"
                ) from e
            if not (has_chat or has_embed or has_rerank):
                raise TypeError(
                    f"Sentinel could not instrument {cls_name}: none of "
                    "``chat``, ``embed``, or ``rerank`` is callable on this client."
                )
            from token_sentinel.wrappers.cohere import wrap_cohere

            return wrap_cohere(client, self)  # type: ignore[no-any-return]

        # Voyage AI: Anthropic's recommended embeddings provider. The SDK
        # ships ``voyageai.Client`` (sync) and ``voyageai.AsyncClient``
        # (async); both expose ``embed`` and ``rerank``. We require AT
        # LEAST one to be reachable so a partially-initialised mock /
        # subclass produces a clear TypeError instead of a silent no-op.
        if module.startswith("voyageai") and cls_name in {"Client", "AsyncClient"}:
            # Either ``embed`` or ``rerank`` is sufficient — voyageai's
            # API is "pick the operation you need"; we don't want to
            # refuse to wrap a customer's subclass that only exposes
            # one of the two.
            try:
                has_embed = callable(getattr(client, "embed", None))
                has_rerank = callable(getattr(client, "rerank", None))
            except Exception as e:
                raise TypeError(
                    f"Sentinel could not instrument {cls_name}: probing "
                    f"embed/rerank raised {type(e).__name__}: {e}"
                ) from e
            if not (has_embed or has_rerank):
                raise TypeError(
                    f"Sentinel could not instrument {cls_name}: neither "
                    "``embed`` nor ``rerank`` is callable on this client."
                )
            from token_sentinel.wrappers.voyage import wrap_voyage

            return wrap_voyage(client, self)  # type: ignore[no-any-return]

        # Replicate : non-token-priced image+video models. Both the
        # stable ``replicate`` package and the Stainless-generated
        # ``replicate-python-beta`` expose a ``Client`` class with
        # ``client.run`` / ``client.predictions.create`` / ``.get`` —
        # the ``module.startswith("replicate")`` test covers both. The
        # wrapper handles ``client.run`` missing defensively, so we only
        # require ``predictions`` to be reachable here.
        if module.startswith("replicate") and cls_name == "Client":
            self._validate_accessors(client, cls_name, [("predictions",)])
            from token_sentinel.wrappers.replicate import wrap_replicate

            return wrap_replicate(client, self)  # type: ignore[no-any-return]

        # Deepgram : STT provider — pre-recorded + live transcription.
        # The v7 ``deepgram-sdk`` exposes ``DeepgramClient`` /
        # ``AsyncDeepgramClient`` under the ``deepgram`` (Fern-generated
        # ``deepgram.client``) module. We require the ``listen.v1`` accessor
        # chain — anything else is a misconfigured or stub client.
        if module.startswith("deepgram") and cls_name in {"DeepgramClient", "AsyncDeepgramClient"}:
            self._validate_accessors(client, cls_name, [("listen", "v1")])
            from token_sentinel.wrappers.deepgram import wrap_deepgram

            return wrap_deepgram(client, self)  # type: ignore[no-any-return]

        # ElevenLabs : TTS provider — text-to-speech (the pair to
        # Deepgram's STT). The Fern-generated ``elevenlabs`` package exposes
        # ``ElevenLabs`` / ``AsyncElevenLabs`` under ``elevenlabs.client``.
        # We require the ``text_to_speech`` accessor — anything else is a
        # misconfigured or stub client. The wrapper itself attribute-checks
        # each TTS method so v1.x SDKs (which lack ``text_to_speech.stream``)
        # work alongside v2.x+ SDKs without a version sniff.
        if module.startswith("elevenlabs") and cls_name in {"ElevenLabs", "AsyncElevenLabs"}:
            self._validate_accessors(client, cls_name, [("text_to_speech",)])
            from token_sentinel.wrappers.elevenlabs import wrap_elevenlabs

            return wrap_elevenlabs(client, self)  # type: ignore[no-any-return]

        # Bedrock: a boto3 client whose service is bedrock-runtime. boto3
        # stuffs all clients under botocore.client, so we detect by class name.
        if "bedrock" in cls_name.lower() or module.startswith("botocore.client"):
            try:
                meta = getattr(client, "meta", None)
                service = getattr(meta, "service_model", None) if meta is not None else None
                service_name = getattr(service, "service_name", "") if service is not None else ""
            except Exception:
                service_name = ""
            if "bedrock" in cls_name.lower() or service_name.startswith("bedrock"):
                # Bedrock clients attach service operations dynamically. The
                # wrapper itself attribute-checks each method, so we only
                # require ``converse`` to exist as the Bedrock fingerprint —
                # if even that's missing, this isn't a usable bedrock-runtime
                # client. ``converse_stream`` is checked but missing is
                # tolerated (older botocore lacks it).
                self._validate_accessors(client, cls_name, [("converse",)])
                from token_sentinel.wrappers.bedrock import wrap_bedrock

                return wrap_bedrock(client, self)  # type: ignore[no-any-return]

        raise TypeError(
            f"Unsupported client type: {cls_name} from {module}. "
            "Sentinel supports Anthropic, OpenAI (+ OpenAI-compatible base_url), "
            "Google Gemini (which covers Vertex via vertexai=True), AWS "
            "Bedrock, Voyage AI, Cohere V2, Replicate, Deepgram, and ElevenLabs clients."
        )

    @staticmethod
    def _validate_accessors(client: Any, cls_name: str, paths: list[tuple[str, ...]]) -> None:
        """Validate that each attribute path on ``client`` is reachable.

        Used by :meth:`wrap` *before* delegating to a provider-specific
        wrapper, so a partially-initialised client (e.g., a property that
        throws until config is set) produces a clear ``TypeError`` instead
        of an opaque ``AttributeError`` from deep inside the wrapper module.

        Each path is a tuple of attribute names; e.g., ``("messages",
        "create")`` validates ``client.messages.create``. We don't *call*
        the accessors — just check that they can be reached.

        Raises:
            TypeError: with a descriptive message and ``__cause__`` set to
                the original exception when an attribute access fails.
        """
        for path in paths:
            target = client
            for attr in path:
                try:
                    target = getattr(target, attr)
                except AttributeError as e:
                    full = ".".join(path)
                    raise TypeError(
                        f"Sentinel could not instrument {cls_name}.{full}: {e}. "
                        f"Is this a fully-initialised {cls_name} client?"
                    ) from e
                except Exception as e:
                    full = ".".join(path)
                    raise TypeError(
                        f"Sentinel could not instrument {cls_name}: accessing "
                        f"{full} raised {type(e).__name__}: {e}"
                    ) from e

    def on_leak(self, handler: LeakHandler) -> LeakHandler:
        """Register a leak event handler. Usable as a decorator.

        Thread-safe: registrations from any thread interleave correctly with
        in-flight dispatches.
        """
        with self._handler_lock:
            self._handlers.append(handler)
        return handler

    # Rebrand alias — "token leak" → "token waste". Identical method object,
    # so ``sentinel.on_waste is sentinel.on_leak`` is True when accessed via
    # ``Sentinel.on_waste`` / ``Sentinel.on_leak`` (the class), and both
    # decorators register into the same ``_handlers`` list. See
    # ``events.WasteEvent`` / ``events.WasteDetected`` for the matching
    # type-level aliases. No DeprecationWarning — this is a brand alias,
    # not a deprecation; the ``on_leak`` name stays first-class.
    on_waste = on_leak

    def unregister(self, handler: LeakHandler) -> bool:
        """Remove a previously registered leak handler.

        Returns ``True`` if the handler was found and removed, ``False`` if
        it was never registered (or already removed). Thread-safe.

        Symmetric counterpart to :meth:`on_leak`. Useful for tests that need
        to clean up a handler, and for long-running processes that want to
        swap handlers without leaking references.
        """
        with self._handler_lock:
            try:
                self._handlers.remove(handler)
            except ValueError:
                return False
            return True

    def record_call(self, call: CallRecord) -> list[LeakEvent]:
        """Record a call and run rules. Returns any fired events.

        In ``mode='block'``, fires every registered handler for *every* event
        before raising ``LeakDetected`` once with the highest-confidence event.
        This is intentional: customers register handlers to ship every leak
        signal somewhere (Slack, Datadog, etc.); silently dropping the second
        event because the first triggered a raise would be a correctness bug.

        Concurrent ``record_call`` invocations on the same session can
        independently observe the same rule window and emit duplicate events.
        We de-duplicate by ``(event.type, event.rule, evidence_hash)`` within
        ``self._dedup_window_seconds`` per session — set
        ``dedup_window_seconds=0`` on construction to disable.

        Intervention Pack — policy enforcement:
            If a :class:`PolicyClient` is wired up (``policy_endpoint``
            configured), the cloud-pushed policy is consulted BEFORE the
            rule loop runs. If the policy says halt, this method raises
            :class:`KillSwitchActive`, :class:`BudgetExceeded`, or
            :class:`VelocityExceeded` — UNCONDITIONALLY, regardless of
            ``mode``. The customer who configures a policy endpoint is
            opting into mid-run halts; logging mode applies to the rule
            engine, not the policy plane.
        """
        # ---- Policy enforcement (Intervention Pack) ---------------
        # Performed BEFORE the rule loop so a policy violation aborts the
        # call without recording it through the rule machinery — keeps the
        # cloud's session aggregate consistent with the SDK's local view
        # (the rejected call doesn't count toward the next budget tick).
        #
        # All policy work is wrapped in a try/except below so a bug in the
        # policy code path NEVER crashes the user's call. The exceptions
        # listed in the ``raise`` re-list are the only ones that propagate
        # — those are the enforcement signals the customer opted into.
        self._enforce_policy(call)

        self.tracer.record(call)
        events: list[LeakEvent] = []
        session = self.tracer.session(call.session_id)
        for rule in self._rules:
            try:
                ev = rule.evaluate(session, project=self.project)
            except Exception:
                continue
            if ev is None:
                continue
            # LOW-5: enforce the LeakEvent confidence contract (0.0–1.0).
            # All current rules cap via min(...) but a future rule author
            # could miss this; clamp defensively at the boundary.
            if ev.confidence < 0.0 or ev.confidence > 1.0:
                ev.confidence = max(0.0, min(1.0, ev.confidence))
            if ev.confidence < self.min_confidence:
                continue
            # Tag-based Chargeback: propagate the originating
            # CallRecord's tags onto the LeakEvent. Rules don't know about
            # tags (they shouldn't — keeping the rule API small), so we
            # stamp here at the boundary. Only set when the call carried
            # tags AND the event author didn't already set them (defensive
            # — no current rule sets ``ev.tags`` but a future rule might
            # want to add its own metadata).
            if call.tags and not ev.tags:
                ev.tags = dict(call.tags)
            events.append(ev)

        # Policy trackers update AFTER the rule loop completes so an
        # oversized call doesn't double-count: the budget check above
        # used the speculative "would this call push us over?" form, the
        # update below records the actual call burn for the NEXT call's
        # check. If the rule loop raises, the call still happened and the
        # burn still counts — but the rule loop is exception-safe
        # (per-rule try/except), so this is a defensive comment on the
        # contract rather than a concern for current code.
        self._update_policy_trackers(call)

        # MED-4: filter out duplicates within the configured window. Done
        # AFTER rule evaluation so the rule pipeline is unchanged, and BEFORE
        # running handlers so duplicate signals never reach the customer.
        if self._dedup_window_seconds > 0 and events:
            events = self._filter_duplicate_events(call.session_id, events)

        # Run handlers for ALL events first so customer handlers see every
        # leak signal, regardless of mode.
        for ev in events:
            self._run_handlers(ev)

        # Then, in block mode, raise exactly once with the highest-confidence
        # event. Tiebreak: first event in iteration order (which is rule
        # registration order).
        if self.mode == "block" and events:
            chosen = self._highest_confidence(events)
            raise LeakDetected(chosen)

        return events

    # ---------------------------------------------------------------------
    # MED-4: event de-duplication helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _event_dedup_key(event: LeakEvent) -> str:
        """Build a stable key identifying an event for dedup purposes.

        Same ``(type, rule, evidence)`` ⇒ same key. Evidence is JSON-serialised
        with ``sort_keys=True`` so dict-ordering doesn't break equivalence;
        ``default=str`` makes datetimes / arbitrary objects hashable. We
        truncate the SHA-256 to 16 hex chars (64 bits) — collision probability
        is negligible at the per-session scale we operate at.
        """
        try:
            payload = json.dumps(event.evidence, sort_keys=True, default=str)
        except Exception:
            # Last-resort fallback if evidence is somehow unrepresentable —
            # build a key from its repr so we still dedup self-equivalent
            # events instead of crashing the dedup path.
            payload = repr(event.evidence)
        digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"{event.type}:{event.rule}:{digest}"

    def _filter_duplicate_events(self, session_id: str, events: list[LeakEvent]) -> list[LeakEvent]:
        """Return ``events`` with duplicates (within the window) removed.

        Uses ``self._dedup_lock`` to serialize lookup + insert. Lazy GC of
        old session entries fires on every call (cheap — O(N_sessions) walk).
        """
        window = self._dedup_window_seconds
        # The cleanup horizon is intentionally larger than the dedup window
        # so a session that fires near the boundary doesn't lose its
        # bookkeeping mid-flight.
        cleanup_horizon = window * 4
        now = time.monotonic()
        kept: list[LeakEvent] = []

        with self._dedup_lock:
            # Lazy cleanup of old per-session dicts. We drop a session's
            # dict entirely when its newest entry is older than
            # ``cleanup_horizon``. Within a session we don't mutate keys
            # here — they're cheap, bounded by the rule count, and naturally
            # expire on the timestamp comparison below.
            stale_sessions = [
                sid
                for sid, keymap in self._recent_event_keys.items()
                if not keymap or (now - max(keymap.values())) > cleanup_horizon
            ]
            for sid in stale_sessions:
                del self._recent_event_keys[sid]

            session_keys = self._recent_event_keys.get(session_id)
            if session_keys is None:
                session_keys = {}
                self._recent_event_keys[session_id] = session_keys

            for ev in events:
                key = self._event_dedup_key(ev)
                last = session_keys.get(key)
                if last is not None and (now - last) <= window:
                    # Same event seen recently for this session — drop it.
                    continue
                session_keys[key] = now
                kept.append(ev)

        return kept

    def _run_handlers(self, event: LeakEvent) -> None:
        """Invoke every registered handler for ``event``.

        Snapshots the handler list under the lock, then runs handlers
        *without* the lock held — a handler may legitimately call
        :meth:`on_leak` to register a new handler (or :meth:`unregister`
        to remove one); holding the lock during invocation would deadlock
        re-entrant calls. Handler exceptions are swallowed so one bad
        handler cannot mute the rest, and so a thrown handler cannot kill
        the user's LLM call. ``BaseException`` (e.g. ``KeyboardInterrupt``,
        ``SystemExit``) is intentionally NOT caught — those propagate.
        """
        with self._handler_lock:
            snapshot = list(self._handlers)
        for h in snapshot:
            try:
                h(event)
            except Exception:
                pass

        # Forward to the cloud sink (if configured). The sink's enqueue is
        # non-blocking and never raises into user code, but we still wrap in
        # try/except defensively — failure-isolation discipline (mirrors the
        # wrapper pattern). LeakDetected from a customer handler is not a
        # concern here: ``_run_handlers`` is called from ``record_call``
        # before block-mode raise, so we never see it propagate at this
        # point. If a future caller threads it through, ``Exception``
        # catches non-BaseException only — KeyboardInterrupt / SystemExit
        # propagate as elsewhere in the codebase.
        if self._cloud_sink is not None:
            try:
                self._cloud_sink.enqueue(event)
            except Exception:
                pass

    @staticmethod
    def _highest_confidence(events: list[LeakEvent]) -> LeakEvent:
        """Return the event with the highest ``confidence``.

        Iteration order is preserved on ties — Python's ``max`` keeps the
        first occurrence when multiple items tie on the key, which gives
        the documented "first iteration order" tiebreak behaviour.
        """
        return max(events, key=lambda e: e.confidence)

    def _dispatch(self, event: LeakEvent) -> None:
        """Backwards-compatible thin wrapper.

        Older callers / tests may invoke ``_dispatch`` directly with a single
        event. We preserve the historical contract: run handlers, then in
        block mode raise ``LeakDetected`` for that single event. New
        ``record_call`` does NOT route through this method — it calls
        :meth:`_run_handlers` per event and raises once at the end.
        """
        self._run_handlers(event)
        if self.mode == "block":
            raise LeakDetected(event)

    # ---------------------------------------------------------------------
    # Intervention Pack — policy enforcement helpers
    # ---------------------------------------------------------------------

    def _enforce_policy(self, call: CallRecord) -> None:
        """Consult the policy and raise if the call must be halted.

        Raises:
            KillSwitchActive: when the operator hit the dashboard kill-switch.
            BudgetExceeded: when this call would push the session past
                ``policy.budget_usd_per_session``.
            VelocityExceeded: when adding this call's tokens would exceed
                ``policy.max_tokens_per_min`` over the rolling 1-minute window.

        Defensive isolation: any exception from the PolicyClient itself
        (an unexpected bug in policy parsing, a thread that died, etc.) is
        swallowed and treated as "no policy" — the user's call goes through.
        ONLY the three enforcement exceptions above propagate.
        """
        client = self._policy_client
        if client is None:
            # Mark the session active even when no policy is configured? No —
            # marking only matters for poll cadence, which is irrelevant
            # when there's no client.
            return

        # Mark active on every call so the daemon knows to use the active
        # poll cadence. Cheap (set membership). The wrapper layer doesn't
        # currently emit "session ended" hooks, so sessions stay active for
        # the SDK's lifetime; the LRU cap on ``_session_burn`` keeps this
        # bounded.
        try:
            client.mark_session_active(call.session_id)
        except Exception:
            # A bug in mark_session_active should not crash the call path.
            pass

        try:
            policy = client.current()
        except Exception:
            policy = None

        if policy is None:
            return

        # Kill switch first — it's the most aggressive halt and short-circuits
        # the other checks. If the operator wants the agent stopped, we don't
        # waste cycles checking budgets.
        if policy.kill_switch_active:
            event = LeakEvent(
                type="kill_switch",
                confidence=1.0,
                project=self.project,
                session_id=call.session_id,
                rule="v0.6.policy",
                evidence={
                    "policy_version": policy.policy_version,
                    "reason": "operator_kill_switch",
                },
                estimated_burn=0.0,
                suggested_action="halt_immediately",
                raised_at=datetime.now(timezone.utc),
            )
            raise KillSwitchActive(event, policy=policy)

        # Budget check. We need to know what this call would *cost* before
        # it runs — same heuristic as ``rules/tool_loop.py``'s burn estimator
        # so the SDK side and the cloud side agree on the units.
        if policy.budget_usd_per_session is not None:
            next_call_burn = _estimate_call_burn_usd(call)
            with self._policy_lock:
                current_burn = self._session_burn.get(call.session_id, 0.0)
            projected = current_burn + next_call_burn
            if projected > policy.budget_usd_per_session:
                event = LeakEvent(
                    type="budget_exceeded",
                    confidence=1.0,
                    project=self.project,
                    session_id=call.session_id,
                    rule="v0.6.policy.budget",
                    evidence={
                        "policy_version": policy.policy_version,
                        "current_usd": round(current_burn, 6),
                        "next_call_usd": round(next_call_burn, 6),
                        "budget_usd": policy.budget_usd_per_session,
                    },
                    estimated_burn=round(projected, 6),
                    suggested_action="halt_session",
                    raised_at=datetime.now(timezone.utc),
                )
                raise BudgetExceeded(
                    event,
                    policy=policy,
                    session_id=call.session_id,
                    current_usd=current_burn,
                    next_call_usd=next_call_burn,
                    budget_usd=policy.budget_usd_per_session,
                )

        # Velocity check — project-wide, rolling 1-minute window. The
        # velocity layer for  is absolute-ceiling-only per the founder
        # decision (no per-customer baseline); a single hard cap on
        # tokens/min. We compute the current minute's tokens INCLUDING the
        # call we're about to make, so a single oversized call past the cap
        # is also caught.
        if policy.max_tokens_per_min is not None:
            now = time.monotonic()
            current_window_tokens = self._current_minute_tokens(now)
            this_call_tokens = max(0, call.prompt_tokens) + max(0, call.completion_tokens)
            projected_tokens = current_window_tokens + this_call_tokens
            if projected_tokens > policy.max_tokens_per_min:
                event = LeakEvent(
                    type="velocity_exceeded",
                    confidence=1.0,
                    project=self.project,
                    session_id=call.session_id,
                    rule="v0.6.policy.velocity",
                    evidence={
                        "policy_version": policy.policy_version,
                        "current_tokens_per_min": current_window_tokens,
                        "this_call_tokens": this_call_tokens,
                        "max_tokens_per_min": policy.max_tokens_per_min,
                    },
                    estimated_burn=round(projected_tokens * _BURN_USD_PER_TOKEN, 6),
                    suggested_action="halt_or_throttle",
                    raised_at=datetime.now(timezone.utc),
                )
                raise VelocityExceeded(
                    event,
                    policy=policy,
                    current_tokens_per_min=projected_tokens,
                    max_tokens_per_min=policy.max_tokens_per_min,
                )

    def _current_minute_tokens(self, now_monotonic: float) -> int:
        """Sum the rolling 1-minute window. Drops entries older than 60s.

        Called from the policy-enforcement hot path. Should be cheap; the
        deque is bounded by traffic rate (e.g., a 10 calls/sec agent has at
        most 600 entries in the window).
        """
        cutoff = now_monotonic - _VELOCITY_WINDOW_SECONDS
        with self._policy_lock:
            window = self._tokens_minute_window
            # Drop expired entries from the left.
            while window and window[0][0] < cutoff:
                window.popleft()
            return sum(tokens for _, tokens in window)

    def _update_policy_trackers(self, call: CallRecord) -> None:
        """Record this call's burn + tokens for future policy checks.

        Bounded by ``max_sessions``: when the per-session burn dict gets
        too large, the LRU entry is evicted (same pattern as
        :class:`Tracer`). The 1-minute deque is naturally bounded by the
        rolling window cleanup performed during reads.
        """
        if self._policy_client is None:
            return

        burn_usd = _estimate_call_burn_usd(call)
        tokens = max(0, call.prompt_tokens) + max(0, call.completion_tokens)
        now = time.monotonic()

        with self._policy_lock:
            existing = self._session_burn.get(call.session_id)
            if existing is None:
                self._session_burn[call.session_id] = burn_usd
                if (
                    self._max_sessions_for_burn is not None
                    and len(self._session_burn) > self._max_sessions_for_burn
                ):
                    self._session_burn.popitem(last=False)
            else:
                # Aggregate, then refresh LRU position.
                self._session_burn[call.session_id] = existing + burn_usd
                self._session_burn.move_to_end(call.session_id)

            self._tokens_minute_window.append((now, tokens))
            # Lazy GC of the velocity window so a long-running idle agent
            # doesn't hold a stale window in memory. The next read will do
            # the same cleanup, but proactively pruning here keeps memory
            # bounded under steady traffic.
            cutoff = now - _VELOCITY_WINDOW_SECONDS
            while self._tokens_minute_window and self._tokens_minute_window[0][0] < cutoff:
                self._tokens_minute_window.popleft()

    # ---------------------------------------------------------------------
    # Cloud sink lifecycle
    # ---------------------------------------------------------------------

    def close(self, timeout: float = 5.0) -> bool:
        """Flush the cloud sink and shut down its background thread.

        Returns ``True`` on clean shutdown, ``False`` if the timeout was
        exceeded (in which case a ``RuntimeWarning`` is also emitted by the
        sink).

        This method is OPTIONAL for short-lived processes — the cloud sink
        runs as a daemon thread and process exit will reap it. It is
        REQUIRED for long-running agents that want every event delivered
        before they go away (e.g. CI runs, batch jobs, deliberate shutdown
        of a worker process).

        Calling ``close`` on a Sentinel that was never configured with a
        cloud sink is a no-op and returns ``True``.

        : also stops the PolicyClient daemon thread when configured.
        Both shutdowns share the timeout — close() returns ``False`` if
        either subsystem timed out.
        """
        sink_ok = True
        if self._cloud_sink is not None:
            try:
                sink_ok = bool(self._cloud_sink.close(timeout=timeout))
            except Exception:
                # Defence-in-depth: ``CloudSink.close`` is documented to
                # never raise, but a future refactor or a buggy subclass
                # could. Treat any failure as an unclean shutdown and
                # continue.
                sink_ok = False

        policy_ok = True
        if self._policy_client is not None:
            try:
                policy_ok = bool(self._policy_client.stop(timeout=timeout))
            except Exception:
                policy_ok = False

        return sink_ok and policy_ok
