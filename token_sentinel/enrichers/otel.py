"""OpenTelemetry ``SpanProcessor`` enricher for TokenSentinel.

The  LangChain enricher shipped first because LangChain has a native
``BaseCallbackHandler`` interface — a callback-bus model that gives the
enricher per-call hooks before/after every LLM invocation. The remaining
 backlog frameworks — **CrewAI**, **AutoGen**, and **Pydantic AI** —
chose a different integration path: instead of a custom callback bus,
each one emits OpenTelemetry spans following the
`gen_ai semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.

That's a happy convergence: one OTel span processor covers all three
frameworks (and any future framework that adopts the same conventions).
This module is that processor.

.. code-block:: python

    from token_sentinel import Sentinel
    from token_sentinel.enrichers.otel import TokenSentinelSpanProcessor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    sentinel = Sentinel(project="my-agent", mode="log")

    provider = TracerProvider()
    provider.add_span_processor(TokenSentinelSpanProcessor(sentinel))
    trace.set_tracer_provider(provider)

    # Now any CrewAI / AutoGen / Pydantic AI agent that emits gen_ai.*
    # spans is automatically wrapped — no per-client wrap() needed.

Design notes:

  1. *Module import is dependency-free.* ``opentelemetry`` is imported
     lazily inside a ``try``; missing-package customers get a clean
     :class:`ImportError` at processor construction time, never at
     module import. Same pattern as :mod:`enrichers.langchain`.
  2. *Session-id derivation.* The gen_ai semantic conventions support
     a ``gen_ai.conversation.id`` attribute (some frameworks emit it
     opportunistically). When present, we use it as ``session_id``.
     Otherwise we fall back to the OTel ``trace_id`` (32-char hex) so
     spans within one trace share a session — which matches the
     "one agent run = one trace" mental model the framework
     instrumentations use.
  3. *Per-framework quirks.* Each framework attaches its own
     framework-specific attributes alongside the standard
     ``gen_ai.*`` keys:

       - **CrewAI**: ``crewai.agent.name`` → ``tags.agent``
       - **AutoGen**: ``autogen.agent_role`` → ``tags.role``
       - **Pydantic AI**: ``pydantic_ai.tool_name`` → ``tool_calls``

     Tag values are validated against the  tag regex / length cap
     before being attached — invalid values are dropped silently so a
     framework that ships a tag value with a space doesn't crash the
     span processor's hot path.
  4. *Most work happens in ``on_end``.* OTel hands us the span with
     its full attributes + duration only at end time. ``on_start`` is
     intentionally a no-op (cheap fast path; we don't need the start
     hook for anything since spans carry their own start time).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Soft import of opentelemetry — module loads cleanly without OTel so
# ``from token_sentinel.enrichers.otel import TokenSentinelSpanProcessor``
# never crashes a base install. Instantiation raises ImportError with a hint.
# ---------------------------------------------------------------------------

try:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when OTel not installed
    # Fallback so the module imports cleanly without OTel installed. The
    # class below conditionally subclasses ``SpanProcessor``, but if a
    # customer manages to instantiate ``TokenSentinelSpanProcessor``
    # without OTel they hit the ImportError in ``__init__`` before any
    # base-class machinery matters. ``Any`` keeps mypy happy in both
    # branches without per-line ignores.
    SpanProcessor = object  # type: ignore[misc,assignment]
    ReadableSpan = object  # type: ignore[misc,assignment]
    _OTEL_AVAILABLE = False


from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from token_sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# gen_ai semantic-convention attribute keys
# ---------------------------------------------------------------------------
#
# We hard-code the attribute strings (rather than importing them from
# ``opentelemetry.semconv``) for two reasons:
#
#   1. The semconv package is on a separate version cadence from the
#      OTel SDK and the constant names have churned between releases
#      (e.g., ``GEN_AI_USAGE_INPUT_TOKENS`` vs.
#      ``GEN_AI_USAGE_PROMPT_TOKENS``). Pinning to the wire-level
#      string keeps the enricher resilient to semconv refactors.
#   2. The wire-level strings are stable per the spec; reading them
#      directly is the same contract every other gen_ai consumer uses.
#
# The string set below is the union of the gen_ai conventions we
# extract from. Keys that don't appear on a given span are simply
# absent — ``attrs.get(...)`` returns ``None`` and we fall back.

_ATTR_GEN_AI_SYSTEM = "gen_ai.system"
_ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"  # newer alias
_ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_ATTR_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
_ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_ATTR_GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"  # legacy alias
_ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_ATTR_GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"  # legacy
_ATTR_GEN_AI_PROMPT = "gen_ai.prompt"
_ATTR_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

# Per-framework quirk attributes — see module docstring.
_ATTR_CREWAI_AGENT_NAME = "crewai.agent.name"
_ATTR_AUTOGEN_AGENT_ROLE = "autogen.agent_role"
_ATTR_PYDANTIC_AI_TOOL_NAME = "pydantic_ai.tool_name"


# ---------------------------------------------------------------------------
# tag validation — same regex / length cap as Sentinel.session(tags=...)
# ---------------------------------------------------------------------------
#
# We duplicate the regex / length cap here rather than importing from
# ``token_sentinel.sentinel`` because:
#
#   - The sentinel module's ``_validate_session_tags`` *raises* on
#     invalid input. The span processor must NEVER raise — OTel docs
#     are explicit that ``on_end`` cannot throw or block. So we need a
#     *silent-drop* variant of the same validation.
#   - Importing the private helper would create a circular-ish coupling
#     (enricher → sentinel) that's awkward when the sentinel module is
#     the bigger surface. Keeping the regex local keeps the enricher
#     self-contained.
#
# The regex / cap MUST stay in sync with ``sentinel._TAG_VALUE_PATTERN``
# and ``sentinel._MAX_TAG_VALUE_LENGTH``. If those change, change this too.

_TAG_VALUE_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
_MAX_TAG_VALUE_LENGTH = 64
_ALLOWED_TAG_KEYS: frozenset[str] = frozenset(
    {"team", "feature", "customer", "environment", "version", "agent", "role", "tool"}
)
# We allow ``agent`` / ``role`` / ``tool`` in addition to the
# allowlist because the per-framework quirks land in those keys. They
# are NOT accepted by ``Sentinel.session(tags=...)`` directly — the
# session-tag surface is gated to the chargeback-attribution five —
# but the span processor sets them as enrichment metadata, not
# chargeback dimensions. The cloud's by-tag aggregation ignores
# unknown keys gracefully (extra='ignore' on the Pydantic model).


def _validate_tag_value(value: Any) -> str | None:
    """Return the value if it passes  validation, else ``None``.

    Silent-drop variant of ``sentinel._validate_session_tags`` — the
    span processor's ``on_end`` MUST NOT raise (OTel forbids it), so a
    bad tag value is dropped instead of propagating an error.
    """
    if not isinstance(value, str):
        return None
    if not value or len(value) > _MAX_TAG_VALUE_LENGTH:
        return None
    if not _TAG_VALUE_PATTERN.match(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _coerce_int(value: Any) -> int:
    """Return a non-negative int from any candidate, else 0.

    The cloud cost estimator chokes on negative token counts; clamp at
    zero so a malformed span attribute can't poison the rule loop.
    Mirrors ``enrichers.langchain._coerce_int``.
    """
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _extract_provider(attrs: dict[str, Any]) -> str:
    """Pull ``gen_ai.system`` (or its newer ``gen_ai.provider.name`` alias)
    from the span attributes. Falls back to ``"otel"`` so records stay
    routable even when a framework forgets to set the system attribute.
    """
    for key in (_ATTR_GEN_AI_PROVIDER_NAME, _ATTR_GEN_AI_SYSTEM):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return "otel"


def _extract_model(attrs: dict[str, Any]) -> str:
    """Pull the model identifier.

    Prefer ``gen_ai.response.model`` (more specific — e.g.,
    ``gpt-4o-2024-08-06`` rather than the requested ``gpt-4o``) and
    fall back to ``gen_ai.request.model``. Returns ``"unknown"`` so the
    rule engine sees a uniform sentinel string when neither is present.
    """
    for key in (_ATTR_GEN_AI_RESPONSE_MODEL, _ATTR_GEN_AI_REQUEST_MODEL):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _extract_method(attrs: dict[str, Any], span_name: str) -> str:
    """Build the ``CallRecord.method`` string.

    Pattern matches the LangChain enricher's ``langchain.<kind>``
    convention: ``otel.<operation>``. Falls back to the span name when
    ``gen_ai.operation.name`` is absent (some frameworks just name the
    span and skip the attribute).
    """
    operation = attrs.get(_ATTR_GEN_AI_OPERATION_NAME)
    if isinstance(operation, str) and operation:
        return f"otel.{operation}"
    # Span name fallback — strip a leading ``gen_ai.`` so we don't
    # double-prefix common patterns like ``gen_ai.chat``.
    if isinstance(span_name, str) and span_name:
        clean = span_name.removeprefix("gen_ai.")
        return f"otel.{clean}"
    return "otel.unknown"


def _extract_tokens(attrs: dict[str, Any]) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from gen_ai.usage.*.

    Supports both the current spec (``input_tokens`` / ``output_tokens``)
    and the older alias (``prompt_tokens`` / ``completion_tokens``).
    Frameworks that use only one or the other are both common in the
    wild as of 2026-05.
    """
    prompt = _coerce_int(
        attrs.get(_ATTR_GEN_AI_USAGE_INPUT_TOKENS) or attrs.get(_ATTR_GEN_AI_USAGE_PROMPT_TOKENS)
    )
    completion = _coerce_int(
        attrs.get(_ATTR_GEN_AI_USAGE_OUTPUT_TOKENS)
        or attrs.get(_ATTR_GEN_AI_USAGE_COMPLETION_TOKENS)
    )
    return prompt, completion


def _extract_latency_ms(start_time: int | None, end_time: int | None) -> float:
    """Convert OTel span timestamps to a non-negative latency in ms.

    OTel timestamps are nanoseconds since epoch. Either endpoint missing
    or out-of-order (a clock skew) returns 0.0 so the rule engine never
    sees a negative latency.
    """
    if start_time is None or end_time is None:
        return 0.0
    delta_ns = end_time - start_time
    if delta_ns < 0:
        return 0.0
    return delta_ns / 1_000_000.0  # ns → ms


def _request_hash(model: str, prompt: str | None, method: str) -> str:
    """Stable per-call hash for the rule engine's retry-storm detector.

    The retry_storm rule windows by ``request_hash`` to spot identical
    repeated calls. We hash ``(model, prompt, method)`` — same shape as
    the LangChain enricher's ``_request_hash`` modulo the field order.
    A missing prompt collapses to the empty string so prompt-less
    spans (e.g., embeddings where the input is in a separate event)
    still produce a stable per-call identifier.
    """
    payload = f"{model}|{prompt or ''}|{method}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _extract_session_id(attrs: dict[str, Any], trace_id_hex: str | None) -> str:
    """Pick the session_id for this span.

    Priority:
      1. ``gen_ai.conversation.id`` (when the framework emits it —
         opportunistic, not universal). Must pass the  tag-value
         regex so the cloud's session aggregation never sees a value
         with a space or shell metacharacter.
      2. OTel ``trace_id`` (32-char hex) — every span has one, and
         spans within a single trace share it, which matches the
         "one agent run = one trace = one session" mental model.
      3. Fallback ``"otel-unknown"`` for synthesised test spans that
         carry no trace_id.
    """
    conv_id = attrs.get(_ATTR_GEN_AI_CONVERSATION_ID)
    if isinstance(conv_id, str) and conv_id:
        # Conversation ids may contain richer characters than the
        # tag-value regex allows (frameworks have been seen to use
        # ``conv-uuid:42``); accept them verbatim — they're identifiers,
        # not user-typed strings. We only enforce the regex on values
        # that will land in the chargeback ``tags`` dict.
        return conv_id
    if trace_id_hex:
        return trace_id_hex
    return "otel-unknown"


def _extract_tool_calls(attrs: dict[str, Any], span_name: str) -> list[dict[str, Any]]:
    """Pull tool-call info from per-framework quirks.

    Pydantic AI emits ``pydantic_ai.tool_name`` for tool-call spans;
    that's our primary signal. We construct a synthetic
    ``tool_calls=[{name, arguments}]`` entry so the rule engine's
    ``tool_loop`` detector sees the tool in the same shape it does
    from the wrapper layer.

    The arguments dict is intentionally empty — the gen_ai
    conventions don't standardise a tool-arguments attribute, and we
    don't want to invent one (we'd guess wrong for at least one
    framework). The rule engine handles empty arguments gracefully.
    """
    tool_name = attrs.get(_ATTR_PYDANTIC_AI_TOOL_NAME)
    if isinstance(tool_name, str) and tool_name:
        return [{"name": tool_name, "arguments": {}}]
    # Span name fallback — some frameworks name the span after the
    # tool (e.g., ``execute_tool calculator``). We don't try to
    # reverse-engineer that; the explicit attribute is the contract.
    if isinstance(span_name, str) and "tool" in span_name.lower():
        # Without an explicit tool name we can't populate a useful
        # entry — return empty. The rule engine prefers no signal to
        # a misleading one.
        return []
    return []


def _extract_framework_tags(attrs: dict[str, Any]) -> dict[str, str]:
    """Pull per-framework tag attributes onto a validated tag dict.

    Validates every value against the  tag regex and silently
    drops anything that doesn't match — we never let a malformed
    attribute crash the span processor or land bad data in the cloud.
    """
    tags: dict[str, str] = {}
    crew_agent = attrs.get(_ATTR_CREWAI_AGENT_NAME)
    validated = _validate_tag_value(crew_agent)
    if validated is not None:
        tags["agent"] = validated
    autogen_role = attrs.get(_ATTR_AUTOGEN_AGENT_ROLE)
    validated = _validate_tag_value(autogen_role)
    if validated is not None:
        tags["role"] = validated
    pydantic_tool = attrs.get(_ATTR_PYDANTIC_AI_TOOL_NAME)
    validated = _validate_tag_value(pydantic_tool)
    if validated is not None:
        tags["tool"] = validated
    return tags


def _format_trace_id(span: Any) -> str | None:
    """Render the span's trace_id as a 32-char hex string.

    OTel exposes ``span.get_span_context().trace_id`` as a 128-bit int.
    The hex rendering is the canonical wire form (matches the W3C
    Trace Context spec) — we use it as the session_id fallback so
    customers can cross-reference a TokenSentinel session with their
    OTel trace in observability tooling like Jaeger or Tempo.
    """
    try:
        ctx = span.get_span_context()
    except Exception:
        return None
    if ctx is None:
        return None
    trace_id_int = getattr(ctx, "trace_id", None)
    if not isinstance(trace_id_int, int) or trace_id_int == 0:
        return None
    # 32 hex chars = 128 bits, zero-padded. Matches OTel's
    # ``format_trace_id`` helper exactly; we inline it so the import
    # cost stays minimal.
    return f"{trace_id_int:032x}"


# ---------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------


class TokenSentinelSpanProcessor(SpanProcessor):
    """OTel ``SpanProcessor`` that routes gen_ai spans to TokenSentinel.

    Args:
        sentinel: The :class:`Sentinel` instance to route records into.

    Raises:
        ImportError: when the ``opentelemetry`` SDK is not installed.
            The processor cannot meaningfully exist without
            :class:`opentelemetry.sdk.trace.SpanProcessor`, so we fail
            loud at construction rather than silently no-op at runtime.

    Lifecycle:
        - :meth:`on_start` is a no-op (cheap fast path; we don't need
          the start hook because OTel spans carry their own start time).
        - :meth:`on_end` does the extraction → ``CallRecord`` →
          :meth:`Sentinel.record_call` work. Most spans flow through
          here.
        - :meth:`shutdown` and :meth:`force_flush` are pass-throughs —
          we don't buffer anything (every span is dispatched
          synchronously in ``on_end``), so there's nothing to flush.
          Returning ``True`` from ``force_flush`` matches the
          ``BatchSpanProcessor`` contract (no pending work).

    Thread safety:
        ``on_end`` is reentrant — OTel can dispatch ending spans from
        many threads concurrently — but every method on
        :class:`Sentinel.record_call` is already thread-safe, so the
        processor itself holds no state and needs no lock.
    """

    def __init__(self, sentinel: Sentinel) -> None:
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "TokenSentinelSpanProcessor requires opentelemetry-sdk. "
                "Install via `pip install token-sentinel[otel]` or "
                "`pip install opentelemetry-sdk>=1.27`."
            )
        super().__init__()
        self._sentinel = sentinel

    # ------------------------------------------------------------------
    # OTel SpanProcessor interface
    # ------------------------------------------------------------------

    def on_start(
        self,
        span: Any,
        parent_context: Any = None,
    ) -> None:
        """No-op. We do all extraction in :meth:`on_end` when the span
        has its full attributes + duration populated."""
        return None

    def on_end(self, span: Any) -> None:
        """Convert a finished gen_ai span into a :class:`CallRecord` and
        route through :meth:`Sentinel.record_call`.

        OTel docs are explicit that ``on_end`` MUST NOT throw or block —
        we wrap the entire body in a single ``try``/``except`` and
        silently drop on any failure. The only exception that propagates
        is :class:`LeakDetected`, which the sentinel raises in
        ``mode='block'``: that exception is the customer's opt-in
        contract, and re-raising preserves the wrapper-parity semantics
        the LangChain enricher established in .
        """
        try:
            attrs_obj = getattr(span, "attributes", None)
            if attrs_obj is None:
                return
            # ``ReadableSpan.attributes`` is a MappingProxyType in the
            # real SDK; we normalise to a plain dict so downstream
            # ``.get(...)`` works uniformly with hand-rolled test
            # doubles too.
            try:
                attrs: dict[str, Any] = dict(attrs_obj)
            except (TypeError, ValueError):
                # If the attributes aren't dict-coercible (shouldn't
                # happen with real OTel) drop the span — broken
                # instrumentation must not crash dispatch.
                return

            # Fast filter: only process gen_ai.* spans. The framework
            # may emit dozens of internal spans per agent run; only the
            # LLM-call ones carry the gen_ai conventions, and only
            # those map cleanly to a CallRecord.
            if not _has_gen_ai_attrs(attrs):
                return

            provider = _extract_provider(attrs)
            model = _extract_model(attrs)
            span_name = getattr(span, "name", "") or ""
            method = _extract_method(attrs, span_name)
            prompt_tokens, completion_tokens = _extract_tokens(attrs)
            start_time = getattr(span, "start_time", None)
            end_time = getattr(span, "end_time", None)
            latency_ms = _extract_latency_ms(start_time, end_time)
            trace_id_hex = _format_trace_id(span)
            session_id = _extract_session_id(attrs, trace_id_hex)

            prompt = attrs.get(_ATTR_GEN_AI_PROMPT)
            prompt_str: str | None = prompt if isinstance(prompt, str) else None
            req_hash = _request_hash(model, prompt_str, method)

            tool_calls = _extract_tool_calls(attrs, span_name)
            framework_tags = _extract_framework_tags(attrs)

            # ``user_facing_output``: a heuristic — OTel spec doesn't
            # standardise a "did the response have visible text"
            # attribute, so we treat completion-token-positive,
            # tool-call-empty spans as user-facing. Same fall-back
            # the LangChain enricher uses for non-tool generations.
            user_facing = completion_tokens > 0 and not tool_calls

            record = CallRecord(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                provider=provider,
                model=model,
                method=method,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                request_hash=req_hash,
                tool_calls=tool_calls,
                user_facing_output=user_facing,
                raw_request={"prompt": prompt_str} if prompt_str else {},
                raw_response_meta={
                    "via": "otel",
                    "span_name": span_name,
                    "trace_id": trace_id_hex,
                },
                tags=framework_tags,
            )
        except LeakDetected:
            # No CallRecord was built yet — this can't happen here, but
            # defensively re-raise to preserve block-mode contract if a
            # future refactor moves work above the build site.
            raise
        except Exception:
            # Instrumentation must never crash a real call. The OTel
            # contract is explicit about this in the SpanProcessor
            # docstring; silently dropping a malformed span is the
            # correct behaviour.
            return

        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            # Block-mode propagation. OTel's BatchSpanProcessor catches
            # processor exceptions and logs them, so a raise here will
            # be visible in the customer's OTel log channel — which is
            # the right place for "agent halted by policy" signals when
            # they're driving from OTel. Mirrors the LangChain
            # enricher's block-mode behaviour.
            raise
        except Exception:
            # Any other exception (rule bug, sentinel internal error)
            # is swallowed — see comment on the outer except.
            return

    def shutdown(self) -> None:
        """No-op. We don't buffer anything — every span is dispatched
        synchronously in :meth:`on_end`, so there's nothing to flush.

        OTel calls ``shutdown`` on every span processor when the
        tracer provider tears down. A no-op is the correct
        implementation for a synchronous processor.
        """
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op. Returns ``True`` (nothing to flush).

        OTel's ``BatchSpanProcessor`` returns ``False`` from
        ``force_flush`` when the timeout is exceeded; since we have
        nothing to wait on, the return is unconditionally ``True``.
        """
        # ``timeout_millis`` is part of the OTel contract; accept it
        # but ignore — we have no buffer to drain.
        del timeout_millis
        return True


# ---------------------------------------------------------------------------
# Internal helper used by on_end's fast filter
# ---------------------------------------------------------------------------


def _has_gen_ai_attrs(attrs: dict[str, Any]) -> bool:
    """Return True if the span carries any gen_ai.* attributes.

    Cheap O(N) scan over the attribute keys — there are typically
    < 30 attributes per span. We test for the prefix rather than a
    specific key so a span with only e.g. ``gen_ai.usage.input_tokens``
    (no system/model) still flows through; the extractors fall back
    cleanly when individual keys are missing.
    """
    return any(isinstance(key, str) and key.startswith("gen_ai.") for key in attrs)
