"""Tests for ``token_sentinel.enrichers.otel.TokenSentinelSpanProcessor``.

These tests synthesise OTel spans using ``unittest.mock`` and
``types.SimpleNamespace`` shapes instead of running a real OTel pipeline,
so the suite stays fast and dependency-light. The real
``opentelemetry-sdk`` IS imported (it's installed as a dev dep), so the
processor subclasses the real ``SpanProcessor`` base and gets the same
type identity OTel checks for in ``TracerProvider.add_span_processor``.

The processor's contract:

  1. Constructs cleanly when ``opentelemetry-sdk`` is available.
  2. Raises a clear ``ImportError`` when OTel is missing.
  3. ``on_end`` for a CrewAI-shaped span → one ``CallRecord`` in
     the Sentinel's tracer, with the agent name on
     ``CallRecord.tags['agent']``.
  4. AutoGen-shaped span → ``CallRecord.tags['role']``.
  5. Pydantic AI-shaped span with a ``pydantic_ai.tool_name`` →
     ``CallRecord.tool_calls=[{name, arguments}]``.
  6. Token counts flow from ``gen_ai.usage.input_tokens`` /
     ``gen_ai.usage.output_tokens`` (and legacy aliases).
  7. Latency computed from start/end timestamps (ns → ms).
  8. ``gen_ai.conversation.id`` becomes ``session_id`` when present.
  9. Falls back to OTel ``trace_id`` (32-char hex) for ``session_id``
     when ``conversation.id`` is absent.
 10. Invalid tag values (per  validation) drop silently — bad
     attributes never crash the span processor's hot path.
 11. Two spans sharing a ``trace_id`` land in the same session.
 12. Block-mode ``LeakDetected`` propagates through the ``on_end``
     frame, preserving wrapper-parity semantics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from token_sentinel import Sentinel
from token_sentinel.enrichers import TokenSentinelSpanProcessor
from token_sentinel.enrichers import otel as otel_module
from token_sentinel.events import CallRecord, LeakDetected, LeakEvent

# ---------------------------------------------------------------------------
# Helpers — build OTel-span-shaped objects without depending on the real SDK
# at runtime. The processor only reads attributes / start_time / end_time /
# name / get_span_context(); a SimpleNamespace covers the surface.
# ---------------------------------------------------------------------------


def _make_span(
    *,
    name: str = "gen_ai.chat",
    attributes: dict[str, Any] | None = None,
    start_time: int | None = 1_700_000_000_000_000_000,  # ns since epoch
    end_time: int | None = 1_700_000_000_120_000_000,  # +120ms
    trace_id: int | None = 0x1234567890ABCDEF1234567890ABCDEF,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like an OTel ``ReadableSpan``."""
    span_context = (
        SimpleNamespace(trace_id=trace_id, span_id=0xABCDEF1234567890)
        if trace_id is not None
        else None
    )
    return SimpleNamespace(
        name=name,
        attributes=dict(attributes or {}),
        start_time=start_time,
        end_time=end_time,
        get_span_context=lambda: span_context,
    )


def _crewai_span(**overrides: Any) -> SimpleNamespace:
    """A representative CrewAI-shaped LLM-call span."""
    attrs: dict[str, Any] = {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-sonnet-4-6",
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 45,
        "crewai.agent.name": "research_agent",
    }
    attrs.update(overrides.pop("attributes_extra", {}))
    overrides.setdefault("name", "gen_ai.chat anthropic")
    return _make_span(attributes=attrs, **overrides)


def _autogen_span(**overrides: Any) -> SimpleNamespace:
    """A representative AutoGen-shaped LLM-call span."""
    attrs: dict[str, Any] = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 200,
        "gen_ai.usage.output_tokens": 80,
        "autogen.agent_role": "planner",
    }
    attrs.update(overrides.pop("attributes_extra", {}))
    overrides.setdefault("name", "gen_ai.chat openai")
    return _make_span(attributes=attrs, **overrides)


def _pydantic_ai_span(**overrides: Any) -> SimpleNamespace:
    """A representative Pydantic AI-shaped tool-call span."""
    attrs: dict[str, Any] = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.operation.name": "tool_call",
        "gen_ai.usage.input_tokens": 50,
        "gen_ai.usage.output_tokens": 10,
        "pydantic_ai.tool_name": "calculator",
    }
    attrs.update(overrides.pop("attributes_extra", {}))
    overrides.setdefault("name", "gen_ai.tool_call openai")
    return _make_span(attributes=attrs, **overrides)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_processor_constructs_when_opentelemetry_installed() -> None:
    """The constructor succeeds when ``opentelemetry-sdk`` is importable.
    Verifies the processor is a real ``SpanProcessor`` so OTel accepts
    it in ``TracerProvider.add_span_processor`` without a duck-type sniff.
    """
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    # Must be a real SpanProcessor so OTel accepts it.
    from opentelemetry.sdk.trace import SpanProcessor

    assert isinstance(processor, SpanProcessor)


def test_processor_raises_clear_importerror_when_otel_missing() -> None:
    """When ``_OTEL_AVAILABLE`` is False, instantiation raises with a
    hint pointing to ``pip install token-sentinel[otel]``."""
    sentinel = Sentinel(project="proj")
    with (
        mock.patch.object(otel_module, "_OTEL_AVAILABLE", False),
        pytest.raises(ImportError, match=r"opentelemetry-sdk"),
    ):
        TokenSentinelSpanProcessor(sentinel)


# ---------------------------------------------------------------------------
# Framework-shaped spans → CallRecord
# ---------------------------------------------------------------------------


def test_crewai_span_produces_call_record_with_agent_tag() -> None:
    """A CrewAI-shaped span flows through ``on_end`` → ``Sentinel.record_call``
    and lands in the tracer as one CallRecord with ``tags['agent']`` set
    to the validated ``crewai.agent.name``."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _crewai_span()
    processor.on_end(span)

    sessions = list(sentinel.tracer.all_sessions())
    assert len(sessions) == 1
    records = sentinel.tracer.session(sessions[0])
    assert len(records) == 1
    rec: CallRecord = records[0]
    assert rec.provider == "anthropic"
    assert rec.model == "claude-sonnet-4-6"
    assert rec.method == "otel.chat"
    assert rec.prompt_tokens == 120
    assert rec.completion_tokens == 45
    # CrewAI quirk: agent name lands on tags.agent.
    assert rec.tags == {"agent": "research_agent"}


def test_autogen_span_produces_call_record_with_role_tag() -> None:
    """An AutoGen-shaped span lands ``autogen.agent_role`` on
    ``CallRecord.tags['role']``."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _autogen_span()
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.provider == "openai"
    assert rec.model == "gpt-4o"
    assert rec.prompt_tokens == 200
    assert rec.completion_tokens == 80
    assert rec.tags == {"role": "planner"}


def test_pydantic_ai_span_populates_tool_calls() -> None:
    """A Pydantic AI tool-call span produces a CallRecord with
    ``tool_calls=[{name, arguments}]`` so the ``tool_loop`` rule can
    see it. ``pydantic_ai.tool_name`` also lands on ``tags['tool']``."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _pydantic_ai_span()
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0] == {"name": "calculator", "arguments": {}}
    # The tool name is also tagged for chargeback attribution.
    assert rec.tags == {"tool": "calculator"}
    # Tool-call spans aren't user-facing — the rule engine cares.
    assert rec.user_facing_output is False


# ---------------------------------------------------------------------------
# Attribute mapping fidelity
# ---------------------------------------------------------------------------


def test_token_counts_flow_from_gen_ai_usage_attributes() -> None:
    """Token counts on ``gen_ai.usage.input_tokens`` /
    ``gen_ai.usage.output_tokens`` land verbatim on the CallRecord."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 1234,
            "gen_ai.usage.output_tokens": 567,
        }
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.prompt_tokens == 1234
    assert rec.completion_tokens == 567


def test_legacy_token_attribute_aliases_supported() -> None:
    """Pre-spec frameworks (or older OTel semconv versions) emit
    ``gen_ai.usage.prompt_tokens`` / ``gen_ai.usage.completion_tokens``
    instead of the current ``input``/``output`` keys. Both shapes are
    common in the wild and the enricher must accept both."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.prompt_tokens": 99,
            "gen_ai.usage.completion_tokens": 11,
        }
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.prompt_tokens == 99
    assert rec.completion_tokens == 11


def test_latency_ms_computed_from_span_timestamps() -> None:
    """Latency = (end_time - start_time) / 1_000_000 (ns → ms). A 250ms
    span produces ``latency_ms == 250.0``."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    start = 1_700_000_000_000_000_000
    end = start + 250 * 1_000_000  # +250ms in ns
    span = _make_span(
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-6",
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 1,
        },
        start_time=start,
        end_time=end,
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.latency_ms == pytest.approx(250.0)


def test_latency_zero_when_timestamps_missing() -> None:
    """A span with no end_time (shouldn't happen in real OTel, but a
    framework with a bug could emit one) produces latency 0 instead
    of crashing."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "x",
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 1,
        },
        start_time=None,
        end_time=None,
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.latency_ms == 0.0


def test_response_model_overrides_request_model() -> None:
    """``gen_ai.response.model`` (specific, e.g.
    ``claude-sonnet-4-6-20260301``) is preferred over the requested
    ``gen_ai.request.model`` (generic, e.g. ``claude-sonnet-4-6``).
    Mirrors the LangChain enricher's behaviour for ``llm_output.model_name``.
    """
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-sonnet-4-6",
            "gen_ai.response.model": "claude-sonnet-4-6-20260301",
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 1,
        }
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.model == "claude-sonnet-4-6-20260301"


# ---------------------------------------------------------------------------
# Session-id derivation
# ---------------------------------------------------------------------------


def test_session_id_uses_conversation_id_when_present() -> None:
    """The gen_ai conventions include ``gen_ai.conversation.id`` (some
    frameworks emit it). When present, it becomes the CallRecord's
    ``session_id`` — taking precedence over the trace_id fallback."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.conversation.id": "conv-42",
            "gen_ai.usage.input_tokens": 5,
            "gen_ai.usage.output_tokens": 2,
        }
    )
    processor.on_end(span)

    sessions = list(sentinel.tracer.all_sessions())
    assert sessions == ["conv-42"]


def test_session_id_falls_back_to_trace_id_when_conversation_id_absent() -> None:
    """Without ``gen_ai.conversation.id`` the OTel trace_id (rendered
    as 32-char hex) is the session bucket. This matches the "one trace
    = one agent run = one session" mental model and lets customers
    cross-reference TokenSentinel sessions with Jaeger/Tempo traces."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    trace_id = 0xABCDEF0123456789ABCDEF0123456789
    span = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 5,
            "gen_ai.usage.output_tokens": 2,
        },
        trace_id=trace_id,
    )
    processor.on_end(span)

    sessions = list(sentinel.tracer.all_sessions())
    assert sessions == ["abcdef0123456789abcdef0123456789"]
    # And it's exactly 32 hex chars (128-bit zero-padded).
    assert len(sessions[0]) == 32


def test_multiple_spans_same_trace_id_share_session() -> None:
    """Two spans sharing a ``trace_id`` (the OTel convention for "one
    agent run") land in the same session bucket so cross-call rules
    like ``retry_storm`` and ``tool_loop`` see the full window."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    shared_trace = 0x11223344556677881122334455667788
    span_a = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
        },
        trace_id=shared_trace,
    )
    span_b = _make_span(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 20,
            "gen_ai.usage.output_tokens": 10,
        },
        trace_id=shared_trace,
    )

    processor.on_end(span_a)
    processor.on_end(span_b)

    sessions = list(sentinel.tracer.all_sessions())
    assert len(sessions) == 1
    records = sentinel.tracer.session(sessions[0])
    assert len(records) == 2
    # Distinct token counts so we know they didn't collapse.
    assert [r.prompt_tokens for r in records] == [10, 20]


# ---------------------------------------------------------------------------
# Robustness — invalid tag values + missing attributes
# ---------------------------------------------------------------------------


def test_invalid_tag_value_dropped_silently() -> None:
    """A ``crewai.agent.name`` value with a space (fails the
    URL-safe regex) is dropped silently — the span still produces a
    CallRecord but ``tags`` stays empty. The processor MUST NEVER
    raise from ``on_end`` per the OTel SpanProcessor contract."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _crewai_span(
        attributes_extra={"crewai.agent.name": "bad name with spaces"},
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    # The bad tag was dropped — the CallRecord still exists.
    assert rec.tags == {}
    assert rec.prompt_tokens == 120  # other fields unaffected


def test_oversized_tag_value_dropped_silently() -> None:
    """A tag value > 64 chars exceeds the  cap and is dropped.
    The CallRecord still flows; the cloud's by-tag aggregation just
    doesn't see this span's agent."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _autogen_span(
        attributes_extra={"autogen.agent_role": "x" * 65},
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.tags == {}


def test_non_gen_ai_spans_are_ignored() -> None:
    """Frameworks emit many non-gen_ai spans (internal bookkeeping,
    network calls, etc.). The processor's fast filter only routes
    spans that carry at least one ``gen_ai.*`` attribute — others
    are dropped without producing a CallRecord."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        name="internal.bookkeeping",
        attributes={
            "service.name": "my-agent",
            "duration_ms": 12.5,
        },
    )
    processor.on_end(span)

    # No gen_ai.* attrs → no CallRecord.
    assert list(sentinel.tracer.all_sessions()) == []


def test_missing_attributes_does_not_crash() -> None:
    """A span whose ``.attributes`` is None (shouldn't happen with
    real OTel, but a buggy custom processor could synthesise one)
    is dropped without crashing — defensive against framework bugs."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    # Span with None attributes — must not raise.
    span = SimpleNamespace(
        name="x",
        attributes=None,
        start_time=0,
        end_time=0,
        get_span_context=lambda: None,
    )
    processor.on_end(span)  # No assertion — the requirement is "doesn't raise".
    assert list(sentinel.tracer.all_sessions()) == []


def test_provider_falls_back_to_otel_when_system_missing() -> None:
    """A span with token-usage attributes but no ``gen_ai.system`` still
    produces a CallRecord — ``provider`` falls back to ``"otel"`` so
    records stay routable."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _make_span(
        attributes={
            "gen_ai.request.model": "mystery-model",
            "gen_ai.usage.input_tokens": 1,
            "gen_ai.usage.output_tokens": 1,
        },
    )
    processor.on_end(span)

    rec = sentinel.tracer.session(list(sentinel.tracer.all_sessions())[0])[0]
    assert rec.provider == "otel"
    assert rec.model == "mystery-model"


# ---------------------------------------------------------------------------
# Block-mode propagation
# ---------------------------------------------------------------------------


def test_block_mode_propagates_leak_detected_through_on_end() -> None:
    """In ``mode='block'``, a rule firing inside ``Sentinel.record_call``
    raises ``LeakDetected``. The span processor must let that propagate
    out of ``on_end`` so OTel's processor pipeline surfaces it — same
    contract as the LangChain enricher established in ."""
    sentinel = Sentinel(project="proj", mode="block")

    fake_event = LeakEvent(
        type="tool_loop",
        confidence=0.9,
        project="proj",
        session_id="x",
        rule="tool_loop",
        evidence={},
        estimated_burn=0.001,
        suggested_action="halt",
    )

    with mock.patch.object(sentinel, "record_call", side_effect=LeakDetected(fake_event)):
        processor = TokenSentinelSpanProcessor(sentinel)
        span = _crewai_span()
        with pytest.raises(LeakDetected):
            processor.on_end(span)


# ---------------------------------------------------------------------------
# Lifecycle methods
# ---------------------------------------------------------------------------


def test_on_start_is_no_op() -> None:
    """``on_start`` is intentionally a no-op (we do all work in
    ``on_end``). Calling it must not raise and must not produce any
    records."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    span = _crewai_span()
    processor.on_start(span)  # Must not raise.
    # No CallRecord produced — only on_end builds one.
    assert list(sentinel.tracer.all_sessions()) == []


def test_force_flush_returns_true() -> None:
    """``force_flush`` returns ``True`` (nothing to flush) — matches
    the OTel ``BatchSpanProcessor`` contract for "no pending work"."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    assert processor.force_flush() is True
    assert processor.force_flush(timeout_millis=1000) is True


def test_shutdown_is_no_op() -> None:
    """``shutdown`` is a no-op — must not raise. OTel calls it on
    TracerProvider teardown."""
    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)

    processor.shutdown()  # Must not raise.


# ---------------------------------------------------------------------------
# Re-export verification
# ---------------------------------------------------------------------------


def test_processor_is_re_exported_from_enrichers_package() -> None:
    """Importing from ``token_sentinel.enrichers`` (not just
    ``...enrichers.otel``) succeeds — the public ``__all__`` includes
    the processor."""
    from token_sentinel.enrichers import TokenSentinelCallbackHandler
    from token_sentinel.enrichers import TokenSentinelSpanProcessor as Processor

    assert Processor is TokenSentinelSpanProcessor
    # And the LangChain enricher still exports (no regression).
    assert TokenSentinelCallbackHandler is not None


# ---------------------------------------------------------------------------
# Integration with real OTel TracerProvider
# ---------------------------------------------------------------------------


def test_processor_registers_with_real_tracer_provider() -> None:
    """End-to-end: register the processor on a real ``TracerProvider``,
    start a real span, set gen_ai attributes, and end it. The
    ``Sentinel.record_call`` path executes in the real OTel callback
    chain. Catches any subclass / signature mismatches that synthetic
    span-namespaces miss."""
    from opentelemetry.sdk.trace import TracerProvider

    sentinel = Sentinel(project="proj")
    processor = TokenSentinelSpanProcessor(sentinel)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("gen_ai.chat") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        span.set_attribute("gen_ai.usage.input_tokens", 7)
        span.set_attribute("gen_ai.usage.output_tokens", 3)

    # The span ends on __exit__; on_end has been called by now.
    sessions = list(sentinel.tracer.all_sessions())
    assert len(sessions) == 1
    rec = sentinel.tracer.session(sessions[0])[0]
    assert rec.provider == "openai"
    assert rec.model == "gpt-4o"
    assert rec.prompt_tokens == 7
    assert rec.completion_tokens == 3
    # Force flush + shutdown should not raise.
    assert processor.force_flush() is True
    processor.shutdown()
