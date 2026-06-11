"""Tests for the v0.3.3 cleanup pass — defensive guards against low-severity
edge cases surfaced during the SDK hardening sweep.

Coverage: streaming OpenAI session-id strip, ``context_bloat`` divide-by-zero
guard, non-dict messages handled in ``_flatten_messages``, ``min_confidence``
clamp, defensive ``event.confidence`` clamp in ``record_call``,
``LeakEvent.__str__`` triage info, thread-safe ``_warn_block_mode_stream_once``.
"""

from __future__ import annotations

import threading
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from token_sentinel import LeakDetected, LeakEvent, Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.rules.context_bloat import ContextBloatRule
from token_sentinel.rules.model_misroute import ModelMisrouteRule, _flatten_messages

# ---------------------------------------------------------------------------
# LOW-1 — Streaming OpenAI strips _sentinel_session_id
# ---------------------------------------------------------------------------


def test_streaming_openai_strips_sentinel_session_id_kwarg():
    """LOW-1: streaming pass-through must not leak our private kwarg into
    the user's underlying OpenAI SDK call.

    The wrapper calls ``kwargs.pop("_sentinel_session_id", ...)`` BEFORE the
    ``stream=True`` early return, so the kwarg should be absent from the
    underlying call regardless of streaming.
    """
    from token_sentinel.wrappers.openai import wrap_openai

    captured: dict = {}

    def real_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()  # streaming returns a stream object normally

    real_create.__name__ = "create"
    real_create.__module__ = "openai.resources.chat.completions"

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=real_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: None)

    s = Sentinel(project="proj", mode="log")
    wrap_openai(client, s)

    client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        _sentinel_session_id="must-not-leak",
    )

    assert "_sentinel_session_id" not in captured
    assert captured.get("stream") is True


# ---------------------------------------------------------------------------
# LOW-2 — context_bloat divide-by-zero guard
# ---------------------------------------------------------------------------


def test_context_bloat_handles_zero_slope_threshold():
    """LOW-2: ``slope_threshold=0`` used to trigger ZeroDivisionError in
    the confidence formula ``(slope - threshold) / (threshold * 4)``.
    Now the denominator is guarded.
    """
    rule = ContextBloatRule({"context_bloat.slope_threshold": 0})
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    session = [
        CallRecord(
            session_id="s1",
            timestamp=now + timedelta(seconds=i),
            provider="anthropic",
            model="claude-sonnet-4-6",
            method="messages.create",
            prompt_tokens=1000 + i * 500,
            completion_tokens=100,
            latency_ms=100.0,
            request_hash=f"h{i}",
        )
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="proj")
    assert ev is not None
    assert ev.type == "context_bloat"
    assert 0.0 <= ev.confidence <= 1.0


# ---------------------------------------------------------------------------
# LOW-3 — non-dict messages don't break _flatten_messages
# ---------------------------------------------------------------------------


def test_flatten_messages_skips_non_dict_items():
    """LOW-3: a Pydantic BaseModel / string / None in the messages list
    must NOT crash the rule with AttributeError on ``m.get``.
    """
    messages = [
        {"role": "user", "content": "hello"},
        "raw string sneaks in",
        None,
        SimpleNamespace(role="assistant", content="object instead of dict"),
        {"role": "user", "content": "classify this"},
    ]
    out = _flatten_messages(messages)
    # Only the dict items contributed. No crash.
    assert "hello" in out
    assert "classify this" in out
    assert "raw string sneaks in" not in out
    assert "object instead of dict" not in out


def test_model_misroute_does_not_crash_on_non_dict_messages():
    """End-to-end: rule.evaluate succeeds on a CallRecord whose raw_request
    has non-dict items in messages. (Returns None — no classification cue.)
    """
    rule = ModelMisrouteRule({})
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    session = [
        CallRecord(
            session_id="s1",
            timestamp=now,
            provider="anthropic",
            model="claude-opus-4-1",
            method="messages.create",
            prompt_tokens=50,
            completion_tokens=5,
            latency_ms=100.0,
            request_hash="h",
            raw_request={"messages": [None, "weird"]},
        )
    ]
    # Must not crash.
    ev = rule.evaluate(session, project="proj")
    assert ev is None


# ---------------------------------------------------------------------------
# LOW-4 — min_confidence clamping
# ---------------------------------------------------------------------------


def test_min_confidence_above_one_is_clamped_with_warning():
    """LOW-4: ``min_confidence=2.0`` would silently disable every rule. Now
    we clamp to 1.0 and warn so the customer notices the misuse.
    """
    with pytest.warns(UserWarning, match="min_confidence=2.0 clamped"):
        s = Sentinel(project="proj", min_confidence=2.0)
    assert s.min_confidence == 1.0


def test_min_confidence_below_zero_is_clamped_with_warning():
    """LOW-4: negative min_confidence would let every event through."""
    with pytest.warns(UserWarning, match="min_confidence=-0.5 clamped"):
        s = Sentinel(project="proj", min_confidence=-0.5)
    assert s.min_confidence == 0.0


def test_min_confidence_in_range_no_warning():
    """LOW-4: valid values must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into a test failure
        s = Sentinel(project="proj", min_confidence=0.7)
    assert s.min_confidence == 0.7


# ---------------------------------------------------------------------------
# LOW-5 — event.confidence clamping in record_call
# ---------------------------------------------------------------------------


def test_record_call_clamps_out_of_range_confidence():
    """LOW-5: a future rule that forgets the ``min(...)`` cap on confidence
    must still produce events in the documented [0.0, 1.0] range.
    """

    class _RogueRule:
        name = "rogue"

        def __init__(self):
            self.config = {}

        def evaluate(self, session, *, project):
            if not session:
                return None
            # Out-of-range confidence — rule author missed the contract.
            return LeakEvent(
                type="rogue",
                confidence=2.5,
                project=project,
                session_id=session[-1].session_id,
                rule="v0.rogue",
                evidence={"reason": "test"},
                estimated_burn=0.01,
                suggested_action="noop",
            )

    s = Sentinel(project="proj", min_confidence=0.0)
    s._rules = [_RogueRule()]
    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    cr = CallRecord(
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        method="messages.create",
        prompt_tokens=100,
        completion_tokens=10,
        latency_ms=50.0,
        request_hash="h",
    )
    s.record_call(cr)

    assert len(seen) == 1
    assert 0.0 <= seen[0].confidence <= 1.0
    assert seen[0].confidence == 1.0


# ---------------------------------------------------------------------------
# NIT-3 — LeakEvent.__str__ includes session_id + evidence_keys
# ---------------------------------------------------------------------------


def test_leakevent_str_includes_session_id_and_evidence_keys():
    """NIT-3: triage info needs to land in the default repr; values still
    omitted (avoid the redaction-leak vector)."""
    ev = LeakEvent(
        type="tool_loop",
        confidence=0.9,
        project="proj",
        session_id="abc-123",
        rule="v0.tool_loop",
        evidence={"tool": "search", "call_count": 4, "mean_similarity": 0.95},
        estimated_burn=0.123,
        suggested_action="pause_for_human_review",
    )
    s = str(ev)
    assert "session_id=abc-123" in s
    assert "evidence_keys=[" in s
    # Keys are listed (alphabetically), values are NOT.
    assert "call_count" in s
    assert "mean_similarity" in s
    assert "tool" in s
    assert "search" not in s  # raw value must not surface
    assert "0.95" not in s  # raw value must not surface


def test_leakdetected_str_includes_session_id():
    """NIT-3: ``str(LeakDetected)`` delegates to the event's str — should
    surface session_id for triage."""
    ev = LeakEvent(
        type="retry_storm",
        confidence=0.9,
        project="proj",
        session_id="prod-customer-42",
        rule="v0.retry_storm",
        evidence={"retry_count": 8},
        estimated_burn=0.05,
        suggested_action="add_backoff",
    )
    err = LeakDetected(ev)
    assert "prod-customer-42" in str(err)


# ---------------------------------------------------------------------------
# Followup NIT — concurrent _warn_block_mode_stream_once doesn't double-warn
# ---------------------------------------------------------------------------


def test_warn_block_mode_stream_once_thread_safe():
    """Followup NIT: under concurrent first-call traffic, the warning must
    fire exactly once per (Sentinel, path), not twice. The lock around the
    membership test eliminates the race window.
    """
    from token_sentinel.wrappers.openai import _warn_block_mode_stream_once

    s = Sentinel(project="proj", mode="block")
    barrier = threading.Barrier(8)
    warned = []

    def fire():
        barrier.wait()  # release all threads at exactly the same moment
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            _warn_block_mode_stream_once(s, "sync")
            warned.extend(caught)

    threads = [threading.Thread(target=fire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one of the eight threads emitted the warning.
    runtime_warnings = [w for w in warned if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1, f"expected exactly 1 warning, got {len(runtime_warnings)}"
