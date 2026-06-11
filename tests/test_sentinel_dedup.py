"""Tests for ``Sentinel`` event de-duplication (MED-4).

Closes the v0.2.0 code-review finding "record_call evaluates rules with a
stale snapshot under concurrent writes". Two threads concurrently
``record_call``-ing the same session can each see overlapping rule windows
and emit duplicate events. We dedup by ``(event.type, event.rule,
evidence_hash)`` per-session within ``dedup_window_seconds`` so customer
leak handlers see one signal, not two.

Tests in this file directly exercise the dedup path by constructing
``LeakEvent`` instances and feeding them through ``_filter_duplicate_events``
where appropriate, and by invoking ``record_call`` end-to-end with rules
that fire in deterministic ways. We avoid coupling these tests to a
particular rule's internals — instead, we monkeypatch a sentinel's
``_rules`` to a tiny synthetic rule that produces a known-shape event.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord, LeakEvent

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    type_: str = "tool_loop",
    rule: str = "tool_loop",
    confidence: float = 0.9,
    evidence: dict[str, Any] | None = None,
    session_id: str = "s1",
) -> LeakEvent:
    return LeakEvent(
        type=type_,
        confidence=confidence,
        project="proj",
        session_id=session_id,
        rule=rule,
        evidence=evidence if evidence is not None else {"k": "v"},
        estimated_burn=0.01,
        suggested_action="x",
        raised_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_call(session_id: str = "s1") -> CallRecord:
    return CallRecord(
        session_id=session_id,
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        provider="anthropic",
        model="claude",
        method="messages.create",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1.0,
        request_hash="h",
    )


class _FixedRule:
    """A synthetic rule that always returns a pre-built event.

    Lets us drive ``record_call`` through known events without needing to
    construct realistic sessions for every default rule. Has no public
    config — uses an empty dict.
    """

    name = "fixed_rule"

    def __init__(self, event: LeakEvent | None) -> None:
        self.config: dict[str, Any] = {}
        self._event = event

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        return self._event


# ---------------------------------------------------------------------------
# Same event within window dedupes
# ---------------------------------------------------------------------------


def test_same_event_within_window_is_deduped():
    """Two ``record_call``s producing the same event in the dedup window —
    only the first triggers a handler invocation."""
    s = Sentinel(project="proj", dedup_window_seconds=2.0)
    ev = _make_event(evidence={"sample": [1, 2, 3]})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    received: list[LeakEvent] = []
    s.on_leak(received.append)

    out1 = s.record_call(_make_call("s1"))
    out2 = s.record_call(_make_call("s1"))

    assert len(out1) == 1
    assert len(out2) == 0  # second call's event was dedup'd
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Same event AFTER window expires fires again
# ---------------------------------------------------------------------------


def test_same_event_after_window_fires_again():
    """A 0.05s window — wait 0.1s and the second call should re-emit."""
    s = Sentinel(project="proj", dedup_window_seconds=0.05)
    ev = _make_event(evidence={"sample": "x"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    out1 = s.record_call(_make_call("s1"))
    time.sleep(0.12)
    out2 = s.record_call(_make_call("s1"))

    assert len(out1) == 1
    assert len(out2) == 1


# ---------------------------------------------------------------------------
# Different sessions never dedupe each other
# ---------------------------------------------------------------------------


def test_different_sessions_do_not_dedupe_each_other():
    """The same event in two different sessions must fire independently —
    a leak in session A doesn't 'cover' session B."""
    s = Sentinel(project="proj", dedup_window_seconds=10.0)
    ev = _make_event(evidence={"k": "v"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    out_a = s.record_call(_make_call("session-a"))
    out_b = s.record_call(_make_call("session-b"))

    assert len(out_a) == 1
    assert len(out_b) == 1


# ---------------------------------------------------------------------------
# Different rule types in the same session don't dedupe each other
# ---------------------------------------------------------------------------


def test_different_event_types_do_not_dedupe_each_other():
    """A ``tool_loop`` and a ``context_bloat`` event with the SAME evidence
    must both fire — the rule/type discriminator is part of the dedup key."""
    s = Sentinel(project="proj", dedup_window_seconds=10.0)

    ev_tool = _make_event(type_="tool_loop", rule="tool_loop", evidence={"k": "v"})
    ev_ctx = _make_event(type_="context_bloat", rule="context_bloat", evidence={"k": "v"})

    # Two rules firing on the same call — they SHOULD both be kept.
    s._rules = [_FixedRule(ev_tool), _FixedRule(ev_ctx)]  # type: ignore[list-item]

    out = s.record_call(_make_call("s1"))
    types = {e.type for e in out}
    assert types == {"tool_loop", "context_bloat"}


# ---------------------------------------------------------------------------
# Disable: dedup_window_seconds=0 is a hard skip
# ---------------------------------------------------------------------------


def test_dedup_disabled_when_window_is_zero():
    """``dedup_window_seconds=0`` must allow duplicates through unchanged.

    Also: zero state mutation under window=0 — ``_recent_event_keys`` stays
    empty so customers who don't want dedup pay zero cost."""
    s = Sentinel(project="proj", dedup_window_seconds=0)
    ev = _make_event(evidence={"k": "v"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    received: list[LeakEvent] = []
    s.on_leak(received.append)

    out1 = s.record_call(_make_call("s1"))
    out2 = s.record_call(_make_call("s1"))

    assert len(out1) == 1
    assert len(out2) == 1
    assert len(received) == 2
    # The dedup state must be untouched (zero-cost path).
    assert s._recent_event_keys == {}


# ---------------------------------------------------------------------------
# Old-session cleanup happens
# ---------------------------------------------------------------------------


def test_dedup_state_cleanup_drops_old_sessions():
    """Sessions whose newest dedup entry is older than ``window * 4``
    are GC'd from ``_recent_event_keys`` on the next dedup pass."""
    s = Sentinel(project="proj", dedup_window_seconds=0.05)
    ev = _make_event(evidence={"k": "v"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    s.record_call(_make_call("old-session"))
    assert "old-session" in s._recent_event_keys

    # Wait long enough for old-session to fall outside cleanup_horizon =
    # window * 4 = 0.2s. Then any new dedup pass triggers GC.
    time.sleep(0.30)
    s.record_call(_make_call("new-session"))

    # old-session should be evicted; new-session should be tracked.
    assert "old-session" not in s._recent_event_keys
    assert "new-session" in s._recent_event_keys


# ---------------------------------------------------------------------------
# Thread safety under N=8 concurrent record_calls
# ---------------------------------------------------------------------------


def test_dedup_thread_safe_concurrent_record_calls():
    """N threads concurrently firing the same event on the same session —
    exactly one event must survive. The lock + monotonic-clock comparison
    must not allow a torn dict update or a duplicate slip-through."""
    s = Sentinel(project="proj", dedup_window_seconds=10.0)
    ev = _make_event(evidence={"k": "v"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    received: list[LeakEvent] = []
    received_lock = threading.Lock()

    def handler(event: LeakEvent) -> None:
        with received_lock:
            received.append(event)

    s.on_leak(handler)

    n = 8
    barrier = threading.Barrier(n)
    results: list[int] = [0] * n

    def worker(idx: int) -> None:
        # Sync start to maximise contention.
        barrier.wait(timeout=5.0)
        out = s.record_call(_make_call("s1"))
        results[idx] = len(out)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "worker thread did not finish"

    # Exactly one record_call must report 1 event; the rest must report 0.
    assert sum(results) == 1
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Performance: dedup adds <1ms p95 over 1000 calls
# ---------------------------------------------------------------------------


def test_dedup_performance_under_one_ms_p95():
    """1000 record_calls on the same session, dedup enabled, must not push
    p95 latency above 1ms. The dedup path is sub-microsecond per call in
    practice; we leave headroom for noisy CI hosts."""
    s = Sentinel(project="proj", dedup_window_seconds=5.0)
    # No rule fires, so we measure the dedup overhead specifically when the
    # ``_dedup_window_seconds > 0 and events`` short-circuit might or might
    # not be hit. To force events into the dedup pipeline AND keep them
    # cheap, use a fixed rule that always emits (each emit gets dedup'd
    # after the first within the 5s window).
    ev = _make_event(evidence={"k": "v"})
    s._rules = [_FixedRule(ev)]  # type: ignore[list-item]

    n = 1000
    timings: list[float] = []
    call = _make_call("perf-session")
    for _ in range(n):
        t0 = time.perf_counter()
        s.record_call(call)
        timings.append(time.perf_counter() - t0)

    timings.sort()
    p95 = timings[int(n * 0.95)]
    # Generous bound — the dedup path itself is well under 100us.
    assert p95 < 0.001, f"p95 latency {p95 * 1000:.3f}ms exceeds 1ms budget"


# ---------------------------------------------------------------------------
# Evidence hashing: dict ordering must not matter
# ---------------------------------------------------------------------------


def test_evidence_dict_ordering_does_not_break_dedup():
    """``evidence={"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` must dedup —
    JSON serialization with ``sort_keys=True`` makes the key stable."""
    s = Sentinel(project="proj", dedup_window_seconds=10.0)

    ev1 = _make_event(evidence={"a": 1, "b": 2})
    ev2 = _make_event(evidence={"b": 2, "a": 1})

    s._rules = [_FixedRule(ev1)]  # type: ignore[list-item]
    out1 = s.record_call(_make_call("s1"))
    s._rules = [_FixedRule(ev2)]  # type: ignore[list-item]
    out2 = s.record_call(_make_call("s1"))

    assert len(out1) == 1
    # Same evidence content with different dict ordering — dedup'd.
    assert len(out2) == 0


# ---------------------------------------------------------------------------
# Different evidence in the same rule fires independently
# ---------------------------------------------------------------------------


def test_different_evidence_does_not_dedupe():
    """Same rule, different evidence — both must fire (e.g., two distinct
    tool loops detected at different points in a session)."""
    s = Sentinel(project="proj", dedup_window_seconds=10.0)

    ev1 = _make_event(evidence={"sample": "first"})
    ev2 = _make_event(evidence={"sample": "second"})

    s._rules = [_FixedRule(ev1)]  # type: ignore[list-item]
    out1 = s.record_call(_make_call("s1"))
    s._rules = [_FixedRule(ev2)]  # type: ignore[list-item]
    out2 = s.record_call(_make_call("s1"))

    assert len(out1) == 1
    assert len(out2) == 1
