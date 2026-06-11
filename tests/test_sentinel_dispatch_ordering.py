"""Dispatch ordering / multi-event tests.

These tests cover the HIGH-severity finding "Sentinel._dispatch raises after
the first event so subsequent events' handlers never run" from the
2026-05-07 code review.

Old behaviour:
    for ev in events:
        self._dispatch(ev)        # raises mid-loop in block mode
                                  # → handlers for events[1:] never run

New behaviour:
    1. ``record_call`` runs handlers for every event first.
    2. In block mode it raises ``LeakDetected`` exactly once, with the
       *highest-confidence* event (tiebreak: first iteration order).
    3. Modes ``log`` and ``alert`` never raise — they just call handlers.

To exercise multi-event dispatch deterministically without relying on
fragile rule-fire conditions, we monkeypatch ``Sentinel._rules`` with a
list of stub rules that return canned ``LeakEvent`` instances.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from token_sentinel import LeakDetected, LeakEvent, Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.rules.base import Rule

# ---------------------------------------------------------------------------
# Test infrastructure: stub rule that emits a canned event.
# ---------------------------------------------------------------------------


class _StubRule(Rule):
    """A rule that returns a pre-configured ``LeakEvent`` (or ``None``)
    whenever ``evaluate`` is called. Fires once per ``record_call``.
    """

    def __init__(self, name: str, event: LeakEvent | None):
        super().__init__(config={})
        self.name = name
        self._event = event
        self.eval_count = 0

    def evaluate(self, session, *, project):  # type: ignore[override]
        self.eval_count += 1
        return self._event


def _make_event(
    *,
    type_: str,
    confidence: float,
    rule: str | None = None,
    session_id: str = "s1",
) -> LeakEvent:
    return LeakEvent(
        type=type_,
        confidence=confidence,
        project="proj",
        session_id=session_id,
        rule=rule or type_,
        evidence={},
        estimated_burn=0.01,
        suggested_action="x",
        raised_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
    )


def _install_stub_rules(s: Sentinel, *events: LeakEvent | None) -> None:
    """Replace ``s._rules`` with one stub rule per event (or ``None``).

    Order matters — iteration order in ``record_call`` matches this order.
    """
    s._rules = [_StubRule(name=f"stub_{i}", event=ev) for i, ev in enumerate(events)]


def _make_call_record(*, session_id: str = "s1") -> CallRecord:
    return CallRecord(
        session_id=session_id,
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        method="messages.create",
        prompt_tokens=100,
        completion_tokens=10,
        latency_ms=50.0,
        request_hash="deadbeef",
    )


# ---------------------------------------------------------------------------
# block mode: every handler runs for every event before the raise
# ---------------------------------------------------------------------------


def test_block_mode_runs_all_handlers_for_all_events_before_raising():
    """The customer's handler must see every event even in block mode.

    Pre-fix bug: ``_dispatch`` raised after the first event, so the second
    rule's event was silently dropped.
    """
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    e1 = _make_event(type_="tool_loop", confidence=0.7)
    e2 = _make_event(type_="context_bloat", confidence=0.9)
    _install_stub_rules(s, e1, e2)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    with pytest.raises(LeakDetected):
        s.record_call(_make_call_record())

    # Both events were dispatched to the handler before the raise.
    assert [ev.type for ev in seen] == ["tool_loop", "context_bloat"]


def test_block_mode_picks_highest_confidence_event_for_raise():
    """When multiple events fire, ``LeakDetected.event`` is the most
    confident one — so the customer's exception handler triages the
    strongest signal first."""
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    e_low = _make_event(type_="tool_loop", confidence=0.55)
    e_high = _make_event(type_="context_bloat", confidence=0.95)
    e_mid = _make_event(type_="retry_storm", confidence=0.75)
    # Install in non-sorted order so the "first iteration order" tiebreak
    # cannot accidentally produce the right answer.
    _install_stub_rules(s, e_low, e_high, e_mid)

    with pytest.raises(LeakDetected) as exc:
        s.record_call(_make_call_record())

    assert exc.value.event is e_high
    assert exc.value.event.type == "context_bloat"
    assert exc.value.event.confidence == 0.95


def test_block_mode_tiebreak_is_first_iteration_order():
    """If two events tie on confidence, the first one in iteration order wins.

    Iteration order is rule-list order, which is deterministic and stable.
    """
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    first = _make_event(type_="tool_loop", confidence=0.8)
    second = _make_event(type_="context_bloat", confidence=0.8)
    third = _make_event(type_="retry_storm", confidence=0.8)
    _install_stub_rules(s, first, second, third)

    with pytest.raises(LeakDetected) as exc:
        s.record_call(_make_call_record())

    assert exc.value.event is first
    assert exc.value.event.type == "tool_loop"


def test_block_mode_with_single_event_raises_with_that_event():
    """The single-event case must still raise with that one event —
    ``_highest_confidence`` of a 1-list is the one element."""
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    only = _make_event(type_="tool_loop", confidence=0.6)
    _install_stub_rules(s, only)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    with pytest.raises(LeakDetected) as exc:
        s.record_call(_make_call_record())

    assert exc.value.event is only
    assert seen == [only]


def test_block_mode_with_no_events_does_not_raise():
    """If no rule fires, block mode is a no-op."""
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    _install_stub_rules(s, None, None)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    # Should NOT raise.
    result = s.record_call(_make_call_record())
    assert result == []
    assert seen == []


# ---------------------------------------------------------------------------
# log + alert modes: never raise, but still fire all handlers
# ---------------------------------------------------------------------------


def test_log_mode_runs_all_handlers_and_does_not_raise():
    s = Sentinel(project="proj", mode="log", min_confidence=0.0)
    e1 = _make_event(type_="tool_loop", confidence=0.6)
    e2 = _make_event(type_="context_bloat", confidence=0.95)
    _install_stub_rules(s, e1, e2)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    # Must NOT raise.
    events = s.record_call(_make_call_record())

    assert [ev.type for ev in seen] == ["tool_loop", "context_bloat"]
    assert [ev.type for ev in events] == ["tool_loop", "context_bloat"]


def test_alert_mode_runs_all_handlers_and_does_not_raise():
    """``alert`` mode is currently equivalent to ``log`` for dispatch
    purposes (the difference is intended for downstream cloud routing).
    Both must fire every handler for every event and never raise."""
    s = Sentinel(project="proj", mode="alert", min_confidence=0.0)
    e1 = _make_event(type_="tool_loop", confidence=0.6)
    e2 = _make_event(type_="context_bloat", confidence=0.95)
    _install_stub_rules(s, e1, e2)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    events = s.record_call(_make_call_record())

    assert [ev.type for ev in seen] == ["tool_loop", "context_bloat"]
    assert events is not None
    assert [ev.type for ev in events] == ["tool_loop", "context_bloat"]


# ---------------------------------------------------------------------------
# Handler exception isolation across multi-event dispatch
# ---------------------------------------------------------------------------


def test_handler_exception_during_first_event_does_not_stop_second_event():
    """A throwing handler must not prevent the same handler (or a sibling
    handler) from running on subsequent events."""
    s = Sentinel(project="proj", mode="log", min_confidence=0.0)
    e1 = _make_event(type_="tool_loop", confidence=0.7)
    e2 = _make_event(type_="context_bloat", confidence=0.8)
    _install_stub_rules(s, e1, e2)

    fired_for: list[str] = []

    @s.on_leak
    def always_raises(ev):
        fired_for.append(f"raises:{ev.type}")
        raise RuntimeError("boom")

    @s.on_leak
    def good(ev):
        fired_for.append(f"good:{ev.type}")

    # log mode: must not raise, despite the bad handler.
    s.record_call(_make_call_record())

    # Both handlers fired for both events.
    assert fired_for == [
        "raises:tool_loop",
        "good:tool_loop",
        "raises:context_bloat",
        "good:context_bloat",
    ]


def test_handler_exception_in_block_mode_does_not_prevent_raise_or_other_handlers():
    """Block mode + throwing handler: every other handler still fires for
    every event, then ``LeakDetected`` is raised once with the
    highest-confidence event. The bad handler doesn't suppress block."""
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    e1 = _make_event(type_="tool_loop", confidence=0.6)
    e2 = _make_event(type_="context_bloat", confidence=0.85)
    _install_stub_rules(s, e1, e2)

    fired_for: list[str] = []

    @s.on_leak
    def bad(ev):
        fired_for.append(f"bad:{ev.type}")
        raise ValueError("boom")

    @s.on_leak
    def good(ev):
        fired_for.append(f"good:{ev.type}")

    with pytest.raises(LeakDetected) as exc:
        s.record_call(_make_call_record())

    # Highest-confidence event won the raise.
    assert exc.value.event is e2
    # Both handlers fired for both events before the raise.
    assert fired_for == [
        "bad:tool_loop",
        "good:tool_loop",
        "bad:context_bloat",
        "good:context_bloat",
    ]


# ---------------------------------------------------------------------------
# Backwards-compatibility: legacy _dispatch still works for single events
# ---------------------------------------------------------------------------


def test_legacy_dispatch_runs_handlers_and_raises_in_block_mode():
    """``_dispatch`` is preserved as a thin compatibility wrapper for any
    test or downstream code that calls it directly. Behaves like the old
    single-event semantics: run handlers then (block) raise."""
    s = Sentinel(project="proj", mode="block")
    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))
    ev = _make_event(type_="tool_loop", confidence=0.7)

    with pytest.raises(LeakDetected) as exc:
        s._dispatch(ev)

    assert exc.value.event is ev
    assert seen == [ev]


def test_legacy_dispatch_in_log_mode_does_not_raise():
    s = Sentinel(project="proj", mode="log")
    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))
    ev = _make_event(type_="tool_loop", confidence=0.7)

    s._dispatch(ev)
    assert seen == [ev]


# ---------------------------------------------------------------------------
# record_call return value preserves all events even in block mode (when
# we don't actually raise — i.e., in log/alert)
# ---------------------------------------------------------------------------


def test_record_call_in_log_mode_returns_all_fired_events():
    """The list returned from ``record_call`` includes every event whose
    confidence >= ``min_confidence``."""
    s = Sentinel(project="proj", mode="log", min_confidence=0.5)
    e_below = _make_event(type_="tool_loop", confidence=0.49)
    e_above_1 = _make_event(type_="context_bloat", confidence=0.9)
    e_above_2 = _make_event(type_="retry_storm", confidence=0.5)
    _install_stub_rules(s, e_below, e_above_1, e_above_2)

    events = s.record_call(_make_call_record())
    # Below-threshold event is filtered out.
    assert [ev.type for ev in events] == ["context_bloat", "retry_storm"]
