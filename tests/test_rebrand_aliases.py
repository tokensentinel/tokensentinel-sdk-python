"""Tests for the V0.16 rebrand aliases ("token leak" → "token waste").

The internal Python API kept its original names (``LeakEvent``, ``LeakDetected``,
``Sentinel.on_leak``) to avoid breaking installed customer code. These tests
pin down the transparent-alias contract for the new names:

* ``WasteEvent is LeakEvent`` (identity, not subclass)
* ``WasteDetected is LeakDetected``
* ``Sentinel.on_waste`` registers into the same dispatch list as ``on_leak``

No ``DeprecationWarning`` is expected — this is a brand alias, not a
deprecation, and the old names remain first-class.
"""

from __future__ import annotations

from datetime import datetime, timezone

from token_sentinel import LeakDetected, LeakEvent, Sentinel
from token_sentinel.events import WasteDetected, WasteEvent


def _make_event(**overrides: object) -> LeakEvent:
    """Build a minimal ``LeakEvent`` for handler-dispatch tests."""
    defaults: dict[str, object] = {
        "type": "test_event",
        "confidence": 0.9,
        "project": "proj",
        "session_id": "sess-1",
        "rule": "test_rule",
        "evidence": {},
        "estimated_burn": 0.0,
        "suggested_action": "noop",
        "raised_at": datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return LeakEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Identity / class aliasing
# ---------------------------------------------------------------------------


def test_waste_event_is_leak_event() -> None:
    """``WasteEvent`` is the SAME class object as ``LeakEvent`` (not a subclass)."""
    assert WasteEvent is LeakEvent


def test_waste_detected_is_leak_detected() -> None:
    """``WasteDetected`` is the SAME class object as ``LeakDetected``."""
    assert WasteDetected is LeakDetected


def test_leak_event_instance_is_waste_event() -> None:
    """A ``LeakEvent`` instance passes ``isinstance(x, WasteEvent)`` and vice-versa.

    Trivially true because the classes are the same object — but the test
    pins the contract so a future refactor that turns ``WasteEvent`` into a
    subclass would fail loudly here.
    """
    event = _make_event()
    assert isinstance(event, LeakEvent)
    assert isinstance(event, WasteEvent)
    # Constructing via the alias yields a ``LeakEvent``-typed instance too.
    via_alias = WasteEvent(
        type="t",
        confidence=0.5,
        project="p",
        session_id="s",
        rule="r",
        evidence={},
        estimated_burn=0.0,
        suggested_action="noop",
    )
    assert isinstance(via_alias, LeakEvent)
    assert type(via_alias) is LeakEvent


def test_waste_detected_catches_leak_detected_raise() -> None:
    """Code that raises ``LeakDetected`` is caught by ``except WasteDetected:``."""
    event = _make_event()
    try:
        raise LeakDetected(event)
    except WasteDetected as exc:
        caught = exc
    assert caught.event is event
    # And the symmetric direction: ``raise WasteDetected(...)`` is caught by
    # ``except LeakDetected:``.
    try:
        raise WasteDetected(event)
    except LeakDetected as exc:
        caught2 = exc
    assert caught2.event is event


# ---------------------------------------------------------------------------
# Sentinel.on_waste method alias
# ---------------------------------------------------------------------------


def test_on_waste_registers_handler_that_fires_on_event() -> None:
    """``sentinel.on_waste(fn)`` registers a handler invoked by ``_run_handlers``."""
    s = Sentinel(project="proj")
    seen: list[LeakEvent] = []

    def handler(ev: LeakEvent) -> None:
        seen.append(ev)

    returned = s.on_waste(handler)
    # Decorator contract: returns the handler unchanged.
    assert returned is handler
    assert handler in s._handlers

    event = _make_event()
    s._run_handlers(event)
    assert seen == [event]


def test_on_waste_and_on_leak_share_dispatch_list() -> None:
    """Both decorators append to the same internal ``_handlers`` list, so a
    single fired event reaches handlers registered via either name."""
    s = Sentinel(project="proj")
    seen_leak: list[LeakEvent] = []
    seen_waste: list[LeakEvent] = []

    @s.on_leak
    def leak_handler(ev: LeakEvent) -> None:
        seen_leak.append(ev)

    @s.on_waste
    def waste_handler(ev: LeakEvent) -> None:
        seen_waste.append(ev)

    # Both handlers end up in the same list — order-preserving so the dispatch
    # contract (handlers fire in registration order) is observable.
    assert s._handlers == [leak_handler, waste_handler]

    event = _make_event()
    s._run_handlers(event)
    assert seen_leak == [event]
    assert seen_waste == [event]


def test_on_leak_and_on_waste_are_same_method_object() -> None:
    """``on_waste`` is assigned as ``on_leak`` in the class body, so the
    underlying function object is identical. The bound-method wrappers on an
    instance compare equal because they share the same ``__func__``."""
    # Class-level: same function object.
    assert Sentinel.on_waste is Sentinel.on_leak

    # Instance-level: bound methods compare equal, share the same __func__
    # and __self__.
    s = Sentinel(project="proj")
    assert s.on_waste == s.on_leak
    assert s.on_waste.__func__ is s.on_leak.__func__
    assert s.on_waste.__self__ is s


def test_both_decorators_on_same_callable_register_twice() -> None:
    """Stacking ``@sentinel.on_leak`` and ``@sentinel.on_waste`` on the same
    callable is legal: each decorator call appends to ``_handlers``. The same
    function therefore appears twice and fires twice per event — the
    handler-list contract is "append on every call", and the aliases share
    that contract."""
    s = Sentinel(project="proj")
    calls: list[LeakEvent] = []

    @s.on_leak
    @s.on_waste
    def shared(ev: LeakEvent) -> None:
        calls.append(ev)

    # Two registrations from two decorator applications.
    assert s._handlers.count(shared) == 2
    assert len(s._handlers) == 2

    event = _make_event()
    s._run_handlers(event)
    # Fires once per registration — same handler, two slots.
    assert calls == [event, event]
