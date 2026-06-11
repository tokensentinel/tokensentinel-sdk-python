"""Tests for the in-process ``Tracer`` (per-session ring buffer)."""

from __future__ import annotations

from datetime import timedelta

from token_sentinel.tracer import Tracer


def test_default_max_records(make_call, now):
    """Default cap is 200; records 1–199 land in buffer, 200 stays, 201 drops oldest."""
    t = Tracer()
    for i in range(250):
        t.record(make_call(session_id="s", timestamp=now + timedelta(seconds=i)))
    session = t.session("s")
    assert len(session) == 200
    # Oldest should be the 50th record (250 - 200), latest the 249th.
    assert session[0].timestamp == now + timedelta(seconds=50)
    assert session[-1].timestamp == now + timedelta(seconds=249)


def test_configurable_max_records(make_call, now):
    t = Tracer(max_records_per_session=5)
    for i in range(10):
        t.record(make_call(session_id="s", timestamp=now + timedelta(seconds=i)))
    session = t.session("s")
    assert len(session) == 5
    assert session[0].timestamp == now + timedelta(seconds=5)
    assert session[-1].timestamp == now + timedelta(seconds=9)


def test_session_returns_empty_for_unknown(make_call, now):
    t = Tracer()
    assert t.session("never-recorded") == []


def test_session_returns_copy_not_internal_buffer(make_call, now):
    """Mutating the returned list must not corrupt the tracer's internal buffer."""
    t = Tracer()
    t.record(make_call(session_id="s", timestamp=now))
    a = t.session("s")
    a.append("not a CallRecord")
    b = t.session("s")
    assert len(b) == 1
    assert isinstance(b[0], type(make_call()))


def test_record_then_clear_specific_session(make_call, now):
    t = Tracer()
    t.record(make_call(session_id="A", timestamp=now))
    t.record(make_call(session_id="B", timestamp=now))
    t.clear("A")
    assert t.session("A") == []
    assert len(t.session("B")) == 1


def test_clear_all_sessions(make_call, now):
    t = Tracer()
    t.record(make_call(session_id="A", timestamp=now))
    t.record(make_call(session_id="B", timestamp=now))
    t.clear()  # no arg = clear all
    assert t.session("A") == []
    assert t.session("B") == []
    assert list(t.all_sessions()) == []


def test_clear_unknown_session_is_noop(make_call, now):
    t = Tracer()
    t.record(make_call(session_id="A", timestamp=now))
    # Should not raise.
    t.clear("never-existed")
    assert len(t.session("A")) == 1


def test_all_sessions_lists_session_ids(make_call, now):
    t = Tracer()
    for sid in ["A", "B", "C"]:
        t.record(make_call(session_id=sid, timestamp=now))
    sessions = sorted(t.all_sessions())
    assert sessions == ["A", "B", "C"]


def test_all_sessions_returns_snapshot(make_call, now):
    """``all_sessions`` returns a snapshot list, not a live view."""
    t = Tracer()
    t.record(make_call(session_id="A", timestamp=now))
    snap = list(t.all_sessions())
    t.record(make_call(session_id="B", timestamp=now))
    # Snapshot taken before B was added must not include B.
    assert snap == ["A"]


def test_multi_session_isolation(make_call, now):
    t = Tracer()
    for i in range(3):
        t.record(make_call(session_id="A", timestamp=now + timedelta(seconds=i)))
    for i in range(5):
        t.record(make_call(session_id="B", timestamp=now + timedelta(seconds=i)))
    assert len(t.session("A")) == 3
    assert len(t.session("B")) == 5


def test_record_preserves_order(make_call, now):
    t = Tracer()
    for i in range(10):
        t.record(make_call(session_id="s", timestamp=now + timedelta(seconds=i)))
    session = t.session("s")
    assert [c.timestamp for c in session] == [now + timedelta(seconds=i) for i in range(10)]


def test_max_records_zero_edge_case(make_call, now):
    """A maxlen=0 deque silently drops every push.

    Not a configuration we recommend, but the tracer should not blow up.
    """
    # NOTE: ``deque(maxlen=0)`` is valid Python — every append is a no-op.
    t = Tracer(max_records_per_session=0)
    t.record(make_call(session_id="s", timestamp=now))
    assert t.session("s") == []
