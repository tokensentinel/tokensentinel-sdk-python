"""Concurrency / locking tests for ``Sentinel._handlers``.

These tests cover the HIGH-severity finding "Sentinel._handlers is mutated
without a lock" from the 2026-05-07 code review:

  - ``on_leak`` and ``unregister`` mutate ``_handlers`` from arbitrary threads.
  - ``_run_handlers`` (and the legacy ``_dispatch``) iterate ``_handlers`` and
    can race with mutators.

The fix wraps both with a ``threading.Lock`` and snapshots the handler list
before invocation, so handler bodies that call back into ``on_leak`` /
``unregister`` cannot deadlock and cannot see partial list state.

Note: CPython's GIL means a plain ``list.append`` is *almost* atomic in
practice, so a strict race-failure assertion would be flaky. Instead, these
tests check the post-conditions that the lock makes possible:

  - All N concurrent registrations end up in the list (no lost append).
  - Iteration during concurrent registration completes without exception.
  - Re-entrant registration / unregistration from inside a handler works.
  - ``unregister`` semantics (returns True/False, removes exactly once).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from token_sentinel import Sentinel
from token_sentinel.events import LeakEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    type_: str = "tool_loop",
    confidence: float = 0.8,
    rule: str = "tool_loop",
) -> LeakEvent:
    return LeakEvent(
        type=type_,
        confidence=confidence,
        project="proj",
        session_id="s1",
        rule=rule,
        evidence={},
        estimated_burn=0.01,
        suggested_action="x",
        raised_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
    )


def _spawn(target, *args, n: int) -> list[threading.Thread]:
    threads = [threading.Thread(target=target, args=args) for _ in range(n)]
    for t in threads:
        t.start()
    return threads


def _join_all(threads: list[threading.Thread], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    for t in threads:
        remaining = max(0.0, deadline - time.monotonic())
        t.join(timeout=remaining)
        assert not t.is_alive(), f"thread {t.name} did not finish — possible deadlock"


# ---------------------------------------------------------------------------
# Concurrent registration
# ---------------------------------------------------------------------------


def test_concurrent_on_leak_registers_all_handlers():
    """N threads each register a distinct handler; all N must end up in the list.

    Without a lock, in-theory a torn append could lose a registration. With
    the lock + GIL, the outcome is guaranteed: ``len(_handlers) == N``.
    """
    s = Sentinel(project="proj")
    n_threads = 32
    handlers_per_thread = 8

    barrier = threading.Barrier(n_threads)

    def register_many(idx):
        # Synchronize start so all threads contend at once.
        barrier.wait(timeout=5.0)
        for j in range(handlers_per_thread):
            # Each handler is a unique closure so identity is distinct.
            def h(ev, _i=idx, _j=j):
                return None

            s.on_leak(h)

    threads = [threading.Thread(target=register_many, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    _join_all(threads)

    assert len(s._handlers) == n_threads * handlers_per_thread
    # Every handler is callable (i.e., not torn / not None).
    for h in s._handlers:
        assert callable(h)


def test_concurrent_on_leak_and_dispatch_no_exception(make_call, now):
    """Register handlers from one thread while another thread fires dispatches.

    The locked snapshot in ``_run_handlers`` means iteration sees a stable
    view; without the lock, ``RuntimeError: list modified during iteration``
    would occasionally surface. We assert no thread observed an exception
    that wasn't swallowed by the per-handler ``try/except``.
    """
    s = Sentinel(project="proj")
    n_handlers = 200
    n_dispatches = 200

    seen_exceptions: list[BaseException] = []
    stop = threading.Event()

    def register_loop():
        for i in range(n_handlers):
            try:

                def h(ev, _i=i):
                    return None

                s.on_leak(h)
            except BaseException as exc:  # noqa: BLE001
                seen_exceptions.append(exc)
        stop.set()

    def dispatch_loop():
        ev = _make_event()
        for _ in range(n_dispatches):
            try:
                s._run_handlers(ev)
            except BaseException as exc:  # noqa: BLE001
                seen_exceptions.append(exc)

    reg_thread = threading.Thread(target=register_loop)
    disp_thread = threading.Thread(target=dispatch_loop)
    reg_thread.start()
    disp_thread.start()
    _join_all([reg_thread, disp_thread])

    assert seen_exceptions == [], f"unexpected exceptions: {seen_exceptions!r}"
    # All n_handlers registrations succeeded.
    assert len(s._handlers) == n_handlers


# ---------------------------------------------------------------------------
# Re-entrant handler
# ---------------------------------------------------------------------------


def test_handler_can_call_on_leak_reentrantly():
    """A handler that registers another handler from inside itself must not
    deadlock.

    The lock is released *before* invoking handlers, so re-entrant
    registration is safe. The newly-registered handler does NOT fire for the
    in-flight event (we iterate over the snapshot that was taken before
    invocation began) — that's the documented behaviour and matches what
    most observer frameworks do.
    """
    s = Sentinel(project="proj")
    inner_calls: list[LeakEvent] = []

    def inner(ev):
        inner_calls.append(ev)

    @s.on_leak
    def outer(ev):
        # Register a new handler from inside a handler. Must not deadlock.
        s.on_leak(inner)

    ev = _make_event()
    s._run_handlers(ev)
    # Outer fired once; inner does NOT fire on this dispatch (it was added
    # mid-flight, after the snapshot was taken).
    assert inner_calls == []
    # But it IS now in the list.
    assert inner in s._handlers
    # On the next dispatch, both fire.
    s._run_handlers(ev)
    assert len(inner_calls) == 1


def test_handler_can_call_unregister_reentrantly():
    """A handler that unregisters itself (or another handler) from inside
    must not deadlock and must take effect on the *next* dispatch."""
    s = Sentinel(project="proj")
    fired: list[str] = []

    def b(ev):
        fired.append("b")

    @s.on_leak
    def a(ev):
        fired.append("a")
        # Self-unregister mid-dispatch.
        s.unregister(a)

    s.on_leak(b)

    ev = _make_event()
    s._run_handlers(ev)
    # First dispatch: both fire (snapshot taken before any handler ran).
    assert fired == ["a", "b"]
    fired.clear()

    # Second dispatch: only b remains.
    s._run_handlers(ev)
    assert fired == ["b"]


# ---------------------------------------------------------------------------
# unregister() semantics
# ---------------------------------------------------------------------------


def test_unregister_returns_true_for_registered_handler():
    s = Sentinel(project="proj")

    def h(ev):
        return None

    s.on_leak(h)
    assert s.unregister(h) is True
    assert h not in s._handlers


def test_unregister_returns_false_for_unknown_handler():
    """Unregistering a handler that was never registered is not an error —
    the method returns ``False``. Symmetric to ``set.discard``."""
    s = Sentinel(project="proj")

    def h(ev):
        return None

    # Not registered.
    assert s.unregister(h) is False
    # Register, unregister, then unregister again.
    s.on_leak(h)
    assert s.unregister(h) is True
    assert s.unregister(h) is False


def test_unregister_only_removes_one_instance_when_double_registered():
    """If a handler was registered twice (legitimate use case: same logger
    attached for two distinct purposes), ``unregister`` removes exactly one
    instance, matching ``list.remove`` semantics. Caller can call again to
    drop the second."""
    s = Sentinel(project="proj")

    def h(ev):
        return None

    s.on_leak(h)
    s.on_leak(h)
    assert s._handlers.count(h) == 2

    assert s.unregister(h) is True
    assert s._handlers.count(h) == 1
    assert s.unregister(h) is True
    assert s._handlers.count(h) == 0
    assert s.unregister(h) is False


def test_concurrent_register_and_unregister():
    """Heavy mixed traffic: half the threads register, half unregister.

    Final-state assertion: no exception escapes either method, and the
    handler list ends up in a sane state (only handlers that survived
    cleanup remain, no torn pointers).
    """
    s = Sentinel(project="proj")
    n = 50
    handlers = [lambda ev, _i=i: None for i in range(n)]

    seen_exceptions: list[BaseException] = []

    def register(idx):
        try:
            s.on_leak(handlers[idx])
        except BaseException as exc:  # noqa: BLE001
            seen_exceptions.append(exc)

    def unregister(idx):
        try:
            # Spin-wait briefly to give the registrar a chance.
            for _ in range(100):
                if handlers[idx] in s._handlers:
                    break
                time.sleep(0.0001)
            s.unregister(handlers[idx])
        except BaseException as exc:  # noqa: BLE001
            seen_exceptions.append(exc)

    threads = []
    for i in range(n):
        threads.append(threading.Thread(target=register, args=(i,)))
        threads.append(threading.Thread(target=unregister, args=(i,)))
    for t in threads:
        t.start()
    _join_all(threads)

    assert seen_exceptions == [], f"unexpected: {seen_exceptions!r}"
    # Every handler is callable; nothing is corrupted.
    for h in s._handlers:
        assert callable(h)


# ---------------------------------------------------------------------------
# Integration: concurrent record_call paths must not deadlock or crash
# ---------------------------------------------------------------------------


def test_concurrent_record_call_with_on_leak_does_not_deadlock(make_call, now):
    """End-to-end: one thread registers handlers, another fires record_calls
    that may invoke ``_run_handlers`` for any fired event. Must complete
    without timing out (the deadline assertion in ``_join_all``)."""
    import hashlib

    s = Sentinel(project="proj", rules=["embedding_waste"])

    stop = threading.Event()
    seen_exceptions: list[BaseException] = []

    def register_loop():
        for i in range(50):
            try:

                def h(ev, _i=i):
                    return None

                s.on_leak(h)
            except BaseException as exc:  # noqa: BLE001
                seen_exceptions.append(exc)
            time.sleep(0.0005)
        stop.set()

    def record_loop():
        try:
            for i in range(50):
                # Two identical embedding calls fire embedding_waste on the
                # second; subsequent identical ones also fire.
                s.record_call(
                    make_call(
                        method="embeddings.create",
                        timestamp=now + timedelta(milliseconds=i),
                        raw_request={"input": "hello"},
                        request_hash=hashlib.sha256(b"x").hexdigest(),
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            seen_exceptions.append(exc)

    reg = threading.Thread(target=register_loop)
    rec = threading.Thread(target=record_loop)
    reg.start()
    rec.start()
    _join_all([reg, rec], timeout=10.0)

    assert seen_exceptions == [], f"unexpected: {seen_exceptions!r}"
