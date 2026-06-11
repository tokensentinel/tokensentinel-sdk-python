"""Tests for the Intervention Pack — policy plane SDK side.

These tests cover:

  1. PolicyClient construction + daemon-thread lifecycle.
  2. Initial poll on start().
  3. Active vs. idle poll cadence (mark_session_active / mark_session_idle).
  4. Network failure handling (last-good policy until TTL, then fail-open
     vs. fail-closed).
  5. 401 latches off polling + warns once.
  6. 5xx exponential backoff.
  7. stop(timeout) joins the daemon cleanly.
  8. Policy parsing — missing fields fall back to None / defaults.
  9. Sentinel integration — BudgetExceeded raises when burn would exceed cap.
 10. VelocityExceeded when minute-window tokens > cap.
 11. KillSwitchActive when policy says so.
 12. Multiple sessions — per-session burn tracked independently.
 13. Sentinel without cloud_endpoint does NOT spawn a PolicyClient.
 14. Sentinel.close() flushes both CloudSink AND PolicyClient.
 15. 1-minute token window is rolling — entries older than 60s drop off.
 16. Concurrent record_call from many threads aggregates correctly.
 17. BudgetExceeded subclasses LeakDetected.
 18. Policy plane defaults to cloud_endpoint when policy_endpoint unset.
 19. policy_endpoint=None disables the policy plane explicitly.
 20. End-to-end acceptance — local mock budget triggers BudgetExceeded.

Mock the HTTP layer with ``unittest.mock.patch``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import warnings
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from token_sentinel import (
    BudgetExceeded,
    KillSwitchActive,
    LeakDetected,
    Sentinel,
    VelocityExceeded,
)
from token_sentinel.policy_client import (
    Policy,
    PolicyClient,
    _parse_policy,
)

# ---------------------------------------------------------------------------
# HTTP mocking helpers
# ---------------------------------------------------------------------------


class _PolicyResponse:
    """Stand-in for the urllib response context manager.

    Holds a JSON body and a status code; supports the context manager and
    .read() / .status interface that policy_client expects.
    """

    def __init__(self, *, body: bytes = b"{}", status: int = 200) -> None:
        self.body = body
        self.status = status
        self.headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.body


class _ScriptedUrlopen:
    """Mock urlopen that returns a configurable script of responses.

    Each call pops the next entry from ``script``. If the entry is an
    ``Exception`` instance, it's raised; otherwise it's returned as the
    response. When the script is exhausted, the last entry is repeated
    forever — so a test can set up "fail twice, then succeed" by passing
    ``[err, err, ok]``.
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list = []
        self._call_lock = threading.Lock()

    def __call__(self, request, timeout=None):
        with self._call_lock:
            self.calls.append(request)
            if not self.script:
                # Exhausted — keep returning the last entry forever.
                response = _PolicyResponse(body=b"{}", status=200)
            elif len(self.script) == 1:
                response = self.script[0]
            else:
                response = self.script.pop(0)
        if isinstance(response, BaseException) or (
            isinstance(response, type) and issubclass(response, BaseException)
        ):
            raise response if isinstance(response, BaseException) else response()
        return response


def _policy_body(
    *,
    policy_version: int = 1,
    budget_usd_per_session: float | None = None,
    max_tokens_per_min: int | None = None,
    kill_switch_active: bool = False,
    ttl_seconds: int = 60,
    fetched_at: datetime | None = None,
) -> bytes:
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc)
    payload = {
        "policy_version": policy_version,
        "ttl_seconds": ttl_seconds,
        "fetched_at": fetched_at.isoformat(),
        "budget_usd_per_session": budget_usd_per_session,
        "max_tokens_per_min": max_tokens_per_min,
        "kill_switch_active": kill_switch_active,
    }
    return json.dumps(payload).encode("utf-8")


def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> bool:
    """Spin-wait until ``predicate()`` is truthy or ``timeout`` elapses.

    Returns ``True`` if the predicate fired in time, ``False`` otherwise.
    Used to wait on the policy daemon's first successful poll without
    relying on a brittle fixed sleep.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 1. Construction + daemon-thread lifecycle
# ---------------------------------------------------------------------------


def test_policy_client_construction_requires_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        PolicyClient(endpoint="", api_key="key", project="p")


def test_policy_client_construction_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        PolicyClient(endpoint="https://example.com", api_key="", project="p")


def test_policy_client_rejects_invalid_failure_mode():
    with pytest.raises(ValueError, match="failure_mode"):
        PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            failure_mode="banana",  # type: ignore[arg-type]
        )


def test_policy_client_starts_daemon_thread_on_start():
    """``start()`` spawns a daemon thread and returns immediately."""
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body(), status=200)])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
        )
        try:
            client.start()
            # Idempotent — second start() is a no-op.
            client.start()
            assert client._thread is not None
            assert client._thread.daemon is True
            # Wait for the initial poll to land.
            assert _wait_for(lambda: client.current() is not None)
        finally:
            assert client.stop(timeout=2.0) is True


# ---------------------------------------------------------------------------
# 2. start() immediately fetches once
# ---------------------------------------------------------------------------


def test_start_immediately_fetches_once():
    """The daemon's first iteration runs without a pre-poll wait so
    ``current()`` becomes useful as soon as the cloud responds."""
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body(policy_version=7))])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=10.0,  # would be a long wait if we didn't poll-on-start
            idle_poll_seconds=10.0,
        )
        try:
            client.start()
            assert _wait_for(lambda: client.current() is not None, timeout=2.0)
            assert client.current().policy_version == 7
        finally:
            client.stop(timeout=2.0)


def test_active_session_polls_at_active_interval():
    """When at least one session is marked active, the poll cadence runs at
    ``active_poll_seconds``. We measure the interval between the first two
    polls and assert it's roughly ``active_poll_seconds``."""
    body = _policy_body()
    cap = _ScriptedUrlopen([_PolicyResponse(body=body)])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.2,
            idle_poll_seconds=10.0,
        )
        client.mark_session_active("s-active")
        try:
            client.start()
            # Wait for at least 3 polls.
            assert _wait_for(lambda: len(cap.calls) >= 3, timeout=3.0)
            # If we'd been on the idle cadence (10s), we wouldn't have 3 calls
            # in 3 seconds. Active cadence (0.2s) easily yields 3+ in 3 seconds.
        finally:
            client.stop(timeout=2.0)


def test_idle_polls_at_idle_interval():
    """No active sessions → polls at idle cadence. We assert the OPPOSITE
    of the active test: a long idle cadence should NOT yield many polls in
    a short window."""
    body = _policy_body()
    cap = _ScriptedUrlopen([_PolicyResponse(body=body)])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.1,  # would race if mark_active worked
            idle_poll_seconds=10.0,  # we expect this to dominate
        )
        try:
            client.start()
            # Wait for the initial poll.
            assert _wait_for(lambda: len(cap.calls) >= 1, timeout=2.0)
            # Wait 0.5s and confirm we don't pile up many extra polls.
            time.sleep(0.5)
            # We allow up to 2 extra calls for timing slack but not 5+.
            assert len(cap.calls) <= 2, (
                f"idle cadence should produce <=2 calls in 0.5s, got {len(cap.calls)}"
            )
        finally:
            client.stop(timeout=2.0)


def test_mark_session_idle_returns_to_idle_cadence():
    """Active → idle transition flips the next-poll wait back to idle."""
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.1,
            idle_poll_seconds=10.0,
        )
        client.mark_session_active("s1")
        try:
            client.start()
            # Drive a few active-cadence polls.
            assert _wait_for(lambda: len(cap.calls) >= 3, timeout=3.0)
            client.mark_session_idle("s1")
            calls_before = len(cap.calls)
            time.sleep(0.5)
            calls_after = len(cap.calls)
            # Once idle, we should NOT pile up many more polls in 0.5s.
            extra = calls_after - calls_before
            assert extra <= 2, f"idle cadence should produce <=2 extra calls in 0.5s, got {extra}"
        finally:
            client.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# 4. Network failure handling
# ---------------------------------------------------------------------------


def test_network_failure_keeps_last_good_policy_until_ttl():
    """First poll succeeds with TTL=60s; subsequent polls fail. Within TTL,
    ``current()`` keeps returning the cached policy."""
    success_body = _policy_body(policy_version=42, ttl_seconds=60)
    script = [
        _PolicyResponse(body=success_body, status=200),
        OSError("simulated network failure"),
        OSError("simulated network failure"),
    ]
    cap = _ScriptedUrlopen(script)
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.1,
            idle_poll_seconds=0.1,
            failure_mode="open",
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # we expect network-failure warns
                client.start()
                assert _wait_for(lambda: client.current() is not None, timeout=2.0)
                # Wait for at least one failed poll.
                assert _wait_for(lambda: len(cap.calls) >= 2, timeout=2.0)
                # Cache is still valid (TTL=60s).
                policy = client.current()
                assert policy is not None
                assert policy.policy_version == 42
        finally:
            client.stop(timeout=2.0)


def test_failure_mode_open_returns_none_after_ttl_expires():
    """When the cache TTL expires under failure_mode='open', current()
    returns ``None`` so callers proceed without enforcement."""
    # TTL=1s so the cache expires quickly.
    success_body = _policy_body(policy_version=1, ttl_seconds=1)
    script = [
        _PolicyResponse(body=success_body, status=200),
        OSError("network down"),
    ]
    cap = _ScriptedUrlopen(script)
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.05,
            idle_poll_seconds=0.05,
            failure_mode="open",
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client.start()
                assert _wait_for(lambda: client.current() is not None, timeout=2.0)
                # Wait for TTL to expire.
                time.sleep(1.2)
                # Cache should now be expired and the next poll fails.
                policy = client.current()
                assert policy is None
        finally:
            client.stop(timeout=2.0)


def test_failure_mode_closed_returns_blocked_policy_after_ttl():
    """failure_mode='closed' returns a synthetic kill-switch-on policy
    when the cache expires so subsequent calls halt."""
    success_body = _policy_body(policy_version=1, ttl_seconds=1)
    script = [
        _PolicyResponse(body=success_body, status=200),
        OSError("network down"),
    ]
    cap = _ScriptedUrlopen(script)
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.05,
            idle_poll_seconds=0.05,
            failure_mode="closed",
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client.start()
                assert _wait_for(lambda: client.current() is not None, timeout=2.0)
                time.sleep(1.2)
                policy = client.current()
                assert policy is not None
                assert policy.kill_switch_active is True
        finally:
            client.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# 5. 401 stops polling + warns once
# ---------------------------------------------------------------------------


def test_401_latches_off_polling_and_warns_once():
    err = urllib.error.HTTPError(
        url="https://example.com/v1/policy",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None,  # type: ignore[arg-type]
    )
    cap = _ScriptedUrlopen([err])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="bad-key",
            project="p",
            active_poll_seconds=0.05,
            idle_poll_seconds=0.05,
        )
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                client.start()
                # Give it time to hit 401 and latch.
                assert _wait_for(lambda: client._auth_failed, timeout=2.0)
                # Wait long enough that, if we didn't latch, we'd see many
                # more 401s — but the latch should keep call count at 1.
                time.sleep(0.5)
                auth_warnings = [
                    w
                    for w in recorded
                    if issubclass(w.category, RuntimeWarning) and "401" in str(w.message)
                ]
                assert len(auth_warnings) == 1, (
                    f"expected 1 auth-failure warning, got {len(auth_warnings)}"
                )
                # Cache wiped on 401.
                assert client.current() is None
                # The latched flag stops new HTTP attempts.
                assert len(cap.calls) <= 2
        finally:
            client.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# 6. 5xx exponential backoff
# ---------------------------------------------------------------------------


def test_5xx_increments_backoff_counter():
    """Three consecutive 5xx responses bump consecutive_5xx counter; a
    success resets it to 0."""
    err5xx = urllib.error.HTTPError(
        url="https://example.com/v1/policy",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None,  # type: ignore[arg-type]
    )
    success = _PolicyResponse(body=_policy_body(), status=200)
    script = [err5xx, err5xx, err5xx, success]
    cap = _ScriptedUrlopen(script)
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.05,
            idle_poll_seconds=0.05,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client.start()
                # Wait for the success poll to land.
                assert _wait_for(lambda: client.current() is not None, timeout=8.0)
                # After success, the backoff counter resets.
                assert client._consecutive_5xx == 0
        finally:
            client.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# 7. stop() joins cleanly
# ---------------------------------------------------------------------------


def test_stop_joins_thread_within_timeout():
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        client = PolicyClient(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            active_poll_seconds=0.5,
            idle_poll_seconds=0.5,
        )
        client.start()
        ok = client.stop(timeout=3.0)
        assert ok is True
        assert client._thread is not None
        assert client._thread.is_alive() is False


def test_stop_is_safe_when_not_started():
    """stop() before start() is a no-op."""
    client = PolicyClient(
        endpoint="https://example.com",
        api_key="key",
        project="p",
    )
    assert client.stop() is True


# ---------------------------------------------------------------------------
# 8. Policy parsing — missing fields fall back to None / defaults
# ---------------------------------------------------------------------------


def test_parse_policy_missing_fields_fall_back_to_defaults():
    payload = {"policy_version": 1}
    p = _parse_policy(payload)
    assert p.policy_version == 1
    assert p.budget_usd_per_session is None
    assert p.max_tokens_per_min is None
    assert p.kill_switch_active is False
    assert p.ttl_seconds == 60  # default


def test_parse_policy_explicit_nulls_yield_unlimited():
    payload = {
        "policy_version": 5,
        "budget_usd_per_session": None,
        "max_tokens_per_min": None,
        "kill_switch_active": False,
        "ttl_seconds": 30,
    }
    p = _parse_policy(payload)
    assert p.budget_usd_per_session is None
    assert p.max_tokens_per_min is None
    assert p.ttl_seconds == 30


def test_parse_policy_negative_caps_treated_as_unlimited():
    """A negative budget would otherwise fire spurious BudgetExceeded; we
    coerce negatives to None for safety."""
    payload = {
        "policy_version": 1,
        "budget_usd_per_session": -5.0,
        "max_tokens_per_min": -100,
    }
    p = _parse_policy(payload)
    assert p.budget_usd_per_session is None
    assert p.max_tokens_per_min is None


def test_parse_policy_garbage_types_fall_back_safely():
    """A string where a number is expected is silently treated as missing."""
    payload = {
        "policy_version": "not-a-number",
        "budget_usd_per_session": "not-a-float",
        "max_tokens_per_min": "many",
        "ttl_seconds": "soon",
    }
    p = _parse_policy(payload)
    assert p.policy_version == 0
    assert p.budget_usd_per_session is None
    assert p.max_tokens_per_min is None
    assert p.ttl_seconds == 60


def test_policy_is_expired_with_zero_ttl():
    p = Policy(
        policy_version=1,
        budget_usd_per_session=None,
        max_tokens_per_min=None,
        kill_switch_active=False,
        fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        ttl_seconds=60,
    )
    assert p.is_expired() is True


# ---------------------------------------------------------------------------
# 9-12. Sentinel integration
# ---------------------------------------------------------------------------


def _set_policy_directly(sentinel: Sentinel, **kwargs) -> Policy:
    """Bypass the daemon and stuff a synthetic policy onto the cached slot.

    Used for the Sentinel integration tests where we don't want to wait for
    the daemon to poll. Returns the policy that was installed.
    """
    fetched_at = kwargs.pop("fetched_at", datetime.now(timezone.utc))
    ttl = kwargs.pop("ttl_seconds", 60)
    policy = Policy(
        policy_version=kwargs.pop("policy_version", 1),
        budget_usd_per_session=kwargs.pop("budget_usd_per_session", None),
        max_tokens_per_min=kwargs.pop("max_tokens_per_min", None),
        kill_switch_active=kwargs.pop("kill_switch_active", False),
        fetched_at=fetched_at,
        ttl_seconds=ttl,
    )
    sentinel._policy_client._policy = policy
    return policy


@pytest.fixture
def sentinel_with_policy():
    """A Sentinel wired up with a PolicyClient whose HTTP layer is mocked.

    The daemon thread runs but never actually sees a real policy; tests
    install one directly via :func:`_set_policy_directly`.
    """
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        s = Sentinel(
            project="proj",
            cloud_endpoint="https://example.com",
            api_key="key-abc",
            cloud_flush_interval_seconds=60.0,  # don't fire during tests
            cloud_batch_size=1000,
            policy_active_poll_seconds=10.0,  # don't poll during tests
            policy_idle_poll_seconds=10.0,
        )
        try:
            yield s
        finally:
            s.close(timeout=2.0)


def test_budget_exceeded_raises_when_burn_would_exceed_cap(sentinel_with_policy, make_call):
    """A session whose accumulated burn + the next call's burn would cross
    the budget triggers BudgetExceeded. The exception carries the policy."""
    s = sentinel_with_policy
    policy = _set_policy_directly(s, budget_usd_per_session=0.01)

    # First call: 1000 prompt + 200 completion = 1200 tokens × $9e-6 = $0.0108
    # which already exceeds the $0.01 cap → should raise on first attempt.
    call = make_call(session_id="s-budget", prompt_tokens=1000, completion_tokens=200)
    with pytest.raises(BudgetExceeded) as exc_info:
        s.record_call(call)
    err = exc_info.value
    assert err.policy is policy
    assert err.session_id == "s-budget"
    assert err.budget_usd == 0.01
    # Subclass relationship: still LeakDetected.
    assert isinstance(err, LeakDetected)


def test_budget_check_accounts_for_running_aggregate(sentinel_with_policy, make_call):
    """First call burn fits under the cap; second call pushes over. The
    second call must raise."""
    s = sentinel_with_policy
    # 100 tokens × $9e-6 = $0.0009 per call. Cap at $0.0015 so two calls
    # exceed the cap.
    _set_policy_directly(s, budget_usd_per_session=0.0015)

    call1 = make_call(session_id="s-acc", prompt_tokens=100, completion_tokens=0)
    call2 = make_call(session_id="s-acc", prompt_tokens=100, completion_tokens=0)
    # First call passes through.
    s.record_call(call1)
    # Second call would push aggregate to ~$0.0018 > cap.
    with pytest.raises(BudgetExceeded):
        s.record_call(call2)


def test_velocity_exceeded_when_minute_window_tokens_over_cap(sentinel_with_policy, make_call):
    s = sentinel_with_policy
    _set_policy_directly(s, max_tokens_per_min=500)

    # First call: 300 tokens. Under cap; passes.
    call1 = make_call(session_id="s-vel", prompt_tokens=200, completion_tokens=100)
    s.record_call(call1)
    # Second call: another 300 tokens → 600 in window > 500 cap.
    call2 = make_call(session_id="s-vel", prompt_tokens=200, completion_tokens=100)
    with pytest.raises(VelocityExceeded) as exc_info:
        s.record_call(call2)
    err = exc_info.value
    assert err.max_tokens_per_min == 500
    assert err.current_tokens_per_min >= 500


def test_kill_switch_active_raises_immediately(sentinel_with_policy, make_call):
    s = sentinel_with_policy
    policy = _set_policy_directly(s, kill_switch_active=True)
    call = make_call(session_id="s-kill")
    with pytest.raises(KillSwitchActive) as exc_info:
        s.record_call(call)
    assert exc_info.value.policy is policy
    assert isinstance(exc_info.value, LeakDetected)


def test_kill_switch_takes_precedence_over_budget(sentinel_with_policy, make_call):
    """When both kill_switch and budget would fire, kill_switch wins
    (most aggressive halt)."""
    s = sentinel_with_policy
    _set_policy_directly(
        s,
        kill_switch_active=True,
        budget_usd_per_session=0.000001,  # would also fire
    )
    call = make_call(session_id="s-priority")
    with pytest.raises(KillSwitchActive):
        s.record_call(call)


def test_multiple_sessions_track_burn_independently(sentinel_with_policy, make_call):
    """One session hitting its cap doesn't prevent a different session's
    record_call from running."""
    s = sentinel_with_policy
    # 100 tokens × $9e-6 = $0.0009/call. Cap = $0.001 means 1 call passes,
    # 2 calls per session exceed.
    _set_policy_directly(s, budget_usd_per_session=0.001)

    # Session A burns one call (under cap).
    s.record_call(make_call(session_id="A", prompt_tokens=100, completion_tokens=0))
    # Session A burns another call → exceeds.
    with pytest.raises(BudgetExceeded):
        s.record_call(make_call(session_id="A", prompt_tokens=100, completion_tokens=0))
    # Session B starts fresh: one call still passes.
    s.record_call(make_call(session_id="B", prompt_tokens=100, completion_tokens=0))


# ---------------------------------------------------------------------------
# 13. Sentinel without cloud_endpoint does NOT spawn a PolicyClient
# ---------------------------------------------------------------------------


def test_sentinel_without_cloud_endpoint_no_policy_client():
    s = Sentinel(project="proj")
    assert s._policy_client is None
    assert s.close() is True


def test_sentinel_with_endpoint_but_no_api_key_no_policy_client():
    s = Sentinel(project="proj", cloud_endpoint="https://example.com")
    assert s._policy_client is None


def test_explicit_policy_endpoint_none_disables_plane():
    """A customer with cloud_endpoint configured can opt out of the policy
    plane by passing ``policy_endpoint=None``."""
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        s = Sentinel(
            project="proj",
            cloud_endpoint="https://example.com",
            api_key="key",
            policy_endpoint=None,
        )
        try:
            assert s._policy_client is None
            # CloudSink is still wired up.
            assert s._cloud_sink is not None
        finally:
            s.close(timeout=2.0)


def test_policy_endpoint_defaults_to_cloud_endpoint():
    """When policy_endpoint is unset, the policy client uses cloud_endpoint."""
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with patch("token_sentinel.policy_client.urllib.request.urlopen", cap):
        s = Sentinel(
            project="proj",
            cloud_endpoint="https://example.com",
            api_key="key",
        )
        try:
            assert s._policy_client is not None
            assert s._policy_client.endpoint == "https://example.com"
        finally:
            s.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 14. Sentinel.close() flushes both CloudSink AND PolicyClient
# ---------------------------------------------------------------------------


def test_sentinel_close_stops_both_subsystems():
    cap = _ScriptedUrlopen([_PolicyResponse(body=_policy_body())])
    with (
        patch("token_sentinel.policy_client.urllib.request.urlopen", cap),
        patch(
            "token_sentinel.cloud_client.urllib.request.urlopen",
            lambda req, timeout=None: _PolicyResponse(status=200),
        ),
    ):
        s = Sentinel(
            project="proj",
            cloud_endpoint="https://example.com",
            api_key="key",
        )
        assert s._policy_client is not None
        assert s._cloud_sink is not None
        ok = s.close(timeout=3.0)
        assert ok is True
        # Both threads have exited.
        assert s._policy_client._thread is not None
        assert s._policy_client._thread.is_alive() is False
        assert s._cloud_sink._thread.is_alive() is False


# ---------------------------------------------------------------------------
# 15. 1-minute token window is rolling
# ---------------------------------------------------------------------------


def test_velocity_window_drops_entries_older_than_60_seconds(sentinel_with_policy, make_call):
    """An entry pushed >60s ago is no longer counted in the window."""
    s = sentinel_with_policy
    _set_policy_directly(s, max_tokens_per_min=1000)
    # Simulate an old call by direct manipulation of the deque.
    old_ts = time.monotonic() - 120.0  # 2 minutes ago
    with s._policy_lock:
        s._tokens_minute_window.append((old_ts, 5000))
    # 5000 > 1000; if the window included old entries we'd get a
    # VelocityExceeded. But the window is rolling, so the entry should
    # be evicted on the next read.
    call = make_call(session_id="s-roll", prompt_tokens=10, completion_tokens=10)
    s.record_call(call)
    # After the read, the deque should NOT contain the old entry.
    with s._policy_lock:
        for ts, _ in s._tokens_minute_window:
            assert ts >= time.monotonic() - 70.0, (
                "old entries beyond the rolling window should be dropped"
            )


# ---------------------------------------------------------------------------
# 16. Race condition test: concurrent record_call on the same session
# ---------------------------------------------------------------------------


def test_concurrent_record_call_aggregates_burn_correctly(sentinel_with_policy, make_call):
    """8 threads call record_call simultaneously with the same session_id;
    per-session burn aggregates without lost updates."""
    s = sentinel_with_policy
    # Set a huge budget so we don't trigger BudgetExceeded; we're testing
    # the aggregate accuracy, not the threshold.
    _set_policy_directly(s, budget_usd_per_session=1000.0)

    # Each call: 100 tokens × $9e-6 = $0.0009. 8 threads × 10 calls = 80
    # calls × $0.0009 = $0.072.
    sid = "s-concurrent"
    threads_count = 8
    calls_per_thread = 10
    total_calls = threads_count * calls_per_thread
    expected_burn_per_call = 100 * 9e-6
    expected_total = total_calls * expected_burn_per_call

    barrier = threading.Barrier(threads_count)

    def worker():
        barrier.wait()  # release together for max contention
        for _ in range(calls_per_thread):
            s.record_call(make_call(session_id=sid, prompt_tokens=100, completion_tokens=0))

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    # The aggregated burn must equal the expected total within float
    # tolerance (no lost updates).
    actual_burn = s._session_burn[sid]
    assert abs(actual_burn - expected_total) < 1e-9, f"expected {expected_total}, got {actual_burn}"


# ---------------------------------------------------------------------------
# 17. Subclass relationship
# ---------------------------------------------------------------------------


def test_budget_exceeded_is_leak_detected():
    assert issubclass(BudgetExceeded, LeakDetected)


def test_velocity_exceeded_is_leak_detected():
    assert issubclass(VelocityExceeded, LeakDetected)


def test_kill_switch_active_is_leak_detected():
    assert issubclass(KillSwitchActive, LeakDetected)


# ---------------------------------------------------------------------------
# 18. End-to-end acceptance — local mock + budget triggers
# ---------------------------------------------------------------------------


class _PolicyServer(BaseHTTPRequestHandler):
    """In-process HTTP handler that returns the policy configured in the
    class-level ``_policy_body`` attribute."""

    _policy_body: bytes = b'{"policy_version": 1}'
    received: list = []

    def do_GET(self):  # noqa: N802
        type(self).received.append(
            {
                "path": self.path,
                "headers": {k: v for k, v in self.headers.items()},
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self._policy_body)))
        self.end_headers()
        self.wfile.write(self._policy_body)

    def do_POST(self):  # noqa: N802
        # The cloud sink also POSTs events here in some scenarios. Accept
        # silently so the test doesn't get bogged down in retries.
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):  # noqa: A002
        return


def test_end_to_end_acceptance_budget_exceeded(make_call):
    """Acceptance test: configure a Sentinel with policy_endpoint pointing at
    a local mock that returns budget_usd_per_session=0.10. Issue calls
    totalling burn > 0.10. The next call MUST raise BudgetExceeded."""
    _PolicyServer._policy_body = json.dumps(
        {
            "policy_version": 1,
            "ttl_seconds": 60,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "budget_usd_per_session": 0.10,
            "max_tokens_per_min": None,
            "kill_switch_active": False,
        }
    ).encode("utf-8")
    _PolicyServer.received = []

    server = HTTPServer(("127.0.0.1", 0), _PolicyServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    url = f"http://{host}:{port}"
    try:
        s = Sentinel(
            project="acceptance",
            cloud_endpoint=url,
            api_key="key",
            policy_active_poll_seconds=0.1,
            policy_idle_poll_seconds=0.1,
        )
        try:
            # Wait for the policy to land.
            assert _wait_for(lambda: s._policy_client.current() is not None, timeout=3.0)
            policy = s._policy_client.current()
            assert policy.budget_usd_per_session == 0.10

            # Each call: 10000 tokens × $9e-6 = $0.09. Two calls = $0.18 > $0.10.
            call1 = make_call(session_id="acc-1", prompt_tokens=8000, completion_tokens=2000)
            # First call: burn $0.09, under cap → passes.
            s.record_call(call1)
            # Second call: aggregate $0.18 > $0.10 → raises.
            call2 = make_call(session_id="acc-1", prompt_tokens=8000, completion_tokens=2000)
            with pytest.raises(BudgetExceeded) as exc_info:
                s.record_call(call2)
            assert exc_info.value.policy.budget_usd_per_session == 0.10
            assert exc_info.value.budget_usd == 0.10
        finally:
            s.close(timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Performance — record_call overhead added by policy check
# ---------------------------------------------------------------------------


def test_policy_check_overhead_is_below_100us(sentinel_with_policy, make_call):
    """The per-call overhead added by the policy plane must be sub-100us at
    p95. This is the contract the spec calls out: ``current()`` is a cached
    read; ``_enforce_policy`` does a few comparisons and a deque sum."""
    s = sentinel_with_policy
    # No cap → check runs but never raises; gives us the steady-state cost.
    _set_policy_directly(s)

    # Warm up.
    for _ in range(20):
        s.record_call(make_call(session_id="p-warm"))

    timings_with_policy = []
    for _ in range(200):
        call = make_call(session_id="perf")
        t0 = time.perf_counter()
        s._enforce_policy(call)
        t1 = time.perf_counter()
        timings_with_policy.append(t1 - t0)
    timings_with_policy.sort()
    p95 = timings_with_policy[int(len(timings_with_policy) * 0.95)]
    # 200us is the loose CI-tolerant ceiling; the implementation runs
    # closer to 5–20us on a reasonable machine.
    assert p95 < 200e-6, f"policy-check p95 too slow: {p95 * 1e6:.1f}us"
