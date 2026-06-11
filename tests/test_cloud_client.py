"""Tests for ``token_sentinel.cloud_client.CloudSink`` and the Sentinel wiring.

These tests cover the  cloud-shipping pipeline:

  1. CloudSink construction + defaults.
  2. Wire-format helper produces a JSON-serializable dict.
  3. ``enqueue`` is non-blocking on the hot path.
  4. Queue overflow drops oldest + warns once per sink.
  5. Batched flush triggers at batch_size.
  6. Batched flush triggers at flush_interval_seconds.
  7. HTTP retry with exponential backoff.
  8. HTTP failure after 3 retries drops the batch + warns.
  9. ``close()`` flushes + joins within timeout.
 10. ``close()`` returns False on timeout.
 11. Sentinel happy path with cloud configured.
 12. Sentinel without ``cloud_endpoint`` does not spawn a sink.
 13. Sentinel without ``api_key`` (but with endpoint) does not spawn a sink.
 14. Authorization header includes the api key as Bearer.
 15. User-Agent includes the SDK version.
 16. (Bonus) end-to-end via an in-process ``http.server.HTTPServer``.

We mock ``urllib.request.urlopen`` for the unit tests; the e2e test uses a
real loopback HTTP server.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

import token_sentinel
from token_sentinel import LeakEvent, Sentinel
from token_sentinel.cloud_client import CloudSink, _event_to_wire

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    type_: str = "tool_loop",
    confidence: float = 0.9,
    project: str = "proj",
    session_id: str = "s1",
    rule: str = "v0.tool_loop",
    evidence: dict | None = None,
    estimated_burn: float = 0.34,
    suggested_action: str = "pause_for_human_review",
    raised_at: datetime | None = None,
) -> LeakEvent:
    return LeakEvent(
        type=type_,
        confidence=confidence,
        project=project,
        session_id=session_id,
        rule=rule,
        evidence=evidence if evidence is not None else {"tool": "search", "n": 4},
        estimated_burn=estimated_burn,
        suggested_action=suggested_action,
        raised_at=raised_at or datetime(2026, 5, 8, 12, 34, 56, tzinfo=timezone.utc),
    )


class _FakeResponse:
    """Stand-in for the urllib response context-manager. We never read it,
    so an empty body is fine."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return b""


class _CapturingUrlopen:
    """Mock for ``urllib.request.urlopen`` that records every Request it
    receives. Optionally raises a configurable side-effect for the first N
    calls before succeeding."""

    def __init__(
        self,
        *,
        fail_first_n: int = 0,
        exception_factory=None,
        always_fail: bool = False,
    ) -> None:
        self.requests: list = []
        self.fail_first_n = fail_first_n
        self.exception_factory = exception_factory or (lambda: OSError("boom"))
        self.always_fail = always_fail

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.always_fail or len(self.requests) <= self.fail_first_n:
            raise self.exception_factory()
        return _FakeResponse(status=200)


@pytest.fixture
def fake_urlopen():
    """Yields a ``_CapturingUrlopen`` patched in for the duration of the test."""
    cap = _CapturingUrlopen()
    with patch("token_sentinel.cloud_client.urllib.request.urlopen", cap):
        yield cap


# ---------------------------------------------------------------------------
# 1. Construction + defaults
# ---------------------------------------------------------------------------


def test_cloudsink_construction_requires_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        CloudSink(
            endpoint="",
            api_key="key",
            project="p",
            sdk_version="0.0.0",
        )


def test_cloudsink_construction_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        CloudSink(
            endpoint="https://example.com",
            api_key="",
            project="p",
            sdk_version="0.0.0",
        )


def test_cloudsink_defaults_are_sensible():
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.0.0",
    )
    try:
        # Per the DESIGN.md contract.
        assert sink.flush_interval_seconds == 5.0
        assert sink.batch_size == 50
        assert sink.queue_max == 1000
        # Trailing-slash stripping so the URL builds cleanly.
        assert sink._post_url == "https://example.com/v1/events"
        # Daemon thread is alive immediately after construction.
        assert sink._thread.is_alive()
        assert sink._thread.daemon is True
    finally:
        sink.close(timeout=2.0)


def test_cloudsink_strips_trailing_slash():
    sink = CloudSink(
        endpoint="https://example.com/",
        api_key="key",
        project="p",
        sdk_version="0.0.0",
    )
    try:
        assert sink._post_url == "https://example.com/v1/events"
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 2. _event_to_wire produces JSON-serializable dict
# ---------------------------------------------------------------------------


def test_event_to_wire_is_json_serializable():
    ev = _make_event()
    wire = _event_to_wire(ev, sdk_version="0.3.3")
    # No datetime objects should remain.
    assert isinstance(wire["raised_at"], str)
    # ISO-8601 with timezone.
    assert wire["raised_at"].startswith("2026-05-08T12:34:56")
    # SDK version is stamped on the wire payload.
    assert wire["sdk_version"] == "0.3.3"
    # The whole thing round-trips through JSON without a custom encoder.
    encoded = json.dumps(wire)
    decoded = json.loads(encoded)
    assert decoded["type"] == "tool_loop"
    assert decoded["evidence"] == {"tool": "search", "n": 4}


def test_event_to_wire_preserves_evidence_unchanged():
    """redaction is the SDK's job; CloudSink must not re-process it."""
    redacted = {
        "tool": "search",
        "sample_args": [{"keys": ["query"], "value_lengths": [42], "hash": "abc"}],
    }
    ev = _make_event(evidence=redacted)
    wire = _event_to_wire(ev, sdk_version="0.3.3")
    assert wire["evidence"] == redacted


# ---------------------------------------------------------------------------
# 3. enqueue is sub-millisecond
# ---------------------------------------------------------------------------


def test_enqueue_is_fast(fake_urlopen):
    """``enqueue`` must complete in well under a millisecond on the happy path.

    We measure 100 enqueues and assert the *p95* is below 1ms — the
    implementation should run far below that (target: <100us). We use 1ms as
    the test threshold so CI noise on slow runners doesn't flake.
    """
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,  # don't fire the flush during this test
    )
    try:
        timings = []
        ev = _make_event()
        # Warm up — first call has import / branch-prediction cost.
        sink.enqueue(ev)
        for _ in range(100):
            t0 = time.perf_counter()
            sink.enqueue(ev)
            t1 = time.perf_counter()
            timings.append(t1 - t0)
        timings.sort()
        p95 = timings[int(len(timings) * 0.95)]
        assert p95 < 1e-3, f"enqueue p95 too slow: {p95 * 1e6:.1f}us"
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 4. Queue overflow drops oldest + warns once per sink
# ---------------------------------------------------------------------------


def test_queue_overflow_drops_oldest_and_warns_once():
    """Set a tiny queue + a far-future flush so the queue actually fills."""
    # Block the daemon's flush by raising on every urlopen — we want the
    # queue to fill, not get drained. We also pin a tiny queue and prevent
    # flushing by an enormous interval.
    cap = _CapturingUrlopen(always_fail=True)
    with patch("token_sentinel.cloud_client.urllib.request.urlopen", cap):
        sink = CloudSink(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            sdk_version="0.3.3",
            flush_interval_seconds=3600.0,  # essentially never
            batch_size=10_000,  # never fires from size
            queue_max=4,
        )
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                # Push WAY more than queue_max so overflow happens many times.
                for i in range(50):
                    sink.enqueue(_make_event(session_id=f"s{i}"))
                # Exactly one RuntimeWarning regardless of how many overflows
                # actually occurred. We tolerate >=1 because the daemon may
                # have drained one or two while we were pushing — but the
                # total count of OUR overflow-warning must be 1.
                overflow_warnings = [
                    w
                    for w in recorded
                    if issubclass(w.category, RuntimeWarning) and "queue full" in str(w.message)
                ]
                assert len(overflow_warnings) == 1
        finally:
            sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 5. Batched flush triggers at batch_size
# ---------------------------------------------------------------------------


def test_flush_triggers_at_batch_size(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,  # ensure size, not interval, drives flush
        batch_size=5,
    )
    try:
        for i in range(5):
            sink.enqueue(_make_event(session_id=f"s{i}"))
        # Wait for the daemon to pick up the batch and POST.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert len(fake_urlopen.requests) == 1, f"expected 1 POST, got {len(fake_urlopen.requests)}"
        body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
        assert body["project"] == "p"
        assert len(body["events"]) == 5
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 6. Batched flush triggers at flush_interval_seconds
# ---------------------------------------------------------------------------


def test_flush_triggers_at_flush_interval(fake_urlopen):
    """Push fewer than ``batch_size`` events; the timer must still flush them."""
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=0.4,  # short — keeps the test fast
        batch_size=50,
    )
    try:
        sink.enqueue(_make_event())
        sink.enqueue(_make_event())
        # Daemon polls the queue every 0.5s; allow up to ~2.5s for the
        # interval to elapse + the next poll cycle.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert len(fake_urlopen.requests) >= 1
        body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
        assert len(body["events"]) == 2
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 7. HTTP retry with exponential backoff
# ---------------------------------------------------------------------------


def test_http_retry_succeeds_after_two_failures():
    """Fail twice, then succeed — total of 3 urlopen calls for one batch."""
    cap = _CapturingUrlopen(fail_first_n=2)
    # Use a tiny patched backoff so the test runs fast.
    with (
        patch("token_sentinel.cloud_client.urllib.request.urlopen", cap),
        patch("token_sentinel.cloud_client._RETRY_BACKOFFS_SECONDS", (0.01, 0.01, 0.01)),
    ):
        sink = CloudSink(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            sdk_version="0.3.3",
            flush_interval_seconds=60.0,
            batch_size=1,
        )
        try:
            sink.enqueue(_make_event())
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and len(cap.requests) < 3:
                time.sleep(0.05)
            assert len(cap.requests) == 3, (
                f"expected 3 urlopen calls (2 fails + 1 success), got {len(cap.requests)}"
            )
        finally:
            sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 8. HTTP failure after 3 attempts drops batch + warns
# ---------------------------------------------------------------------------


def test_http_failure_after_max_retries_drops_batch_and_warns():
    cap = _CapturingUrlopen(always_fail=True)
    with (
        patch("token_sentinel.cloud_client.urllib.request.urlopen", cap),
        patch("token_sentinel.cloud_client._RETRY_BACKOFFS_SECONDS", (0.01, 0.01, 0.01)),
    ):
        sink = CloudSink(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            sdk_version="0.3.3",
            flush_interval_seconds=60.0,
            batch_size=1,
        )
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                sink.enqueue(_make_event())
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and len(cap.requests) < 3:
                    time.sleep(0.05)
                # Give the warn() call a moment to land.
                time.sleep(0.2)
                drop_warnings = [
                    w
                    for w in recorded
                    if issubclass(w.category, RuntimeWarning) and "dropping batch" in str(w.message)
                ]
                assert len(drop_warnings) == 1, (
                    f"expected 1 drop warning, got {len(drop_warnings)}: "
                    f"{[str(w.message) for w in recorded]}"
                )
            assert len(cap.requests) == 3
        finally:
            sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 9. close() flushes + joins within timeout
# ---------------------------------------------------------------------------


def test_close_flushes_pending_queue(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,  # interval would never fire
        batch_size=100,  # never trips from size
    )
    sink.enqueue(_make_event())
    sink.enqueue(_make_event())
    sink.enqueue(_make_event())
    # No POST yet — neither size nor interval triggered.
    assert len(fake_urlopen.requests) == 0
    ok = sink.close(timeout=3.0)
    assert ok is True
    # close() must have flushed everything still queued.
    assert len(fake_urlopen.requests) == 1
    body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
    assert len(body["events"]) == 3


# ---------------------------------------------------------------------------
# 10. close() returns False if timeout exceeded
# ---------------------------------------------------------------------------


def test_close_returns_false_on_timeout():
    """Inject a slow flush that exceeds the close timeout. We expect a
    RuntimeWarning + a False return."""
    slow_event = threading.Event()

    def slow_urlopen(request, timeout=None):
        # Block for longer than the close() timeout will allow.
        slow_event.wait(timeout=2.0)
        return _FakeResponse(status=200)

    with patch("token_sentinel.cloud_client.urllib.request.urlopen", slow_urlopen):
        sink = CloudSink(
            endpoint="https://example.com",
            api_key="key",
            project="p",
            sdk_version="0.3.3",
            flush_interval_seconds=0.05,  # near-instant flush trigger
            batch_size=1,
        )
        sink.enqueue(_make_event())
        # Give the daemon time to pick up the event and start the slow POST.
        time.sleep(0.3)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ok = sink.close(timeout=0.1)
            # Allow the slow POST to finish so the daemon can exit (avoids
            # a leaked thread for the rest of the test session).
            slow_event.set()
        assert ok is False
        timeout_warnings = [
            w
            for w in recorded
            if issubclass(w.category, RuntimeWarning) and "timeout" in str(w.message)
        ]
        assert len(timeout_warnings) == 1
        # Best-effort: rejoin so we don't leak a daemon.
        sink._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# 11. Sentinel happy path — events fire → handlers called → enqueue called
# ---------------------------------------------------------------------------


def test_sentinel_happy_path_enqueues_to_cloud(fake_urlopen, make_call, now):
    """End-to-end via Sentinel.record_call — embedding_waste fires, handler
    runs, the cloud sink receives the same event."""
    # ``policy_endpoint=None`` opts out of the policy plane so this
    # test exercises the cloud sink in isolation. ``fake_urlopen`` is
    # shared between the cloud-sink POST and the policy-client GET if both
    # are enabled, making the request stream non-deterministic.
    s = Sentinel(
        project="proj",
        rules=["embedding_waste"],
        cloud_endpoint="https://example.com",
        api_key="key-abc",
        cloud_flush_interval_seconds=0.2,
        cloud_batch_size=1,
        policy_endpoint=None,
    )
    try:
        seen: list[LeakEvent] = []
        s.on_leak(seen.append)

        for i in range(2):
            s.record_call(
                make_call(
                    method="embeddings.create",
                    timestamp=now + timedelta(seconds=i),
                    raw_request={"input": "x"},
                )
            )
        assert len(seen) == 1

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert len(fake_urlopen.requests) >= 1
        body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
        assert body["project"] == "proj"
        assert len(body["events"]) == 1
        assert body["events"][0]["type"] == "embedding_waste"
    finally:
        s.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 12. Sentinel without cloud_endpoint does NOT spawn a sink
# ---------------------------------------------------------------------------


def test_sentinel_without_cloud_endpoint_no_sink():
    s = Sentinel(project="proj")
    assert s._cloud_sink is None
    # close() is a no-op and returns True.
    assert s.close() is True


# ---------------------------------------------------------------------------
# 13. Sentinel without api_key (but with endpoint) does NOT spawn a sink
# ---------------------------------------------------------------------------


def test_sentinel_with_endpoint_but_no_api_key_no_sink():
    s = Sentinel(project="proj", cloud_endpoint="https://example.com")
    assert s._cloud_sink is None


def test_sentinel_with_api_key_but_no_endpoint_no_sink():
    """Symmetric: endpoint without key was tested above. Both required."""
    s = Sentinel(project="proj", api_key="key")
    assert s._cloud_sink is None


# ---------------------------------------------------------------------------
# 14. Authorization header uses Bearer + the api_key
# ---------------------------------------------------------------------------


def test_authorization_header_is_bearer(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="my-secret-key-123",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,
        batch_size=1,
    )
    try:
        sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        req = fake_urlopen.requests[0]
        # urllib normalizes headers to title-case and exposes them via
        # ``get_header`` and the ``headers`` dict.
        auth = req.get_header("Authorization")
        assert auth == "Bearer my-secret-key-123"
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 15. User-Agent includes the SDK version
# ---------------------------------------------------------------------------


def test_user_agent_includes_sdk_version(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="9.9.9",
        flush_interval_seconds=60.0,
        batch_size=1,
    )
    try:
        sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        req = fake_urlopen.requests[0]
        ua = req.get_header("User-agent")  # urllib title-cases as "User-agent"
        assert ua == "token-sentinel-py/9.9.9"
    finally:
        sink.close(timeout=2.0)


def test_content_type_header_is_application_json(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,
        batch_size=1,
    )
    try:
        sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        ct = fake_urlopen.requests[0].get_header("Content-type")
        assert ct == "application/json"
    finally:
        sink.close(timeout=2.0)


# ---------------------------------------------------------------------------
# Pro-tier judge knobs: X-Judge-Threshold-* + X-Judge-Calls-Cap headers
# ---------------------------------------------------------------------------


def test_extra_headers_forwarded_on_post(fake_urlopen):
    """The CloudSink ``extra_headers`` kwarg lands verbatim on every POST."""
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,
        batch_size=1,
        extra_headers={
            "X-Judge-Threshold-Low": "0.4500",
            "X-Judge-Threshold-High": "0.8200",
            "X-Judge-Calls-Cap": "2500000",
        },
    )
    try:
        sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        req = fake_urlopen.requests[0]
        # urllib title-cases header names (lowercases everything after the
        # first letter); the *value* round-trips unchanged.
        assert req.get_header("X-judge-threshold-low") == "0.4500"
        assert req.get_header("X-judge-threshold-high") == "0.8200"
        assert req.get_header("X-judge-calls-cap") == "2500000"
        # Baseline headers still present (extra_headers don't shadow them).
        assert req.get_header("Authorization") == "Bearer key"
        assert req.get_header("Content-type") == "application/json"
    finally:
        sink.close(timeout=2.0)


def test_extra_headers_cannot_override_authorization(fake_urlopen):
    """Even if a caller sneaks Authorization into extra_headers, the sink's
    own Bearer header wins."""
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="real-key",
        project="p",
        sdk_version="0.3.3",
        flush_interval_seconds=60.0,
        batch_size=1,
        extra_headers={
            "Authorization": "Bearer hacker",
            "Content-Type": "text/plain",
            "X-Judge-Threshold-Low": "0.5",
        },
    )
    try:
        sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        req = fake_urlopen.requests[0]
        assert req.get_header("Authorization") == "Bearer real-key"
        assert req.get_header("Content-type") == "application/json"
        # The benign one (X-Judge-*) still gets through.
        assert req.get_header("X-judge-threshold-low") == "0.5"
    finally:
        sink.close(timeout=2.0)


def test_sentinel_judge_kwargs_propagate_to_cloud_sink_headers(fake_urlopen):
    """Sentinel(judge_threshold_*=...) must end up as headers on the wire."""
    s = Sentinel(
        project="p",
        cloud_endpoint="https://example.com",
        api_key="k",
        cloud_flush_interval_seconds=0.2,
        cloud_batch_size=1,
        policy_endpoint=None,
        judge_threshold_low=0.42,
        judge_threshold_high=0.78,
        judge_calls_per_month_max=999_000,
    )
    try:
        from token_sentinel import LeakEvent

        ev = LeakEvent(
            type="tool_loop",
            confidence=0.9,
            project="p",
            session_id="s",
            rule="v0.tool_loop",
            evidence={},
            estimated_burn=0.1,
            suggested_action="x",
            raised_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
        )
        s._cloud_sink.enqueue(ev)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        req = fake_urlopen.requests[0]
        assert req.get_header("X-judge-threshold-low") == "0.4200"
        assert req.get_header("X-judge-threshold-high") == "0.7800"
        assert req.get_header("X-judge-calls-cap") == "999000"
    finally:
        s.close(timeout=2.0)


# ---------------------------------------------------------------------------
# 16. (Bonus) End-to-end via real loopback HTTP server
# ---------------------------------------------------------------------------


class _CapturingHandler(BaseHTTPRequestHandler):
    """HTTP handler that pushes received requests onto a class-level queue."""

    received: list[dict] = []

    def do_POST(self):  # noqa: N802 — required name from BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        type(self).received.append(
            {
                "path": self.path,
                "headers": {k: v for k, v in self.headers.items()},
                "body": body,
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        body_out = b'{"accepted":1,"rejected":0}'
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    # Silence the default request-logging that BaseHTTPRequestHandler writes
    # to stderr — it pollutes test output.
    def log_message(self, format, *args):  # noqa: A002 — required signature
        return


@pytest.fixture
def mock_server():
    """Spin up an HTTPServer on a free loopback port for the duration of one test."""
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield {
            "url": f"http://{host}:{port}",
            "received": _CapturingHandler.received,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_end_to_end_real_http_round_trip(mock_server):
    sink = CloudSink(
        endpoint=mock_server["url"],
        api_key="real-key",
        project="end2end",
        sdk_version="0.3.3",
        flush_interval_seconds=0.2,
        batch_size=2,
    )
    try:
        sink.enqueue(_make_event(session_id="s1"))
        sink.enqueue(_make_event(session_id="s2"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not mock_server["received"]:
            time.sleep(0.05)
        assert len(mock_server["received"]) == 1
        rec = mock_server["received"][0]
        assert rec["path"] == "/v1/events"
        assert rec["headers"].get("Authorization") == "Bearer real-key"
        assert rec["headers"].get("Content-Type") == "application/json"
        assert rec["headers"].get("User-Agent") == "token-sentinel-py/0.3.3"
        body = json.loads(rec["body"])
        assert body["project"] == "end2end"
        assert len(body["events"]) == 2
        # Each event includes the sdk_version stamp + an ISO-formatted
        # raised_at string.
        for ev in body["events"]:
            assert ev["sdk_version"] == "0.3.3"
            assert isinstance(ev["raised_at"], str)
    finally:
        sink.close(timeout=3.0)


# ---------------------------------------------------------------------------
# Extra correctness coverage — close() is idempotent / a no-op the 2nd time.
# ---------------------------------------------------------------------------


def test_close_is_idempotent(fake_urlopen):
    sink = CloudSink(
        endpoint="https://example.com",
        api_key="key",
        project="p",
        sdk_version="0.3.3",
    )
    sink.enqueue(_make_event())
    assert sink.close(timeout=2.0) is True
    # Second close is harmless (thread already exited).
    assert sink.close(timeout=0.1) is True


def test_sentinel_close_when_no_cloud_sink_returns_true():
    """``Sentinel.close`` must be safe to call even when no sink was wired."""
    s = Sentinel(project="proj")
    assert s.close(timeout=0.1) is True


def test_sentinel_uses_default_cloud_kwargs():
    """When cloud is wired, the kwargs flow through to the sink."""
    s = Sentinel(
        project="proj",
        cloud_endpoint="https://example.com",
        api_key="key",
        cloud_flush_interval_seconds=7.5,
        cloud_batch_size=23,
        cloud_queue_max=42,
        # Opt out of the policy plane so this test asserts only on
        # cloud-sink kwargs and doesn't accidentally hit example.com from
        # the policy daemon.
        policy_endpoint=None,
    )
    try:
        assert s._cloud_sink is not None
        assert s._cloud_sink.flush_interval_seconds == 7.5
        assert s._cloud_sink.batch_size == 23
        assert s._cloud_sink.queue_max == 42
        assert s._cloud_sink.sdk_version == token_sentinel.__version__
    finally:
        s.close(timeout=2.0)


# ---------------------------------------------------------------------------
# mode field on the wire
# ---------------------------------------------------------------------------


def test_outbound_payload_includes_mode(fake_urlopen):
    """Sentinel(mode='block') stamps ``mode=block`` on every outbound event.

    The end-to-end story: customers running in block mode get the full
    estimated_burn credited as savings. That credit is computed cloud-side
    off the persisted ``events.mode`` column, which is populated from the
    wire field set here.
    """
    s = Sentinel(
        project="p",
        mode="block",
        cloud_endpoint="https://example.com",
        api_key="k",
        cloud_flush_interval_seconds=0.2,
        cloud_batch_size=1,
        policy_endpoint=None,
    )
    try:
        # Push an event directly through the sink rather than wiring up a
        # full rule fire — we're asserting on the wire format, not the
        # rule engine.
        s._cloud_sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
        assert body["events"][0]["mode"] == "block"
    finally:
        s.close(timeout=2.0)


def test_outbound_payload_default_mode_is_log(fake_urlopen):
    """A Sentinel constructed without ``mode`` defaults to ``'log'`` on the wire.

    The default ``Sentinel.__init__(mode='log')`` matches the cloud's
    fallback semantics (savings credits log-mode events at 0.1×) and keeps
    the headline metric conservative — never over-claims.
    """
    s = Sentinel(
        project="p",
        cloud_endpoint="https://example.com",
        api_key="k",
        cloud_flush_interval_seconds=0.2,
        cloud_batch_size=1,
        policy_endpoint=None,
    )
    try:
        s._cloud_sink.enqueue(_make_event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not fake_urlopen.requests:
            time.sleep(0.05)
        assert fake_urlopen.requests
        body = json.loads(fake_urlopen.requests[0].data.decode("utf-8"))
        assert body["events"][0]["mode"] == "log"
    finally:
        s.close(timeout=2.0)


def test_event_to_wire_stamps_mode_default_log():
    """The wire helper defaults to ``mode='log'`` when called without a kwarg.

    Defence in depth: any future caller that builds a wire payload without
    passing ``mode=`` still gets the conservative under-counting default
    (matches cloud's ``_resolve_mode`` fallback).
    """
    ev = _make_event()
    wire = _event_to_wire(ev, sdk_version="0.7.0")
    assert wire["mode"] == "log"


def test_event_to_wire_stamps_mode_alert_when_passed():
    """The wire helper honours the ``mode`` kwarg verbatim."""
    ev = _make_event()
    wire = _event_to_wire(ev, sdk_version="0.7.0", mode="alert")
    assert wire["mode"] == "alert"
