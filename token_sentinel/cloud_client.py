"""Cloud sink — ships :class:`LeakEvent` instances to the TokenSentinel
backend.

The sink is structured as a fire-and-forget background pipeline:

    Sentinel._run_handlers(event)
        └── self._cloud_sink.enqueue(event)   # <100us hot path; queue.put_nowait
                 │
                 ▼
        bounded queue.Queue (default cloud_queue_max=1000)
                 │
                 ▼
        daemon thread loop  ── batches events ──▶  POST /v1/events
            │ flushes every cloud_flush_interval_seconds (default 5.0s)
            │ OR once cloud_batch_size events accumulate (default 50)
            └── HTTP retry: 3x with exponential backoff (1s, 2s, 4s); drop after.

Discipline:
    * The agent's call path NEVER blocks on this sink.
    * The sink NEVER raises into user code. All exceptions are either dropped
      silently (where loud reporting would itself disturb the agent) or
      surfaced through ``warnings.warn(..., RuntimeWarning)`` so they are
      visible without coupling to user error handling.
    * Stdlib only — ``urllib.request`` for HTTP. The SDK is keeping its
      zero-dep core; adding ``requests`` here would cost users a transitive
      dep just to enable the cloud feature.

Wire format::

    POST /v1/events
    Authorization: Bearer <api_key>
    Content-Type: application/json
    User-Agent: token-sentinel-py/<version>

    {"project": "<project>", "events": [<event-dict>, ...]}

The wire contract is small and stable. To self-host the cloud sink, point
``cloud_endpoint`` at your own server and accept this POST shape.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from token_sentinel.events import LeakEvent


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Retry policy: 3 attempts total, with exponential backoff between retries.
# Sleeps fire after a *failure*, not before the first attempt — so
# (1s, 2s, 4s) describe the delay between the four attempts.
_RETRY_BACKOFFS_SECONDS = (1.0, 2.0, 4.0)

# How long the daemon thread will block on the queue before checking the
# flush-interval timer. Keeping this short (vs. waiting on a Condition) keeps
# the implementation a single Queue without extra signalling. The cost is one
# wakeup every 0.5s when idle — negligible.
_QUEUE_POLL_TIMEOUT_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Wire serialization
# ---------------------------------------------------------------------------


def _event_to_wire(event: LeakEvent, sdk_version: str, *, mode: str = "log") -> dict[str, Any]:
    """Build the JSON-serializable dict for a single event.

    ``dataclasses.asdict`` flattens the LeakEvent for us, but ``raised_at`` is
    a ``datetime`` and the cloud expects an ISO-8601 string. We post-process
    that one field rather than write a full custom encoder — keeps the helper
    cheap and the contract obvious.

    ``mode`` (the Sentinel's ``log`` / ``alert`` / ``block`` setting) is
    stamped onto every event so the cloud's savings aggregator credits the
    right per-mode weight. Defaults to ``'log'`` for cloud back-compat —
    the field is additive on the wire, and older cloud Pydantic models
    discard unknown keys silently.

    Tag-based Chargeback: the LeakEvent's ``tags`` field (a
    ``dict[str, str]`` populated from the originating ``Session``) is
    already included by ``dataclasses.asdict`` and round-trips through
    the JSON encoder unchanged. Pre- cloud Pydantic models discard
    the field silently (``extra='ignore'``), so the SDK can ship this
    to any cloud version. Empty dict (the default for sessions opened
    without ``tags=...``) round-trips as ``{}`` — semantically identical
    to a pre- SDK omitting the field.

    Note: ``LeakEvent.evidence`` is already redacted by the  redaction
    contract. This helper does not re-process it.
    """
    payload = dataclasses.asdict(event)
    raised_at = payload.get("raised_at")
    if isinstance(raised_at, datetime):
        # Use ``isoformat`` directly — it preserves tz info if present.
        payload["raised_at"] = raised_at.isoformat()
    payload["sdk_version"] = sdk_version
    payload["mode"] = mode
    # defensive — coerce ``tags`` to a plain dict if a future
    # subclass somehow ships a non-dict. ``dataclasses.asdict`` already
    # returns the original dict by value (it doesn't deepcopy strings),
    # but an OrderedDict or a Mapping subclass would still serialize
    # correctly — this guard exists to short-circuit the surprising
    # case where a customer mutated ``LeakEvent.tags`` to a list.
    if "tags" in payload and not isinstance(payload["tags"], dict):
        payload["tags"] = {}
    return payload


# ---------------------------------------------------------------------------
# CloudSink
# ---------------------------------------------------------------------------


class CloudSink:
    """Bounded-queue + daemon-thread shipper for ``LeakEvent``s.

    The sink is intentionally simple: ``enqueue`` does a single
    ``queue.put_nowait`` (sub-millisecond), and a single daemon thread does
    everything else. We rely on the GIL plus the queue's internal mutex for
    correctness; no extra locking is required for normal traffic.

    Public surface:
        * ``enqueue(event)`` — non-blocking; safe to call from the agent's
          hot path.
        * ``close(timeout)`` — drains the queue, flushes any remaining batch,
          and joins the daemon thread. Returns True on clean shutdown,
          False if the timeout was exceeded.

    Failure isolation:
        * Queue overflow → drop oldest, emit ``RuntimeWarning`` once per
          sink instance (see ``_overflow_warning_emitted``).
        * HTTP failure (after 3 retries) → drop the batch, emit
          ``RuntimeWarning``.
        * Unexpected exception in the daemon loop → swallowed and logged via
          ``RuntimeWarning``; the daemon keeps going. Better to silently lose
          one batch than to take the agent down.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        project: str,
        sdk_version: str,
        flush_interval_seconds: float = 5.0,
        batch_size: int = 50,
        queue_max: int = 1000,
        extra_headers: dict[str, str] | None = None,
        mode: str = "log",
    ) -> None:
        if not endpoint:
            raise ValueError("CloudSink: endpoint is required")
        if not api_key:
            raise ValueError("CloudSink: api_key is required")

        # Strip a trailing slash so callers can pass either form. We append
        # ``/v1/events`` regardless.
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.project = project
        self.sdk_version = sdk_version
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.batch_size = int(batch_size)
        self.queue_max = int(queue_max)
        # stamp the Sentinel's mode onto every outbound event so the
        # cloud-side savings aggregator credits the right per-mode weight.
        # We coerce to str defensively — the public API accepts only
        # ``Literal["log", "alert", "block"]``, but a buggy caller could
        # send something else; the cloud falls back to "log" on unknown
        # values rather than reject the row.
        self.mode = str(mode) if mode else "log"
        # Pro-tier: ``extra_headers`` carries the judge tunables
        # (X-Judge-Threshold-Low/High/Calls-Cap). Passed verbatim on every
        # POST. We coerce to str on construction so a buggy caller passing
        # ints doesn't surface as a urllib TypeError mid-flight; non-string
        # values would otherwise be rejected by ``urllib.request.Request``.
        if extra_headers:
            self._extra_headers: dict[str, str] = {str(k): str(v) for k, v in extra_headers.items()}
        else:
            self._extra_headers = {}

        # Pre-build the URL once — saves a string concat per batch.
        self._post_url = f"{self.endpoint}/v1/events"

        # The queue itself is the synchronization primitive. ``maxsize``
        # bounds memory usage; on full we drop the oldest entry to make room
        # so the *newest* signal (most useful for live debugging) wins.
        self._queue: queue.Queue[LeakEvent] = queue.Queue(maxsize=self.queue_max)

        # ``threading.Event`` rather than a bool — ``wait`` lets us nudge the
        # daemon awake immediately on close().
        self._stop = threading.Event()

        # One-time gate for the overflow warning — we don't want to spam stderr
        # if a misconfigured agent fires a flood of events.
        self._overflow_warning_emitted = False

        self._thread = threading.Thread(
            target=self._run,
            name=f"token-sentinel-cloud-sink-{project}",
            daemon=True,
        )
        self._thread.start()

    # -----------------------------------------------------------------
    # Hot path — enqueue
    # -----------------------------------------------------------------

    def enqueue(self, event: LeakEvent) -> None:
        """Hand a leak event to the sink. Non-blocking.

        Discipline:
            * Must complete in <100us. We use ``put_nowait`` and handle the
              ``Full`` exception locally rather than ever blocking on the
              queue (which would be unbounded latency on a full queue).
            * Never raises. Any failure is either dropped silently or
              surfaced as a ``RuntimeWarning``.
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to make room, then re-attempt. We only warn once
            # per sink lifetime; firing on every overflow would itself become
            # the bottleneck (warnings are not free) and the customer only
            # needs to see the signal once to know to bump ``cloud_queue_max``.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                # Race: another consumer drained it between Full and get.
                # No-op — the next put_nowait will succeed.
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # Still full. Drop the new event (caller's intent: the
                # *newest* events are most valuable, but a one-off lost
                # event under heavy contention is acceptable).
                pass
            if not self._overflow_warning_emitted:
                self._overflow_warning_emitted = True
                warnings.warn(
                    f"TokenSentinel CloudSink: queue full (max={self.queue_max}); "
                    "dropping oldest event. Bump cloud_queue_max if this is "
                    "expected, or check that the cloud endpoint is reachable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception:
            # Defence-in-depth: ``put_nowait`` shouldn't raise anything else,
            # but a buggy subclass or a deserialization issue could. Swallow
            # so the agent's call path is never disturbed.
            pass

    # -----------------------------------------------------------------
    # Daemon loop
    # -----------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: drain → batch → POST.

        Termination: ``self._stop`` is set by ``close()``. We finish the
        in-flight batch, drain anything still on the queue (so callers that
        ``close(timeout=…)`` get every event delivered up to the timeout),
        and exit.
        """
        batch: list[LeakEvent] = []
        last_flush = time.monotonic()

        while not self._stop.is_set():
            timeout = max(0.0, _QUEUE_POLL_TIMEOUT_SECONDS)
            try:
                event = self._queue.get(timeout=timeout)
                batch.append(event)
            except queue.Empty:
                pass
            except Exception:
                # The Queue itself shouldn't raise other exceptions, but if
                # it does we don't want to spin a tight loop. Sleep briefly
                # and continue.
                time.sleep(0.1)
                continue

            now = time.monotonic()
            full_batch = len(batch) >= self.batch_size
            interval_elapsed = (now - last_flush) >= self.flush_interval_seconds

            if batch and (full_batch or interval_elapsed):
                self._flush(batch)
                batch = []
                last_flush = now

        # Stop requested. Drain everything still queued and flush in batches
        # so the timeout caller gets full delivery (subject to network
        # success). We deliberately reuse the same batch_size cap so a huge
        # backlog doesn't become one giant POST.
        try:
            while True:
                event = self._queue.get_nowait()
                batch.append(event)
                if len(batch) >= self.batch_size:
                    self._flush(batch)
                    batch = []
        except queue.Empty:
            pass

        if batch:
            self._flush(batch)

    # -----------------------------------------------------------------
    # Flush — turn a batch into one POST request
    # -----------------------------------------------------------------

    def _flush(self, batch: list[LeakEvent]) -> None:
        """POST a batch with retries. Never raises.

        Failure handling:
            * After ``len(_RETRY_BACKOFFS_SECONDS)`` retries, emit a
              ``RuntimeWarning`` and drop the batch.
            * Any unexpected exception is also caught and warned about — we
              never want the daemon thread to die.
        """
        if not batch:
            return
        try:
            body = self._serialize(batch)
        except Exception as exc:
            # Serialization failure is a bug, not a transient. Warn loudly so
            # the customer notices, but don't crash.
            warnings.warn(
                f"TokenSentinel CloudSink: failed to serialize batch of "
                f"{len(batch)} events: {exc!r}; dropping batch.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        last_exc: BaseException | None = None
        # Total attempts == len(backoffs); we sleep between attempts.
        for attempt in range(len(_RETRY_BACKOFFS_SECONDS)):
            try:
                self._post(body)
                return  # success
            except Exception as exc:  # noqa: BLE001 — treat all HTTP errors uniformly
                last_exc = exc
                # Sleep before the *next* attempt; do not sleep after the
                # final attempt (we're about to drop the batch anyway).
                if attempt < len(_RETRY_BACKOFFS_SECONDS) - 1:
                    delay = _RETRY_BACKOFFS_SECONDS[attempt]
                    # Use ``self._stop.wait`` so close() can wake us
                    # immediately rather than sitting on a long backoff.
                    if self._stop.wait(delay):
                        # Stop signalled during backoff — abandon retries.
                        break

        warnings.warn(
            f"TokenSentinel CloudSink: dropping batch of {len(batch)} events "
            f"after {len(_RETRY_BACKOFFS_SECONDS)} failed attempts; "
            f"last error: {last_exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _serialize(self, batch: list[LeakEvent]) -> bytes:
        """Build the JSON request body for ``batch``."""
        payload = {
            "project": self.project,
            "events": [_event_to_wire(ev, self.sdk_version, mode=self.mode) for ev in batch],
        }
        return json.dumps(payload, default=str).encode("utf-8")

    def _post(self, body: bytes) -> None:
        """Single HTTP POST. Raises on any non-2xx or network error.

        Stdlib only: ``urllib.request``. We construct a Request with the
        documented headers and let urllib raise ``HTTPError`` for non-2xx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"token-sentinel-py/{self.sdk_version}",
        }
        # ``extra_headers`` (currently the V1 X-Judge-* knobs) merge after the
        # baseline so callers can never overwrite Authorization or
        # Content-Type by accident — those come from sink construction.
        for k, v in self._extra_headers.items():
            if k.lower() not in {"authorization", "content-type"}:
                headers[k] = v
        req = urllib.request.Request(
            url=self._post_url,
            data=body,
            method="POST",
            headers=headers,
        )
        # ``urlopen`` returns a context-manager response. We don't read the
        # body — the daemon doesn't need it. Closing immediately frees the
        # underlying socket.
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                # Older urllib responses expose ``getcode()`` instead.
                getter = getattr(resp, "getcode", None)
                status = getter() if callable(getter) else 200
            if not (200 <= int(status) < 300):
                raise urllib.error.HTTPError(
                    self._post_url,
                    status,
                    f"unexpected status {status}",
                    resp.headers,
                    None,
                )

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the daemon thread and flush remaining events.

        Returns ``True`` on clean shutdown (thread joined within ``timeout``),
        ``False`` if the timeout was exceeded. Either way, the sink is no
        longer usable after this call returns.

        This method is OPTIONAL for short-lived processes — the daemon thread
        is daemonic, so process exit will reap it. Call ``close`` from
        long-running agents that want every event delivered before they go
        away.
        """
        self._stop.set()
        # Joining outside any lock — the daemon's only synchronization is
        # via the Queue (lock-free w.r.t. us) and the ``_stop`` Event.
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            warnings.warn(
                f"TokenSentinel CloudSink: close() timeout after {timeout}s; "
                "daemon thread still running (events may be lost).",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        return True
