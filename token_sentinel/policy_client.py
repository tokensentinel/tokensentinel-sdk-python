"""Policy client — pulls intervention policy from the cloud control plane.

The client runs as a single daemon thread that periodically fetches the
current policy for a project from the cloud's ``GET /v1/policy?project=…``
endpoint, caches the result in memory, and exposes ``current()`` for
in-process consumers (``Sentinel.record_call``) to consult on every call.

Discipline (mirrors :class:`token_sentinel.cloud_client.CloudSink`):
    * The agent's call path NEVER blocks on this client. ``current()`` is a
      cached attribute read — sub-microsecond.
    * The client NEVER raises into user code. All exceptions are either
      swallowed (and logged via ``warnings.warn(..., RuntimeWarning)``) or
      contained within the daemon thread.
    * Stdlib only — ``urllib.request`` for HTTP. The SDK is keeping its
      zero-dep core; adding ``requests`` here would cost users a transitive
      dep just to enable the cloud feature.

Polling cadence:
    * **Active** (any session marked active via :meth:`mark_session_active`):
      ``active_poll_seconds`` (default 2.0s).
    * **Idle** (no active sessions): ``idle_poll_seconds`` (default 5.0s).

The active/idle distinction matters because the operator-driven kill-switch
must propagate to running agents quickly (≤4s P99 lag with a 2s active
poll), but a Sentinel sitting idle in a long-lived process should not burn
HTTP volume against the cloud control plane.

Failure handling:
    * Network failure during a poll: keep the last-good policy until its TTL
      expires. After TTL, behaviour depends on ``failure_mode``:
        - ``'open'`` (default): ``current()`` returns ``None`` so callers
          see "no policy" and proceed as if the policy plane were absent.
        - ``'closed'``: ``current()`` returns a synthetic "blocked" policy
          (kill_switch_active=True). High-stakes deployments that would
          rather halt than accept policy ambiguity opt into this.
    * 401 Unauthorized: stop polling, emit a one-time warning, treat as
      no-policy. The customer needs to fix their api_key — retrying every 2
      seconds against a permanently-failing auth would just be log spam.
    * 5xx Server Error: exponential backoff (1s, 2s, 4s caps). After the
      backoff, resume normal polling; honour TTL throughout so a long
      outage gracefully transitions to the failure-mode handling.

Wire format::

    GET /v1/policy?project=<project>
    Authorization: Bearer <api_key>
    User-Agent: token-sentinel-py/<version>

    Response 200:
    {
      "policy_version": 42,
      "ttl_seconds": 60,
      "fetched_at": "2026-05-09T12:34:56Z",
      "budget_usd_per_session": 0.50,    # null = unlimited
      "max_tokens_per_min": 100000,       # null = unlimited
      "kill_switch_active": false
    }
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# 5xx backoff schedule. Sleeps fire after a *failure*, capped by the third
# value so a sustained outage doesn't get worse than 4-second polls.
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

# Generous timeout — cloud is intended to respond in <100ms but transient
# congestion shouldn't blow up the daemon thread. Same value as CloudSink.
_REQUEST_TIMEOUT_SECONDS = 10.0

# How long the daemon will block on its stop-event between polls. Keeping the
# wait granular (vs. one long sleep) lets ``stop()`` interrupt the daemon
# promptly without a dedicated signalling primitive.
_STOP_WAIT_GRANULARITY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """A single snapshot of the cloud-pushed intervention policy.

    Immutable so callers can't accidentally mutate the cached policy. The
    fields are deliberately optional (``None`` ⇒ unlimited) so a project
    that hasn't configured a budget gets a policy with no enforcement
    pressure rather than a policy that silently falls back to a hard-coded
    cap.

    Attributes:
        policy_version: monotonically-increasing integer; bumps any time a
            field changes. Useful for cache invalidation and audit.
        budget_usd_per_session: per-session USD cap. ``None`` means no
            session budget; the SDK won't track per-session burn at all in
            that case.
        max_tokens_per_min: project-wide tokens/min ceiling for the
            absolute-ceiling velocity check. ``None`` means no velocity cap.
        kill_switch_active: when ``True``, every ``record_call`` raises
            :class:`KillSwitchActive` — the operator has hit the dashboard
            kill-switch.
        fetched_at: when the policy was retrieved (UTC). Used together with
            ``ttl_seconds`` to decide whether the cache is still fresh
            during a network outage.
        ttl_seconds: policy validity window. Beyond this, the
            ``failure_mode`` controls behaviour.
    """

    policy_version: int
    budget_usd_per_session: float | None
    max_tokens_per_min: int | None
    kill_switch_active: bool
    fetched_at: datetime
    ttl_seconds: int

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return ``True`` when the policy's TTL has elapsed.

        Used by :meth:`PolicyClient.current` to decide whether a cached
        policy survives a network outage. The check is timezone-aware to
        avoid the classic naive-vs-aware datetime gotcha.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        # Defensive: if fetched_at is naive (shouldn't be, but cloud could
        # ship a malformed payload that we patched up), treat it as UTC.
        ref = self.fetched_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        elapsed = (now - ref).total_seconds()
        return elapsed > self.ttl_seconds


# A synthetic policy used in ``failure_mode='closed'`` when no last-good
# policy is available. Kill-switch on, no budget, no velocity cap — the
# kill-switch alone is enough to halt all calls.
def _build_blocked_policy() -> Policy:
    return Policy(
        policy_version=-1,
        budget_usd_per_session=None,
        max_tokens_per_min=None,
        kill_switch_active=True,
        fetched_at=datetime.now(timezone.utc),
        ttl_seconds=0,
    )


# ---------------------------------------------------------------------------
# Wire parsing
# ---------------------------------------------------------------------------


def _parse_policy(payload: dict[str, Any]) -> Policy:
    """Turn a JSON-decoded response body into a :class:`Policy` instance.

    Missing or malformed fields fall back to a permissive default
    (``None`` for caps, ``False`` for the kill-switch). This is deliberate:
    if the cloud ships a partial response — say, a future schema field
    we don't recognise — we'd rather degrade to "no policy enforced"
    than crash the daemon.

    The ``policy_version`` is required (it's the cache key). If absent, we
    fall back to 0 so the parser still produces a Policy and the cloud
    operator gets a chance to fix the response before users notice.

    ``fetched_at`` accepts ISO-8601 strings (with or without timezone) and
    falls back to "now" if missing or unparseable. The TTL countdown then
    starts from "now" in the worst case, which is the safer fallback —
    overestimating freshness would let stale policies survive longer than
    intended.
    """
    # policy_version: required, default 0.
    try:
        policy_version = int(payload.get("policy_version", 0))
    except (TypeError, ValueError):
        policy_version = 0

    # budget_usd_per_session: optional float; None = unlimited.
    budget_raw = payload.get("budget_usd_per_session")
    budget: float | None
    if budget_raw is None:
        budget = None
    else:
        try:
            budget = float(budget_raw)
            # Negative budgets are nonsensical — treat as None so the
            # comparison logic in record_call doesn't fire spuriously.
            if budget < 0:
                budget = None
        except (TypeError, ValueError):
            budget = None

    # max_tokens_per_min: optional int; None = unlimited.
    velocity_raw = payload.get("max_tokens_per_min")
    velocity: int | None
    if velocity_raw is None:
        velocity = None
    else:
        try:
            velocity = int(velocity_raw)
            if velocity < 0:
                velocity = None
        except (TypeError, ValueError):
            velocity = None

    # kill_switch_active: bool, default False.
    kill = bool(payload.get("kill_switch_active", False))

    # ttl_seconds: int, default 60. We clamp to a sane minimum so a
    # misconfigured cloud doesn't push a 0-second TTL that immediately
    # invalidates the cache on every poll.
    try:
        ttl = int(payload.get("ttl_seconds", 60))
    except (TypeError, ValueError):
        ttl = 60
    if ttl < 1:
        ttl = 60

    # fetched_at: ISO string, default "now".
    fetched_at_raw = payload.get("fetched_at")
    fetched_at: datetime
    if isinstance(fetched_at_raw, str):
        try:
            # Python's fromisoformat in 3.11+ handles "Z" suffix; for older
            # 3.10 we transform it manually. We aim to keep this stdlib-only.
            iso = fetched_at_raw.replace("Z", "+00:00")
            fetched_at = datetime.fromisoformat(iso)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        except ValueError:
            fetched_at = datetime.now(timezone.utc)
    else:
        fetched_at = datetime.now(timezone.utc)

    return Policy(
        policy_version=policy_version,
        budget_usd_per_session=budget,
        max_tokens_per_min=velocity,
        kill_switch_active=kill,
        fetched_at=fetched_at,
        ttl_seconds=ttl,
    )


# ---------------------------------------------------------------------------
# PolicyClient
# ---------------------------------------------------------------------------


class PolicyClient:
    """Daemon-thread policy poller for the Intervention Pack.

    Public surface:
        * :meth:`start` — kick off the daemon thread. Idempotent.
        * :meth:`stop` — signal the daemon to exit and join it.
        * :meth:`current` — return the current cached :class:`Policy` (or
          ``None`` when no policy is in effect / available). Sub-microsecond
          attribute read; safe to call from the hot path on every call.
        * :meth:`mark_session_active` / :meth:`mark_session_idle` — flip
          the poller into its 2s / 5s cadence.

    Failure isolation:
        * Network failure → keep last-good policy until TTL; then
          ``failure_mode`` decides.
        * 401 → stop polling; warn once.
        * 5xx → exponential backoff per :data:`_BACKOFF_SECONDS`.
        * Unexpected exception in the daemon loop → swallowed + warn; the
          daemon keeps going. Better to silently lose one poll cycle than
          to take the agent down.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        project: str,
        *,
        active_poll_seconds: float = 2.0,
        idle_poll_seconds: float = 5.0,
        failure_mode: Literal["open", "closed"] = "open",
        sdk_version: str = "",
    ) -> None:
        if not endpoint:
            raise ValueError("PolicyClient: endpoint is required")
        if not api_key:
            raise ValueError("PolicyClient: api_key is required")
        if failure_mode not in ("open", "closed"):
            raise ValueError(
                f"PolicyClient: failure_mode must be 'open' or 'closed', got {failure_mode!r}"
            )

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.project = project
        self.active_poll_seconds = float(active_poll_seconds)
        self.idle_poll_seconds = float(idle_poll_seconds)
        self.failure_mode = failure_mode
        self.sdk_version = sdk_version

        # Pre-build the URL once. ``urllib.parse.quote`` so a project name
        # with spaces / unicode survives the GET.
        encoded_project = urllib.parse.quote(self.project, safe="")
        self._poll_url = f"{self.endpoint}/v1/policy?project={encoded_project}"

        # The cached policy. ``None`` means "no policy in effect" — either
        # the daemon hasn't fetched yet, the policy expired and we're in
        # ``failure_mode='open'``, or auth failed.
        # Read on the hot path; written only by the daemon thread + stop().
        # Python attribute reads/writes are atomic w.r.t. the GIL, so the
        # hot-path read needs no lock.
        self._policy: Policy | None = None

        # Active session count drives the poll cadence. We track session
        # IDs (not just a count) so duplicate active/idle calls are
        # idempotent and we don't drift to a negative count under bugs.
        self._active_sessions: set[str] = set()
        self._sessions_lock = threading.Lock()

        # Daemon synchronisation. ``_stop`` is the canonical termination
        # signal; ``_started`` guards against double-starts.
        self._stop = threading.Event()
        self._started = False
        self._thread: threading.Thread | None = None

        # One-time gates for warnings — we don't want to spam stderr if a
        # misconfigured agent loops forever against a 401 or a flaky
        # endpoint.
        self._auth_failure_warned = False
        self._auth_failed = False  # latched: stop polling on 401

        # Track consecutive 5xx so backoff escalates correctly. Reset on
        # any successful poll.
        self._consecutive_5xx = 0

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start the daemon thread. Idempotent.

        We perform an immediate first poll INSIDE the daemon thread (not
        synchronously here) so ``start()`` returns instantly and never
        blocks on the network — important because Sentinel construction
        is on the user's main path.

        The daemon's first iteration runs without any pre-poll wait so
        ``current()`` becomes useful as soon as the cloud responds, rather
        than after the first ``active_poll_seconds`` interval.
        """
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"token-sentinel-policy-client-{self.project}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the daemon to exit and join it.

        Returns ``True`` on a clean stop (thread joined within ``timeout``),
        ``False`` if the timeout was exceeded. Either way, the client is no
        longer usable after this returns.

        Symmetric to :meth:`CloudSink.close` — both sinks/clients run as
        daemon threads and process exit will reap them, but long-running
        agents that need a deterministic shutdown should call ``stop``.
        """
        if not self._started:
            return True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                warnings.warn(
                    f"TokenSentinel PolicyClient: stop() timeout after {timeout}s; "
                    "daemon thread still running.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return False
        return True

    # -----------------------------------------------------------------
    # Hot-path reads
    # -----------------------------------------------------------------

    def current(self) -> Policy | None:
        """Return the currently-effective policy (or ``None``).

        This is the hot-path read. It must be sub-microsecond — no locking,
        no IO, no serialisation work. The daemon thread updates
        ``self._policy`` atomically (Python attribute writes are atomic
        w.r.t. the GIL), so a hot-path reader either sees the previous
        snapshot or the new one — never a half-written instance.

        Behaviour when the cached policy has expired:
            * ``failure_mode='open'`` → return ``None``. Callers treat this
              as "no policy in effect" and proceed without enforcement.
            * ``failure_mode='closed'`` → return a synthetic blocked policy
              so the next call halts at :class:`KillSwitchActive`.
        """
        policy = self._policy
        if policy is None:
            # Auth latched off, or daemon hasn't fetched yet, or fail-open
            # after expiry.
            if self.failure_mode == "closed" and self._auth_failed:
                # Auth failure under closed mode means "we can't talk to
                # cloud, halt everything." Build the blocked policy on
                # the fly so it always reflects the current time.
                return _build_blocked_policy()
            return None
        # Cached policy exists. Check freshness.
        if policy.is_expired():
            if self.failure_mode == "closed":
                return _build_blocked_policy()
            return None
        return policy

    def mark_session_active(self, session_id: str) -> None:
        """Mark a session as active so the daemon polls at the active rate.

        Idempotent — calling twice with the same session_id is a no-op.
        Thread-safe — the underlying set is mutated under a lock so
        concurrent active/idle transitions don't lose updates.
        """
        with self._sessions_lock:
            self._active_sessions.add(session_id)

    def mark_session_idle(self, session_id: str) -> None:
        """Mark a session as idle. Idempotent and thread-safe.

        Removing a session that wasn't tracked is a no-op (we use
        ``discard``, not ``remove``) — this matters because the SDK
        wrapper marks-active on every call but only marks-idle on
        explicit session lifecycle hooks.
        """
        with self._sessions_lock:
            self._active_sessions.discard(session_id)

    def _has_active_sessions(self) -> bool:
        with self._sessions_lock:
            return bool(self._active_sessions)

    # -----------------------------------------------------------------
    # Daemon loop
    # -----------------------------------------------------------------

    def _run(self) -> None:
        """Main daemon loop: poll → cache → wait → repeat.

        Termination: ``self._stop.set()`` signals exit; we drop the current
        wait and return. Failures are swallowed inside ``_poll_once`` so
        a transient cloud blip never tears the daemon down.
        """
        # Immediate first poll — don't make the customer wait an interval
        # for the cache to populate.
        if not self._auth_failed:
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001 — daemon hardening
                warnings.warn(
                    f"TokenSentinel PolicyClient: unexpected error in initial poll: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        while not self._stop.is_set():
            # Auth latched off → we still keep the thread alive (so stop()
            # works cleanly) but don't spin against the failed endpoint.
            if self._auth_failed:
                # Long sleep, interruptible by stop.
                if self._stop.wait(self.idle_poll_seconds):
                    return
                continue

            # Determine the wait based on active/idle and any backoff
            # currently in effect.
            interval = self._next_poll_interval()
            if self._stop.wait(interval):
                return

            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"TokenSentinel PolicyClient: unexpected error in poll: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _next_poll_interval(self) -> float:
        """Compute the seconds to wait before the next poll.

        Three inputs:
            * Active vs. idle (``active_poll_seconds`` vs.
              ``idle_poll_seconds``).
            * 5xx backoff: if the last poll was a 5xx, escalate to the
              backoff schedule. We take the max of the regular interval
              and the backoff so a 5xx never *shortens* the next poll.
        """
        base = self.active_poll_seconds if self._has_active_sessions() else self.idle_poll_seconds
        if self._consecutive_5xx > 0:
            # Index into _BACKOFF_SECONDS, capped at the last value so a
            # sustained outage stays at the cap rather than overflowing.
            idx = min(self._consecutive_5xx - 1, len(_BACKOFF_SECONDS) - 1)
            backoff = _BACKOFF_SECONDS[idx]
            return max(base, backoff)
        return base

    # -----------------------------------------------------------------
    # Single poll
    # -----------------------------------------------------------------

    def _poll_once(self) -> None:
        """Fetch the policy once. Updates ``self._policy`` on success.

        Errors are translated to internal state changes, NOT raised:
            * 401 → latch ``_auth_failed=True``, warn once.
            * 5xx → bump ``_consecutive_5xx`` for backoff.
            * Other network errors → log via ``warnings.warn`` and keep
              the previous cache. The hot-path consumer's TTL check will
              eventually drop the stale cache.

        On success: parse the body, atomically swap ``self._policy``,
        reset ``_consecutive_5xx`` to 0.
        """
        try:
            body = self._fetch()
        except urllib.error.HTTPError as exc:
            self._handle_http_error(exc)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # Network-layer failures are non-fatal; the cache survives
            # until TTL.
            warnings.warn(
                f"TokenSentinel PolicyClient: poll failed (network): {exc!r}; "
                f"keeping cached policy until TTL.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object, got {type(payload).__name__}")
            policy = _parse_policy(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"TokenSentinel PolicyClient: failed to parse policy response: "
                f"{exc!r}; keeping cached policy.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        # Successful poll — atomic swap, clear backoff.
        self._policy = policy
        self._consecutive_5xx = 0

    def _handle_http_error(self, exc: urllib.error.HTTPError) -> None:
        """Translate an HTTPError into the appropriate internal state."""
        status = getattr(exc, "code", 0) or 0
        if status == 401:
            if not self._auth_failure_warned:
                self._auth_failure_warned = True
                warnings.warn(
                    "TokenSentinel PolicyClient: 401 Unauthorized — stopping "
                    "policy polling. Check your api_key.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._auth_failed = True
            # Wipe any cached policy so ``current()`` returns None
            # (or the blocked policy under failure_mode='closed').
            self._policy = None
            return
        if 500 <= status < 600:
            self._consecutive_5xx += 1
            warnings.warn(
                f"TokenSentinel PolicyClient: server error {status}; backing "
                f"off (attempt {self._consecutive_5xx}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        # Other HTTP error (4xx that isn't 401, e.g. 404 if the project
        # isn't provisioned) — log and continue. Keep the cache intact;
        # the operator may be in the middle of provisioning.
        warnings.warn(
            f"TokenSentinel PolicyClient: HTTP {status}: {exc!r}; keeping cached policy.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _fetch(self) -> bytes:
        """Single GET request to the policy endpoint.

        Raises whatever urllib raises — the caller handles the translation
        to internal state.
        """
        req = urllib.request.Request(
            url=self._poll_url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": f"token-sentinel-py/{self.sdk_version}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            status = getattr(resp, "status", None)
            if status is None:
                getter = getattr(resp, "getcode", None)
                status = getter() if callable(getter) else 200
            if not (200 <= int(status) < 300):
                # Surface as HTTPError so _handle_http_error gets to do the
                # status-class dispatch above.
                raise urllib.error.HTTPError(
                    self._poll_url,
                    status,
                    f"unexpected status {status}",
                    resp.headers,
                    None,
                )
            return resp.read()  # type: ignore[no-any-return]
