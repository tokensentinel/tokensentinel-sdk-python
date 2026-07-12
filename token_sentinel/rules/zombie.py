"""Rule: agent silent but still firing API calls."""

from __future__ import annotations

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule
from token_sentinel.rules.timeutil import elapsed_seconds


class ZombieRule(Rule):
    name = "zombie"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        threshold_min = self.get("threshold_minutes", 5)
        min_recent_calls = self.get("min_recent_calls", 5)
        if not session:
            return None
        if len(session) < min_recent_calls:
            return None

        session_id = session[-1].session_id
        # Opt-out: ``Sentinel.mark_long_running`` attaches a shared set on
        # the rule instance; config key also works for unit tests.
        long_running = getattr(self, "_long_running_sessions", None)
        if long_running is None:
            long_running = self.get("long_running_sessions", set()) or set()
        if session_id in long_running:
            return None

        now = session[-1].timestamp
        threshold_seconds = float(threshold_min) * 60.0

        last_user_facing = next((c for c in reversed(session) if c.user_facing_output), None)
        if last_user_facing is None:
            # Never produced user-facing output: anchor silence on the first
            # call in the buffer. A pure tool-only stuck agent is the strongest
            # zombie shape; previously this returned None and never fired.
            anchor = session[0]
            never_user_facing = True
        else:
            anchor = last_user_facing
            never_user_facing = False

        elapsed_sec = elapsed_seconds(now, anchor.timestamp)
        if elapsed_sec < threshold_seconds:
            return None

        recent_calls = sum(
            1 for c in session if elapsed_seconds(now, c.timestamp) <= threshold_seconds
        )
        if recent_calls < min_recent_calls:
            return None

        evidence: dict = {
            "minutes_since_user_facing_output": round(elapsed_sec / 60.0, 1),
            "recent_calls": recent_calls,
        }
        if never_user_facing:
            evidence["never_user_facing"] = True

        return LeakEvent(
            type="zombie",
            confidence=0.75,
            project=project,
            session_id=session_id,
            rule="v0.zombie",
            evidence=evidence,
            estimated_burn=round(recent_calls * 0.005, 4),
            suggested_action="kill_session_or_request_user_input",
        )
