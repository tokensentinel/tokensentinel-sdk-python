"""Rule: agent silent but still firing API calls."""

from __future__ import annotations

from datetime import timedelta

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule


class ZombieRule(Rule):
    name = "zombie"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        threshold_min = self.get("threshold_minutes", 5)
        min_recent_calls = self.get("min_recent_calls", 5)

        if len(session) < min_recent_calls:
            return None

        last_user_facing = next((c for c in reversed(session) if c.user_facing_output), None)
        if last_user_facing is None:
            return None

        now = session[-1].timestamp
        elapsed = now - last_user_facing.timestamp
        if elapsed < timedelta(minutes=threshold_min):
            return None

        recent_calls = sum(
            1 for c in session if (now - c.timestamp) <= timedelta(minutes=threshold_min)
        )
        if recent_calls < min_recent_calls:
            return None

        return LeakEvent(
            type="zombie",
            confidence=0.75,
            project=project,
            session_id=session[-1].session_id,
            rule="v0.zombie",
            evidence={
                "minutes_since_user_facing_output": round(elapsed.total_seconds() / 60, 1),
                "recent_calls": recent_calls,
            },
            estimated_burn=round(recent_calls * 0.005, 4),
            suggested_action="kill_session_or_request_user_input",
        )
