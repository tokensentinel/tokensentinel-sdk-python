"""Rule: same call retried many times in a window."""

from __future__ import annotations

from collections import Counter

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule


class RetryStormRule(Rule):
    name = "retry_storm"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None
        window = self.get("window_seconds", 30)
        min_retries = self.get("min_retries", 5)

        now = session[-1].timestamp
        recent = [c for c in session if (now - c.timestamp).total_seconds() <= window]
        if not recent:
            return None

        counts = Counter(c.request_hash for c in recent)
        for hash_, count in counts.items():
            if count >= min_retries:
                samples = [c for c in recent if c.request_hash == hash_]
                wasted_tokens = sum(c.prompt_tokens + c.completion_tokens for c in samples)
                return LeakEvent(
                    type="retry_storm",
                    confidence=0.9,
                    project=project,
                    session_id=session[-1].session_id,
                    rule="v0.retry_storm",
                    evidence={
                        "request_hash": hash_[:16],
                        "retry_count": count,
                        "window_seconds": window,
                        "model": samples[0].model,
                    },
                    estimated_burn=round(wasted_tokens * 9e-6, 4),
                    suggested_action="add_backoff_or_check_upstream_health",
                )
        return None
