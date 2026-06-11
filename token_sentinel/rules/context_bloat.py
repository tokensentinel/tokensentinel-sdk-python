"""Rule: prompt-tokens-per-turn slope rising past threshold."""

from __future__ import annotations

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule


class ContextBloatRule(Rule):
    name = "context_bloat"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        lookback = self.get("lookback_turns", 10)
        slope_threshold = self.get("slope_threshold", 1500)
        min_turns = self.get("min_turns", 5)

        if len(session) < min_turns:
            return None

        recent = session[-lookback:]
        # Need at least 2 points to compute a slope. We don't re-apply
        # min_turns here: the user explicitly chose `lookback` and may have
        # set lookback < min_turns intentionally to focus on the recent
        # window after a long warmup.
        if len(recent) < 2:
            return None

        prompts = [c.prompt_tokens for c in recent]
        slope = _linear_slope(prompts)

        if slope >= slope_threshold:
            # Guard divide-by-zero: a customer who sets slope_threshold=0
            # (or a very small value) shouldn't blow up the confidence calc.
            denom = slope_threshold * 4 if slope_threshold > 0 else 1
            confidence = min(0.55 + (slope - slope_threshold) / denom, 0.95)
            return LeakEvent(
                type="context_bloat",
                confidence=confidence,
                project=project,
                session_id=session[-1].session_id,
                rule="v0.context_bloat",
                evidence={
                    "tokens_per_turn_slope": round(slope, 1),
                    "first_turn_tokens": prompts[0],
                    "last_turn_tokens": prompts[-1],
                    "turns_evaluated": len(prompts),
                },
                estimated_burn=round(slope * 9e-6 * 5, 4),
                suggested_action="truncate_or_summarize_history",
            )
        return None


def _linear_slope(values: list[int]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0
