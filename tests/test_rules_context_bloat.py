"""Tests for ``ContextBloatRule``.

Defaults: lookback_turns=10, slope_threshold=1500, min_turns=5.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from token_sentinel.rules.context_bloat import ContextBloatRule, _linear_slope

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    rule = ContextBloatRule({})
    assert rule.evaluate([], project="p") is None


def test_below_min_turns(make_call, now):
    """4 turns < min_turns=5 — no fire."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 5000)
        for i in range(4)
    ]
    assert rule.evaluate(session, project="p") is None


def test_flat_token_slope(make_call, now):
    """Tokens stay constant — slope ≈ 0, no fire."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000) for i in range(8)
    ]
    assert rule.evaluate(session, project="p") is None


def test_decreasing_token_slope(make_call, now):
    """Tokens shrinking — slope is negative, no fire."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=10000 - i * 1000)
        for i in range(8)
    ]
    assert rule.evaluate(session, project="p") is None


def test_small_positive_slope_below_threshold(make_call, now):
    """Slope of ~100 tok/turn — well below 1500."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 100)
        for i in range(8)
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_steep_rising_slope_fires(make_call, now):
    """Tokens grow by 2500/turn — well past 1500 threshold."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i * 30), prompt_tokens=2000 + i * 2500)
        for i in range(8)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "context_bloat"
    assert ev.rule == "v0.context_bloat"
    assert ev.evidence["turns_evaluated"] == 8
    assert ev.evidence["first_turn_tokens"] == 2000
    assert ev.evidence["last_turn_tokens"] == 2000 + 7 * 2500
    assert ev.evidence["tokens_per_turn_slope"] == pytest.approx(2500.0)
    assert ev.suggested_action == "truncate_or_summarize_history"
    assert ev.estimated_burn > 0
    assert 0.55 <= ev.confidence <= 0.95


def test_only_lookback_window_used(make_call, now):
    """Slope is computed over the last ``lookback_turns``, not the whole session."""
    rule = ContextBloatRule({})
    # 5 flat turns at 1000, then 8 steeply growing turns.
    flat = [make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000) for i in range(5)]
    growing = [
        make_call(
            timestamp=now + timedelta(seconds=5 + i),
            prompt_tokens=2000 + i * 2500,
        )
        for i in range(8)
    ]
    session = flat + growing
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["turns_evaluated"] == 10  # default lookback


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_at_min_turns_exactly_can_fire(make_call, now):
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 2000)
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None


def test_one_below_min_turns(make_call, now):
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 5000)
        for i in range(4)
    ]
    assert rule.evaluate(session, project="p") is None


def test_slope_at_threshold_exactly_fires(make_call, now):
    """Slope of exactly 1500 must fire (>= boundary)."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 1500)
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None


def test_slope_one_below_threshold_no_fire(make_call, now):
    """Slope of 1499 should NOT fire."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 1499)
        for i in range(5)
    ]
    assert rule.evaluate(session, project="p") is None


def test_custom_slope_threshold_override(make_call, now):
    """Stricter customer config: 100 tok/turn growth still fires."""
    rule = ContextBloatRule({"context_bloat.slope_threshold": 50})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 100)
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None


def test_custom_min_turns_override(make_call, now):
    rule = ContextBloatRule({"context_bloat.min_turns": 3})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 2000)
        for i in range(3)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None


def test_custom_lookback_truncates(make_call, now):
    """lookback_turns=3 means only last 3 turns considered.

    In a session with 5 flat turns at 1000 then 3 growing turns, slope of those
    3 should be 2500.
    """
    rule = ContextBloatRule({"context_bloat.lookback_turns": 3})
    flat = [make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000) for i in range(5)]
    growing = [
        make_call(timestamp=now + timedelta(seconds=5 + i), prompt_tokens=1000 + i * 2500)
        for i in range(3)
    ]
    ev = rule.evaluate(flat + growing, project="p")
    assert ev is not None
    assert ev.evidence["turns_evaluated"] == 3


def test_confidence_capped_at_0_95(make_call, now):
    """Even astronomic slopes shouldn't push confidence above 0.95."""
    rule = ContextBloatRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), prompt_tokens=1000 + i * 1_000_000)
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _linear_slope helper
# ---------------------------------------------------------------------------


def test_linear_slope_zero_for_constant():
    assert _linear_slope([5, 5, 5, 5]) == 0.0


def test_linear_slope_positive_for_rising():
    assert _linear_slope([0, 1, 2, 3, 4]) == pytest.approx(1.0)


def test_linear_slope_negative_for_falling():
    assert _linear_slope([4, 3, 2, 1, 0]) == pytest.approx(-1.0)


def test_linear_slope_single_value_is_zero():
    """A 1-element list should return 0.0 (n<2 guard)."""
    assert _linear_slope([42]) == 0.0


def test_linear_slope_empty_is_zero():
    assert _linear_slope([]) == 0.0


# ---------------------------------------------------------------------------
# False-positive hazards
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 mitigation: combine with completion-token entropy to detect novel work",
    strict=False,
)
def test_legitimate_long_research_should_not_fire(make_call, now):
    """Multi-step research where prompt growth is justified.

    V0 cannot distinguish — LLM-as-judge or entropy heuristic should.
    """
    rule = ContextBloatRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            prompt_tokens=2000 + i * 2500,
            completion_tokens=500,  # plenty of novel completion
            user_facing_output=True,
        )
        for i in range(8)
    ]
    assert rule.evaluate(session, project="p") is None
