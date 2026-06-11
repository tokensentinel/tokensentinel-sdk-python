"""Tests for ``ZombieRule``.

Defaults: threshold_minutes=5, min_recent_calls=5.

Signal: ``now() - last_user_facing_output > 5min`` AND ``≥5`` API calls in the
last 5min window.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from token_sentinel.rules.zombie import ZombieRule

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert ZombieRule({}).evaluate([], project="p") is None


def test_below_min_recent_calls(make_call, now):
    """Total session length below min_recent_calls — bail early."""
    session = [
        make_call(
            timestamp=now - timedelta(minutes=10) + timedelta(seconds=i),
            user_facing_output=(i == 0),
        )
        for i in range(4)
    ]
    assert ZombieRule({}).evaluate(session, project="p") is None


def test_no_user_facing_output_ever(make_call, now):
    """If there's never been a user-facing output, rule cannot fire."""
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "tool", "arguments": {}}],
        )
        for i in range(6)
    ]
    assert ZombieRule({}).evaluate(session, project="p") is None


def test_recent_user_facing_output(make_call, now):
    """User-facing output 1 minute ago — well under 5 min threshold."""
    base = now - timedelta(minutes=2)
    session = [
        make_call(timestamp=base, user_facing_output=True),
    ] + [
        make_call(
            timestamp=base + timedelta(seconds=10 + i * 5),
            tool_calls=[{"name": "tool", "arguments": {}}],
        )
        for i in range(5)
    ]
    assert ZombieRule({}).evaluate(session, project="p") is None


def test_old_user_facing_but_too_few_recent_calls(make_call, now):
    """Old user output but only 2 calls inside last 5 min — no fire."""
    old = now - timedelta(minutes=20)
    very_old_calls = [
        make_call(
            timestamp=old + timedelta(minutes=i),
            tool_calls=[{"name": "tool", "arguments": {}}],
        )
        for i in range(5)
    ]
    very_old_calls[0] = make_call(timestamp=old, user_facing_output=True)
    # Only one or two recent (within last 5 min relative to most-recent ts which is old + 4min)
    # All these calls are within 5 min of the last one, so actually all 5 are recent.
    # That would fire. To test the "few recent calls" branch we need a session
    # where the user output is old AND most calls are also old, with fewer than
    # min_recent_calls inside the last threshold window.
    # Build directly:
    session = [
        # very old user output
        make_call(timestamp=now - timedelta(minutes=30), user_facing_output=True),
        # 4 more very old calls so total >= min_recent_calls (5)
        *[
            make_call(
                timestamp=now - timedelta(minutes=29 - i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
            for i in range(4)
        ],
        # 2 recent calls — below min_recent_calls=5
        make_call(
            timestamp=now - timedelta(minutes=4),
            tool_calls=[{"name": "t", "arguments": {}}],
        ),
        make_call(timestamp=now, tool_calls=[{"name": "t", "arguments": {}}]),
    ]
    assert ZombieRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive case
# ---------------------------------------------------------------------------


def test_classic_zombie_fires(make_call, now):
    """User output 10 min ago, then 6 tool-call bursts in last 5 min."""
    base = now - timedelta(minutes=10)
    session = [make_call(timestamp=base, user_facing_output=True)]
    for i in range(6):
        session.append(
            make_call(
                timestamp=now - timedelta(minutes=4) + timedelta(seconds=i * 10),
                tool_calls=[{"name": "tool", "arguments": {"i": i}}],
            )
        )
    ev = ZombieRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "zombie"
    assert ev.rule == "v0.zombie"
    assert ev.confidence == 0.75
    assert ev.suggested_action == "kill_session_or_request_user_input"
    assert ev.evidence["recent_calls"] == 6
    assert ev.evidence["minutes_since_user_facing_output"] >= 5.0
    assert ev.estimated_burn > 0


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_exactly_at_threshold_fires(make_call, now):
    """At exactly threshold_minutes the elapsed condition (``< 5m``) is False,
    so the rule passes the staleness gate and fires (assuming enough recent
    calls). This validates the rule's strict-less-than guard."""
    base = now - timedelta(minutes=5)  # exactly 5 min stale at session end
    session = [make_call(timestamp=base, user_facing_output=True)]
    # Five recent tool-call records. Last one MUST land at `now` so that
    # `session[-1].timestamp - base == exactly 5min` and the staleness gate
    # uses `<` (strict less than) to admit the boundary.
    for i in range(5):
        session.append(
            make_call(
                timestamp=now - timedelta(seconds=4 - i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    ev = ZombieRule({}).evaluate(session, project="p")
    assert ev is not None


def test_just_below_threshold_no_fire(make_call, now):
    """4 min 59s stale → rule must not fire."""
    base = now - timedelta(minutes=4, seconds=59)
    session = [make_call(timestamp=base, user_facing_output=True)]
    for i in range(6):
        session.append(
            make_call(
                timestamp=now - timedelta(seconds=10 - i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    assert ZombieRule({}).evaluate(session, project="p") is None


def test_min_recent_calls_exact(make_call, now):
    """Exactly min_recent_calls (5) fires."""
    base = now - timedelta(minutes=10)
    session = [make_call(timestamp=base, user_facing_output=True)]
    # We need ≥5 recent calls. The user output is also a "call" in the session;
    # but it's outside the 5-min window so won't count toward recent_calls.
    for i in range(5):
        session.append(
            make_call(
                timestamp=now - timedelta(minutes=2) + timedelta(seconds=i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    ev = ZombieRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["recent_calls"] == 5


def test_one_below_min_recent_calls(make_call, now):
    base = now - timedelta(minutes=10)
    session = [make_call(timestamp=base, user_facing_output=True)]
    # The first user-facing call counts toward len(session), so need to pad
    # with extra OLD calls so len(session) >= min_recent_calls=5.
    for i in range(5):
        session.append(
            make_call(
                timestamp=base + timedelta(seconds=i + 1),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    # And only 4 recent calls
    for i in range(4):
        session.append(
            make_call(
                timestamp=now - timedelta(minutes=2) + timedelta(seconds=i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    assert ZombieRule({}).evaluate(session, project="p") is None


def test_custom_threshold_minutes(make_call, now):
    rule = ZombieRule({"zombie.threshold_minutes": 1})
    base = now - timedelta(minutes=2)
    session = [make_call(timestamp=base, user_facing_output=True)]
    for i in range(5):
        session.append(
            make_call(
                timestamp=now - timedelta(seconds=10 - i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    ev = rule.evaluate(session, project="p")
    assert ev is not None


def test_custom_min_recent_calls(make_call, now):
    rule = ZombieRule({"zombie.min_recent_calls": 2})
    base = now - timedelta(minutes=10)
    session = [make_call(timestamp=base, user_facing_output=True)]
    for i in range(2):
        session.append(
            make_call(
                timestamp=now - timedelta(seconds=10 - i),
                tool_calls=[{"name": "t", "arguments": {}}],
            )
        )
    ev = rule.evaluate(session, project="p")
    assert ev is not None


# ---------------------------------------------------------------------------
# False-positive hazards
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 mitigation: Sentinel.mark_long_running(session_id) opt-out",
    strict=False,
)
def test_long_running_research_should_not_fire(make_call, now):
    """Overnight research agent — V0 cannot distinguish from a stuck loop."""
    base = now - timedelta(hours=8)
    session = [make_call(timestamp=base, user_facing_output=True)]
    for i in range(20):
        session.append(
            make_call(
                timestamp=now - timedelta(minutes=4) + timedelta(seconds=i * 10),
                tool_calls=[{"name": "research_step", "arguments": {"i": i}}],
            )
        )
    assert ZombieRule({}).evaluate(session, project="p") is None
