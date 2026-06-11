"""Tests for ``RetryStormRule``.

Defaults: window_seconds=30, min_retries=5.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from token_sentinel.rules.retry_storm import RetryStormRule

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert RetryStormRule({}).evaluate([], project="p") is None


def test_single_call_no_fire(make_call, now):
    session = [make_call(timestamp=now)]
    assert RetryStormRule({}).evaluate(session, project="p") is None


def test_distinct_calls_no_fire(make_call, now):
    """5 calls in window but each has a unique hash → not retries."""
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 2),
            request_hash=hashlib.sha256(f"unique-{i}".encode()).hexdigest(),
        )
        for i in range(5)
    ]
    assert RetryStormRule({}).evaluate(session, project="p") is None


def test_below_min_retries(make_call, now):
    """4 retries < min=5 → no fire."""
    same = hashlib.sha256(b"same").hexdigest()
    session = [make_call(timestamp=now + timedelta(seconds=i), request_hash=same) for i in range(4)]
    assert RetryStormRule({}).evaluate(session, project="p") is None


def test_retries_outside_window(make_call, now):
    """5 identical calls but spread over 5 minutes — only 1 inside 30s window."""
    same = hashlib.sha256(b"same").hexdigest()
    session = [make_call(timestamp=now + timedelta(minutes=i), request_hash=same) for i in range(5)]
    assert RetryStormRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_five_identical_calls_fire(make_call, now):
    same = hashlib.sha256(b"identical").hexdigest()
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 2),
            request_hash=same,
            prompt_tokens=100,
            completion_tokens=10,
        )
        for i in range(5)
    ]
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "retry_storm"
    assert ev.rule == "v0.retry_storm"
    assert ev.confidence == 0.9
    assert ev.evidence["retry_count"] == 5
    assert ev.evidence["window_seconds"] == 30
    assert len(ev.evidence["request_hash"]) == 16
    assert ev.suggested_action == "add_backoff_or_check_upstream_health"
    # Wasted tokens = 5 * 110 = 550, burn = 550 * 9e-6 ≈ 0.005
    assert ev.estimated_burn == pytest.approx(round(550 * 9e-6, 4))


def test_burst_of_six_identical_fires(make_call, now):
    same = hashlib.sha256(b"x").hexdigest()
    session = [
        make_call(timestamp=now + timedelta(seconds=i * 3), request_hash=same) for i in range(6)
    ]
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["retry_count"] == 6


def test_two_groups_of_retries(make_call, now):
    """If two distinct hashes both reach min, rule fires once for the first found."""
    h1 = hashlib.sha256(b"one").hexdigest()
    h2 = hashlib.sha256(b"two").hexdigest()
    session = [
        make_call(timestamp=now + timedelta(seconds=i), request_hash=h1) for i in range(5)
    ] + [make_call(timestamp=now + timedelta(seconds=10 + i), request_hash=h2) for i in range(5)]
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["retry_count"] == 5


def test_intermixed_with_unique_calls(make_call, now):
    """Storm coexists with unique calls — still fires."""
    same = hashlib.sha256(b"same").hexdigest()
    session = []
    for i in range(5):
        session.append(make_call(timestamp=now + timedelta(seconds=i * 2), request_hash=same))
        session.append(
            make_call(
                timestamp=now + timedelta(seconds=i * 2 + 1),
                request_hash=hashlib.sha256(f"u-{i}".encode()).hexdigest(),
            )
        )
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_min_retries_exact(make_call, now):
    """count == min_retries fires (>= boundary)."""
    same = hashlib.sha256(b"x").hexdigest()
    session = [make_call(timestamp=now + timedelta(seconds=i), request_hash=same) for i in range(5)]
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None


def test_one_below_min_retries(make_call, now):
    same = hashlib.sha256(b"x").hexdigest()
    session = [make_call(timestamp=now + timedelta(seconds=i), request_hash=same) for i in range(4)]
    assert RetryStormRule({}).evaluate(session, project="p") is None


def test_at_window_boundary(make_call, now):
    """5 calls at 0,5,10,15,20,25,30s — last (30s) is the now-anchor.

    Rule uses ``(now - c.timestamp).total_seconds() <= window``. Calls at
    timestamps 0 through 30s relative to now anchor are all in-window when
    now=30s.
    """
    same = hashlib.sha256(b"x").hexdigest()
    session = [
        make_call(timestamp=now + timedelta(seconds=t), request_hash=same)
        for t in [0, 5, 10, 15, 20, 25, 30]
    ]
    ev = RetryStormRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["retry_count"] == 7


def test_just_outside_window(make_call, now):
    same = hashlib.sha256(b"x").hexdigest()
    session = [
        # only the last is in-window when ts is 0s, 31s, 62s, ...
        make_call(timestamp=now + timedelta(seconds=i * 31), request_hash=same)
        for i in range(5)
    ]
    assert RetryStormRule({}).evaluate(session, project="p") is None


def test_custom_min_retries(make_call, now):
    rule = RetryStormRule({"retry_storm.min_retries": 2})
    same = hashlib.sha256(b"x").hexdigest()
    session = [make_call(timestamp=now + timedelta(seconds=i), request_hash=same) for i in range(2)]
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["retry_count"] == 2


def test_custom_window(make_call, now):
    rule = RetryStormRule({"retry_storm.window_seconds": 5})
    same = hashlib.sha256(b"x").hexdigest()
    session = [
        # 5 within 5s
        make_call(timestamp=now + timedelta(seconds=i), request_hash=same)
        for i in range(5)
    ]
    ev = rule.evaluate(session, project="p")
    assert ev is not None


# ---------------------------------------------------------------------------
# False-positive hazards
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 mitigation: 5 retries in 30s implies no backoff; "
    "exponential backoff would space them further apart.",
    strict=False,
)
def test_exponential_backoff_should_not_fire(make_call, now):
    """Customer with proper backoff: spacing 1s, 2s, 4s, 8s, 16s = 31s total.

    V0 still fires because 5 are inside 30s. V1 should detect spacing pattern.
    """
    same = hashlib.sha256(b"x").hexdigest()
    session = [
        make_call(timestamp=now + timedelta(seconds=t), request_hash=same)
        for t in [0, 1, 3, 7, 15]  # all within 30s of t=15
    ]
    assert RetryStormRule({}).evaluate(session, project="p") is None
