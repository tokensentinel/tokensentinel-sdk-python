"""Tests for ``ToolLoopRule``.

Defaults: min_calls=3, window_seconds=60, cosine_threshold=0.85.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from token_sentinel.rules.tool_loop import ToolLoopRule


def _evaluate(rule, calls, project="proj"):
    return rule.evaluate(calls, project=project)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    rule = ToolLoopRule({})
    assert _evaluate(rule, []) is None


def test_single_tool_call_does_not_fire(make_call, now):
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now,
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
    ]
    assert _evaluate(rule, session) is None


def test_two_calls_below_min_calls(make_call, now):
    """min_calls defaults to 3 — two similar calls must not fire."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
        for i in range(2)
    ]
    assert _evaluate(rule, session) is None


def test_three_dissimilar_calls_no_fire(make_call, now):
    """Three calls but with disjoint argument tokens — Jaccard far below 0.85."""
    rule = ToolLoopRule({})
    arg_sets = [
        {"q": "alpha bravo charlie"},
        {"q": "delta echo foxtrot"},
        {"q": "golf hotel india"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None


def test_partial_window_only_counts_recent(make_call, now):
    """Calls at 0/30/60/90s — relative to anchor=90s, the 0s call is excluded
    (90 - 0 = 90s > 60s window). Remaining 3 identical calls fire."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 30),  # 0, 30, 60, 90
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
        for i in range(4)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    # Only 3 of the 4 are within window (the 0s call is dropped).
    assert ev.evidence["call_count"] == 3


def test_calls_outside_window_strict(make_call, now):
    """With timestamps spread far apart, only the latest call is in-window."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 100),
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
        for i in range(4)
    ]
    assert _evaluate(rule, session) is None


def test_different_tool_names_not_grouped(make_call, now):
    """Three calls in a window, but to three different tools — no fire."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": f"tool_{i}", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is None


def test_calls_without_tool_invocations_skipped(make_call, now):
    """Plain assistant messages with no tool_calls cannot fire tool_loop."""
    rule = ToolLoopRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), user_facing_output=True) for i in range(5)
    ]
    assert _evaluate(rule, session) is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_three_identical_calls_fire(make_call, now):
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "web of life game"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.type == "tool_loop"
    assert ev.rule == "v0.tool_loop"
    assert ev.evidence["tool"] == "search"
    assert ev.evidence["call_count"] == 3
    # identical args → Jaccard 1.0 → confidence min(0.6 + (1.0-0.85)*4, 0.99) = 0.99
    assert ev.confidence == pytest.approx(0.99)
    assert ev.suggested_action == "pause_for_human_review"
    assert ev.estimated_burn > 0
    assert ev.evidence["mean_similarity"] == 1.0
    assert ev.evidence["window_seconds"] == 60
    assert len(ev.evidence["sample_args"]) == 3


def test_paraphrased_args_fire_above_threshold(make_call, now):
    """Three paraphrases that share most tokens — Jaccard ≥ 0.85."""
    rule = ToolLoopRule({})
    arg_sets = [
        {"query": "web of life game"},
        {"query": "Web of Life game"},
        {"query": '"Web of Life" game'},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "web_search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["tool"] == "web_search"
    # Confidence formula: 0.6 + (sim - 0.85) * 4
    assert 0.5 < ev.confidence <= 0.99


def test_calls_via_multi_tool_response(make_call, now):
    """A single response carrying 3 identical tool_use blocks should also fire."""
    rule = ToolLoopRule({})
    args = {"q": "kittens"}
    tool_calls = [{"name": "search", "arguments": args}] * 3
    session = [
        make_call(
            timestamp=now,
            tool_calls=tool_calls,
        )
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["call_count"] == 3


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_min_calls_override(make_call, now):
    """If config sets min_calls=2, two identical calls fire."""
    rule = ToolLoopRule({"tool_loop.min_calls": 2})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
        for i in range(2)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["call_count"] == 2


def test_window_seconds_override_inclusive(make_call, now):
    """Custom window of 5s. Three calls 0/4/5s apart should still fire."""
    rule = ToolLoopRule({"tool_loop.window_seconds": 5})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=t),
            tool_calls=[{"name": "search", "arguments": {"q": "kittens"}}],
        )
        for t in [0, 4, 5]
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["window_seconds"] == 5


def test_cosine_threshold_override_blocks_fire(make_call, now):
    """Raise threshold to 0.99 — 0.5-similarity calls should not fire."""
    rule = ToolLoopRule({"tool_loop.cosine_threshold": 0.99})
    arg_sets = [
        {"q": "alpha bravo"},
        {"q": "alpha charlie"},
        {"q": "alpha delta"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None


def test_at_min_calls_exactly(make_call, now):
    """Exactly min_calls (3) at 1.0 similarity must fire (>= boundary)."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is not None


def test_below_min_calls_exact_boundary(make_call, now):
    """min_calls is strict: 2 calls when min_calls=3 should not fire."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
        for i in range(2)
    ]
    assert _evaluate(rule, session) is None


def test_at_cosine_threshold_exactly(make_call, now):
    """Similarity exactly equal to threshold should fire (>= boundary)."""
    # Construct args where Jaccard is exactly 1.0 for simplicity.
    rule = ToolLoopRule({"tool_loop.cosine_threshold": 1.0})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None


def test_confidence_capped_at_0_99(make_call, now):
    """Even with similarity 1.0 confidence should never exceed 0.99."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.confidence <= 0.99


# ---------------------------------------------------------------------------
# False-positive hazards (per docs/04_leak_taxonomy.md)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 mitigation: detect monotonic increment in numeric arg fields",
    strict=False,
)
def test_paged_calls_with_monotonic_page_should_not_fire(make_call, now):
    """Legitimate pagination — arguments differ only in monotonic page index.

    V0 has no suppression, so this currently fires. V1 should suppress.
    """
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[
                {
                    "name": "list_items",
                    "arguments": {"query": "products", "page": i + 1},
                }
            ],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is None


@pytest.mark.xfail(
    reason="V1 mitigation: per-customer polling_tools allow-list",
    strict=False,
)
def test_polling_check_status_should_not_fire(make_call, now):
    """check_status() called repeatedly until ready — allow-list candidate."""
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "check_status", "arguments": {"job_id": "abc-123"}}],
        )
        for i in range(5)
    ]
    assert _evaluate(rule, session) is None


def test_multi_armed_exploration_should_not_fire(make_call, now):
    """Agent intentionally tries variations of similar queries.

    Originally marked xfail pending LLM-as-judge ratification, but the
    v0.3.0 TF-IDF char-n-gram metric handles this case naturally — three
    queries that share a prefix but diverge in the suffix tokens land
    below the 0.70 cosine threshold, so the rule correctly does NOT fire.
    """
    rule = ToolLoopRule({})
    arg_sets = [
        {"q": "best programming language for ml"},
        {"q": "best programming language for ml beginners"},
        {"q": "best programming language for ml production"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None
