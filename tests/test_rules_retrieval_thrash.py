"""Tests for ``RetrievalThrashRule``.

Defaults: min_calls=3, window_seconds=120, cosine_threshold=0.65 (TF-IDF
char-n-gram). Scope: only tool calls whose name matches one of the
retrieval patterns (substring or glob). Calibrated looser than
``tool_loop`` because retrieval queries naturally overlap.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from token_sentinel.rules.retrieval_thrash import RetrievalThrashRule


def _evaluate(rule, calls, project="proj"):
    return rule.evaluate(calls, project=project)


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    rule = RetrievalThrashRule({})
    assert _evaluate(rule, []) is None


def test_no_tool_calls_no_fire(make_call, now):
    rule = RetrievalThrashRule({})
    session = [
        make_call(timestamp=now + timedelta(seconds=i), user_facing_output=True) for i in range(5)
    ]
    assert _evaluate(rule, session) is None


def test_non_retrieval_tool_ignored(make_call, now):
    """Three identical calls to a tool whose name doesn't match any retrieval pattern."""
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "send_email", "arguments": {"to": "x@y"}}],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is None


def test_two_identical_retrieval_calls_below_min(make_call, now):
    """min_calls defaults to 3 — two identical retrievals must not fire."""
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": {"q": "kittens"}}],
        )
        for i in range(2)
    ]
    assert _evaluate(rule, session) is None


def test_three_dissimilar_retrieval_calls_no_fire(make_call, now):
    """Three retrieval calls but with disjoint argument tokens."""
    rule = RetrievalThrashRule({})
    arg_sets = [
        {"q": "alpha bravo charlie"},
        {"q": "delta echo foxtrot"},
        {"q": "golf hotel india"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None


def test_retrieval_calls_outside_window(make_call, now):
    """Spread far apart so only the latest one is within the 120s window."""
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 200),
            tool_calls=[{"name": "vector_search", "arguments": {"q": "kittens"}}],
        )
        for i in range(4)
    ]
    assert _evaluate(rule, session) is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_three_identical_retrieval_calls_fire(make_call, now):
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "vector_search", "arguments": {"q": "kubernetes pod stuck"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.type == "retrieval_thrash"
    assert ev.rule == "v0.retrieval_thrash"
    assert ev.evidence["tool"] == "vector_search"
    assert ev.evidence["call_count"] == 3
    assert ev.evidence["window_seconds"] == 120
    assert ev.evidence["mean_similarity"] == 1.0
    assert ev.suggested_action == "cache_retrieval_results_or_widen_initial_query_or_dedupe"
    assert ev.evidence["matched_pattern"] != ""


def test_paraphrased_retrieval_queries_fire(make_call, now):
    """Three paraphrases sharing most tokens — TF-IDF char-n-gram cosine ≥ 0.65.

    Calibrated against the V0 TF-IDF char-n-gram metric: queries that share
    a 4-word stem with one trailing year/keyword variation land in the
    0.75–0.80 range, comfortably above 0.65.
    """
    rule = RetrievalThrashRule({})
    arg_sets = [
        {"query": "quarterly revenue report 2024"},
        {"query": "quarterly revenue report 2025"},
        {"query": "quarterly revenue report 2023"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "similarity_search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["tool"] == "similarity_search"
    # Confidence formula: 0.55 + (sim - 0.65) * 4, capped at 0.95.
    assert 0.5 < ev.confidence <= 0.95


def test_substring_match_in_tool_name(make_call, now):
    """Tool named ``do_search_v2`` matches the ``search`` substring pattern."""
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "do_search_v2", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["tool"] == "do_search_v2"


def test_glob_pattern_rag_prefix(make_call, now):
    """``rag_*`` glob matches a tool name with ``rag_`` prefix.

    We pick ``rag_fetch`` (no overlap with any substring pattern) so the
    matched pattern is unambiguously the glob.
    """
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "rag_fetch", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["matched_pattern"] == "rag_*"


def test_glob_pattern_retrieve_suffix(make_call, now):
    """``*_retrieve`` glob — but a tool named ``X_retrieve`` is also matched
    by the ``retrieve`` substring earlier in the list, which wins.

    The behavior we care about: the rule fires regardless of which pattern
    matched first.
    """
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "doc_retrieve", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    # ``retrieve`` substring matches before ``*_retrieve`` glob in the
    # default ordering — both are valid matches, the rule fires either way.
    assert ev.evidence["matched_pattern"] in ("retrieve", "*_retrieve")


def test_default_threshold_looser_than_tool_loop(make_call, now):
    """Args calibrated to land in (0.65, 0.70) — the gap between
    retrieval_thrash's threshold and tool_loop's default. Should fire under
    retrieval_thrash but NOT under a tool_loop-style 0.70 threshold."""
    rule = RetrievalThrashRule({})
    # TF-IDF char-n-gram of these three lands around 0.674 — above 0.65,
    # below 0.70.
    arg_sets = [
        {"q": "kubernetes pod stuck pending"},
        {"q": "kubernetes pod stuck pending state"},
        {"q": "kubernetes pod pending stuck"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 2),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert 0.65 <= ev.evidence["mean_similarity"] < 0.70

    # And confirm the same corpus does NOT fire under a 0.70 threshold.
    rule_strict = RetrievalThrashRule({"retrieval_thrash.cosine_threshold": 0.70})
    assert _evaluate(rule_strict, session) is None


def test_at_min_calls_exactly(make_call, now):
    """Exactly 3 retrieval calls at similarity 1.0 must fire."""
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is not None


# ---------------------------------------------------------------------------
# Configuration overrides
# ---------------------------------------------------------------------------


def test_min_calls_override(make_call, now):
    rule = RetrievalThrashRule({"retrieval_thrash.min_calls": 2})
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


def test_cosine_threshold_override_blocks_fire(make_call, now):
    """Set threshold to 0.99 — 3 paraphrases will not fire."""
    rule = RetrievalThrashRule({"retrieval_thrash.cosine_threshold": 0.99})
    arg_sets = [
        {"q": "alpha bravo charlie"},
        {"q": "alpha bravo delta"},
        {"q": "alpha bravo echo"},
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None


def test_window_override_inclusive(make_call, now):
    """Custom 5s window. Three calls 0/4/5s apart should still fire."""
    rule = RetrievalThrashRule({"retrieval_thrash.window_seconds": 5})
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


def test_custom_retrieval_patterns(make_call, now):
    """Custom pattern list — only tools matching it should be evaluated."""
    rule = RetrievalThrashRule({"retrieval_thrash.retrieval_tool_patterns": ["my_custom_fetch"]})
    # The default-named ``vector_search`` should NOT fire under custom patterns.
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    assert _evaluate(rule, session) is None

    # But the custom-named tool should fire.
    session2 = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "my_custom_fetch", "arguments": {"q": "kittens"}}],
        )
        for i in range(3)
    ]
    ev = _evaluate(rule, session2)
    assert ev is not None
    assert ev.evidence["tool"] == "my_custom_fetch"


# ---------------------------------------------------------------------------
# False-positive hazards (per docs/04_leak_taxonomy.md §8)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1: legitimate query-refinement loops where the agent intentionally widens/narrows shouldn't fire",
    strict=False,
)
def test_intentional_query_refinement_should_not_fire(make_call, now):
    """Agent legitimately widens then narrows — currently fires (queries overlap)."""
    rule = RetrievalThrashRule({})
    arg_sets = [
        {"q": "best italian restaurants nyc"},  # initial wide
        {"q": "best italian restaurants nyc cheap"},  # narrowed
        {"q": "best italian restaurants nyc cheap downtown"},  # narrowed more
    ]
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i * 5),
            tool_calls=[{"name": "search", "arguments": a}],
        )
        for i, a in enumerate(arg_sets)
    ]
    assert _evaluate(rule, session) is None
