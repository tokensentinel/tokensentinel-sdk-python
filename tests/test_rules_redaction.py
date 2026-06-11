"""Tests for ``sample_args`` redaction in ``tool_loop`` and
``retrieval_thrash``.

Background: v0.2.0 shipped raw tool arguments verbatim into
``LeakEvent.evidence['sample_args']``. That argument blob can carry user
PII (chat queries, account IDs, secrets pasted into a prompt) — and the
customer's leak handler may forward events to Slack/Datadog/Sentry/our
cloud. v0.3 redacts by default and gates raw args behind an explicit
opt-in (``<rule>.include_raw_args: True``).

These tests pin the redacted shape, prove the opt-in path actually opts
in, and confirm that redaction does not change the firing behaviour of
either rule (the redaction touches evidence only, not the similarity
computation).
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from token_sentinel.rules.retrieval_thrash import RetrievalThrashRule
from token_sentinel.rules.tool_loop import ToolLoopRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _three_search_calls(make_call, now, args: dict):
    """Three identical search-tool calls in a tight window.

    Identical args → similarity 1.0 → the rule fires reliably regardless of
    threshold tuning, which is what we want when we're testing evidence
    shape rather than the similarity math.
    """
    return [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": args}],
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# 1. Default: redacted output shape
# ---------------------------------------------------------------------------


def test_default_redacts_sample_args_in_tool_loop(make_call, now):
    """By default tool_loop must NOT ship raw args; it ships a redacted summary."""
    rule = ToolLoopRule({})
    raw_args = {"query": "user-secret-question", "top_k": 5}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    sample = ev.evidence["sample_args"]
    assert isinstance(sample, list)
    assert len(sample) == 3

    # Each entry is a redacted summary dict, NOT the raw arg dict.
    for entry in sample:
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"keys", "value_lengths", "hash"}
        # Crucial privacy assertion: the raw value never appears.
        assert "user-secret-question" not in json.dumps(entry)
        # Redacted keys are exactly the original arg keys, sorted.
        assert entry["keys"] == ["query", "top_k"]
        # value_lengths reports the per-key serialised JSON length.
        assert entry["value_lengths"]["query"] == len(json.dumps("user-secret-question"))
        assert entry["value_lengths"]["top_k"] == len(json.dumps(5))
        # Hash is a 16-char hex prefix.
        assert isinstance(entry["hash"], str)
        assert len(entry["hash"]) == 16
        assert all(c in "0123456789abcdef" for c in entry["hash"])


# ---------------------------------------------------------------------------
# 2. Opt-in raw-args path
# ---------------------------------------------------------------------------


def test_opt_in_raw_args_ships_originals_in_tool_loop(make_call, now):
    """``tool_loop.include_raw_args: True`` ships the original arg dicts.

    Customers who have explicitly accepted the privacy tradeoff (e.g.,
    local-dev triage on a non-production project) get the raw values back.
    """
    rule = ToolLoopRule({"tool_loop.include_raw_args": True})
    raw_args = {"query": "user-secret-question", "top_k": 5}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    sample = ev.evidence["sample_args"]
    # Raw arg dicts pass through untouched.
    for entry in sample:
        assert entry == raw_args


def test_opt_in_raw_args_in_retrieval_thrash(make_call, now):
    """``retrieval_thrash.include_raw_args: True`` mirrors the tool_loop opt-in."""
    rule = RetrievalThrashRule({"retrieval_thrash.include_raw_args": True})
    raw_args = {"q": "kubernetes pod stuck pending"}
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": raw_args}],
        )
        for i in range(3)
    ]

    ev = rule.evaluate(session, project="proj")
    assert ev is not None
    for entry in ev.evidence["sample_args"]:
        assert entry == raw_args


# ---------------------------------------------------------------------------
# 3. JSON-serialisability — the redacted form must round-trip cleanly
# ---------------------------------------------------------------------------


def test_redacted_evidence_is_json_serialisable(make_call, now):
    """Leak handlers commonly ``json.dumps(event.evidence)`` for transport.

    The redacted form must survive round-tripping without TypeError.
    """
    rule = ToolLoopRule({})
    # Mix of types in the value to make sure we didn't introduce non-JSON-
    # safe primitives (e.g., bytes, datetime) in the redacted summary.
    raw_args = {"query": "x", "ids": [1, 2, 3], "meta": {"k": True}}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None
    # Should not raise.
    serialised = json.dumps(ev.evidence)
    assert "sample_args" in serialised
    # And round-trip cleanly.
    round_tripped = json.loads(serialised)
    assert round_tripped["sample_args"][0]["keys"] == ["ids", "meta", "query"]


# ---------------------------------------------------------------------------
# 4. Hash stability — identical args produce identical hashes
# ---------------------------------------------------------------------------


def test_redaction_hash_is_stable_across_identical_args(make_call, now):
    """Three identical calls → three identical hashes.

    This is the consumer's "tell whether the same call repeated" signal —
    the hash MUST be deterministic over equal inputs.
    """
    rule = ToolLoopRule({})
    raw_args = {"q": "deterministic"}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    hashes = [entry["hash"] for entry in ev.evidence["sample_args"]]
    assert hashes[0] == hashes[1] == hashes[2]
    # And cross-check the value: it should match a manually-computed
    # SHA-256 prefix of the canonical sorted JSON.
    expected_full = json.dumps(raw_args, sort_keys=True, default=str)
    expected_prefix = hashlib.sha256(expected_full.encode()).hexdigest()[:16]
    assert hashes[0] == expected_prefix


# ---------------------------------------------------------------------------
# 5. Sorted-keys determinism — handlers can rely on deterministic key lists
# ---------------------------------------------------------------------------


def test_redacted_keys_are_sorted_deterministically(make_call, now):
    """Whatever the input dict key order, the redacted ``keys`` is sorted."""
    rule = ToolLoopRule({})
    # Build args with a deliberately scrambled key order.
    raw_args = {"zeta": 1, "alpha": 2, "mike": 3, "bravo": 4}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    for entry in ev.evidence["sample_args"]:
        assert entry["keys"] == ["alpha", "bravo", "mike", "zeta"]


# ---------------------------------------------------------------------------
# 6. Nested values are handled in value_lengths
# ---------------------------------------------------------------------------


def test_value_lengths_handles_nested_values(make_call, now):
    """Nested dicts/lists must produce a sensible byte length, not crash."""
    rule = ToolLoopRule({})
    nested = {
        "filters": {"author": "alice", "tags": ["physics", "ml"]},
        "page": 1,
    }
    session = _three_search_calls(make_call, now, nested)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    entry = ev.evidence["sample_args"][0]
    assert entry["keys"] == ["filters", "page"]
    # The nested dict's serialised length should match a real json.dumps.
    expected_filters_len = len(json.dumps(nested["filters"]))
    assert entry["value_lengths"]["filters"] == expected_filters_len
    assert entry["value_lengths"]["page"] == len(json.dumps(1))
    # And nothing about the nested *values* leaks into the summary.
    assert "alice" not in json.dumps(entry)
    assert "physics" not in json.dumps(entry)


# ---------------------------------------------------------------------------
# 7. Both rules redact identically (single source of truth)
# ---------------------------------------------------------------------------


def test_tool_loop_and_retrieval_thrash_redact_identically(make_call, now):
    """Same args → same redacted form regardless of which rule fires.

    Both rules call into ``_build_sample_args``; this test pins that
    contract so a future refactor can't accidentally diverge them.
    """
    raw_args = {"q": "shared-query", "top_k": 3}

    # tool_loop with default config
    tl_session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": raw_args}],
        )
        for i in range(3)
    ]
    tl_ev = ToolLoopRule({}).evaluate(tl_session, project="p")

    # retrieval_thrash with default config (search is a retrieval pattern)
    rt_session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": raw_args}],
        )
        for i in range(3)
    ]
    rt_ev = RetrievalThrashRule({}).evaluate(rt_session, project="p")

    assert tl_ev is not None
    assert rt_ev is not None

    tl_sample = tl_ev.evidence["sample_args"][0]
    rt_sample = rt_ev.evidence["sample_args"][0]

    # Same raw args → same keys, value_lengths, hash. The redaction
    # contract is identical between the two rules.
    assert tl_sample == rt_sample


# ---------------------------------------------------------------------------
# 8. Redaction does not change firing behaviour
# ---------------------------------------------------------------------------


def test_redaction_does_not_change_firing_behaviour(make_call, now):
    """The redaction lives in evidence-building only; it must NOT touch the
    similarity computation. Same input → same fire/no-fire decision and
    same mean_similarity, regardless of include_raw_args."""
    raw_args = {"query": "shared text shared text shared text"}

    # Default (redacted)
    rule_default = ToolLoopRule({})
    session = _three_search_calls(make_call, now, raw_args)
    ev_default = rule_default.evaluate(session, project="p")

    # Opt-in (raw)
    rule_raw = ToolLoopRule({"tool_loop.include_raw_args": True})
    ev_raw = rule_raw.evaluate(session, project="p")

    assert ev_default is not None and ev_raw is not None
    assert ev_default.confidence == ev_raw.confidence
    assert ev_default.evidence["mean_similarity"] == ev_raw.evidence["mean_similarity"]
    assert ev_default.evidence["call_count"] == ev_raw.evidence["call_count"]
    assert ev_default.evidence["tool"] == ev_raw.evidence["tool"]


# ---------------------------------------------------------------------------
# Bonus: explicit "no-PII-leakage" assertion across realistic value shapes
# ---------------------------------------------------------------------------


def test_no_raw_value_substring_appears_in_redacted_evidence(make_call, now):
    """Stronger assertion than dict-shape: even string values must not leak.

    We feed a recognisable secret-like substring and assert it's nowhere in
    the serialised evidence blob. This is the test a security-conscious
    customer will write to confirm we aren't leaking their data.
    """
    rule = ToolLoopRule({})
    secret = "sk-deadbeef-NEVER-LEAK-ME-12345"
    raw_args = {"api_key": secret, "endpoint": "https://example.com"}
    session = _three_search_calls(make_call, now, raw_args)

    ev = rule.evaluate(session, project="proj")
    assert ev is not None

    blob = json.dumps(ev.evidence)
    assert secret not in blob
    assert "https://example.com" not in blob
