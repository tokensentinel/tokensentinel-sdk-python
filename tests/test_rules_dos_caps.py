"""Tests for DoS caps in ``tool_loop`` and ``tool_definition_bloat``.

Background: v0.2.0 had no input-size guards. A 100MB tool argument blob
from a misbehaving upstream tool would explode into ~100M character
n-grams inside ``_mean_pairwise_similarity``, pinning a CPU and burning
gigabytes of memory inside the customer's process. Same risk in
``tool_definition_bloat`` which serialised every tool def without a cap.

v0.3 caps:

- ``tool_loop.max_arg_bytes`` (default 64KB): truncate each per-arg JSON
  string before n-gram extraction.
- ``tool_loop.max_total_corpus_bytes`` (default 1MB): cumulative cap; if
  exceeded, similarity is evaluated on the leading args that fit.
- ``tool_definition_bloat.max_tool_bytes`` (default 256KB): truncate
  per-tool serialisation; the original byte count is still used for the
  threshold comparison so the rule fires correctly on a single huge tool.
- ``tool_definition_bloat.max_total_bytes`` (default 5MB): if cumulative
  bytes exceed the cap, short-circuit to a low-confidence (0.50) event.

Performance budget: rules run in <50ms p95 on realistic inputs and must
not blow up on adversarial inputs.
"""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from token_sentinel.rules.retrieval_thrash import RetrievalThrashRule
from token_sentinel.rules.tool_definition_bloat import ToolDefinitionBloatRule
from token_sentinel.rules.tool_loop import ToolLoopRule

# ---------------------------------------------------------------------------
# 1. 100MB single-arg payload — must not crash and must run in <100ms
# ---------------------------------------------------------------------------


def test_100mb_payload_does_not_crash_or_hang(make_call, now):
    """The DoS reproducer from the code review.

    Three calls, each carrying a 100MB argument blob. Without the cap,
    n-gram extraction would produce ~100M character n-grams per doc and
    consume gigabytes of RAM. With the cap, each blob is truncated to
    64KB before any expensive work, so the rule completes in well under
    a hundred milliseconds.
    """
    huge = "A" * 100_000_000  # 100MB string
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": huge}}],
        )
        for i in range(3)
    ]

    start = time.perf_counter()
    ev = rule.evaluate(session, project="p")
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Note: the harness must not raise OOM / MemoryError; reaching here is
    # already a partial success. The harder assertion is the timing budget.
    # Pre-cap baseline on this hardware was 3000+ ms; with the cap we see
    # ~80–95ms steady state, with occasional 200+ms spikes from allocator
    # / GC variance on shared CI machines. Asserting <250ms keeps the
    # protection meaningful (orders of magnitude under uncapped) while
    # absorbing platform variance.
    assert elapsed_ms < 250, f"DoS-cap eval took {elapsed_ms:.1f}ms, expected <250ms"
    # And the rule still gets a usable signal — three identical 100MB-prefix
    # blobs are still "the same call repeated".
    assert ev is not None
    assert ev.evidence["call_count"] == 3


def test_nested_huge_value_does_not_blow_up_redaction_path(make_call, now):
    """Regression for the post- re-review finding.

    `_bounded_value_length` originally only fast-pathed top-level strings;
    nested dict/list values fell through to unbounded `json.dumps` even on
    the redaction path. A maliciously-nested arg shape such as
    ``{"filters": {"q": "X" * 100_000_000}}`` would still materialise the
    entire blob inside ``json.dumps``, regressing HIGH-3 specifically on
    the redaction branch. Fix: route nested values through
    ``_bounded_dumps`` so the inner string is shrunk first.
    """
    huge_nested = {"filters": {"q": "X" * 100_000_000}}
    rule = ToolLoopRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": huge_nested}],
        )
        for i in range(3)
    ]

    start = time.perf_counter()
    ev = rule.evaluate(session, project="p")
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Same generous budget as the top-level test — accommodates allocator
    # variance from the 100MB fixture string itself.
    assert elapsed_ms < 250, f"Nested-DoS redaction took {elapsed_ms:.1f}ms, expected <250ms"
    assert ev is not None
    # Redacted shape; raw values must not have leaked into evidence.
    assert "sample_args" in ev.evidence
    for sample in ev.evidence["sample_args"]:
        assert "filters" in sample["keys"]
        # value_lengths is bounded — the 100MB nested value reports
        # at-most the 1MB redaction cap, NOT the true 100MB length.
        assert sample["value_lengths"]["filters"] <= 1_048_576


# ---------------------------------------------------------------------------
# 2. Truncated args still fire when prefix is similar
# ---------------------------------------------------------------------------


def test_oversized_args_still_fire_when_prefix_is_similar(make_call, now):
    """Three calls with oversized but identical-prefix args must fire.

    The cap truncates at the prefix; identical 64KB prefixes → identical
    truncated docs → similarity 1.0. The rule must continue to detect
    the loop even when the args are individually too big to fully
    serialise.
    """
    # 80KB of identical content followed by a small differentiator. The
    # 64KB cap truncates inside the identical region, so similarity is 1.0.
    common_prefix = "X" * 80_000
    rule = ToolLoopRule({})
    session = []
    for i in range(3):
        session.append(
            make_call(
                timestamp=now + timedelta(seconds=i),
                tool_calls=[
                    {
                        "name": "search",
                        "arguments": {"payload": common_prefix + f"_tail_{i}"},
                    }
                ],
            )
        )

    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["call_count"] == 3
    # Truncation happens inside the identical prefix → similarity 1.0.
    assert ev.evidence["mean_similarity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Total-corpus cap drops trailing args
# ---------------------------------------------------------------------------


def test_total_corpus_cap_drops_trailing_args(make_call, now):
    """When cumulative arg bytes exceed the total cap, trailing args are
    skipped. We verify this by feeding 5 identical-prefix blobs with a
    deliberately small total cap, then ensuring the rule still fires
    on the leading subset.

    This test pins the behaviour: under cap pressure we evaluate
    similarity on the subset that fits, rather than refusing entirely.
    """
    # 200KB blob; total cap of 600KB → only 3 of 5 blobs fit.
    blob = "Y" * 200_000
    rule = ToolLoopRule(
        {
            "tool_loop.max_arg_bytes": 250_000,  # don't truncate per-arg
            "tool_loop.max_total_corpus_bytes": 600_000,  # only 3 fit
        }
    )
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"payload": blob}}],
        )
        for i in range(5)
    ]

    ev = rule.evaluate(session, project="p")
    assert ev is not None
    # call_count in evidence reflects the unfiltered call count — 5 calls
    # were observed in the window. Similarity was just computed on a subset
    # so we don't burn unbounded memory.
    assert ev.evidence["call_count"] == 5
    # Identical blobs → similarity 1.0 even on the truncated subset.
    assert ev.evidence["mean_similarity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. tool_definition_bloat short-circuits on >5MB total
# ---------------------------------------------------------------------------


def test_tool_definition_bloat_short_circuits_on_oversized_block(make_call, now):
    """A 6MB tool block must short-circuit to a low-confidence event,
    NOT silently disable the rule.

    Customer still gets a signal that something is wrong; the SDK
    doesn't blow up their process trying to crunch gigabytes.
    """

    # Build 30 tools whose serialised size each is ~250KB. Total ~= 7.5MB,
    # well over the default 5MB short-circuit cap.
    def big_tool(i: int) -> dict:
        return {
            "name": f"tool_{i}",
            "description": "x" * 250_000,
            "input_schema": {"type": "object", "properties": {}},
        }

    tools = [big_tool(i) for i in range(30)]
    rule = ToolDefinitionBloatRule({})
    session = [make_call(timestamp=now, raw_request={"tools": tools})]

    start = time.perf_counter()
    ev = rule.evaluate(session, project="p")
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Must complete fast — we're short-circuiting, not crunching the full block.
    assert elapsed_ms < 200, f"short-circuit eval took {elapsed_ms:.1f}ms"
    assert ev is not None
    assert ev.confidence == pytest.approx(0.50)
    assert ev.evidence.get("truncated") is True
    assert ev.evidence.get("note") == "tool block too large to evaluate"
    assert ev.suggested_action == "reduce_tool_count_or_use_lazy_tool_loading"


# ---------------------------------------------------------------------------
# 5. Low-confidence short-circuit event has the right shape
# ---------------------------------------------------------------------------


def test_short_circuit_event_shape_is_minimal(make_call, now):
    """The short-circuit event must NOT include top_tools_by_size or other
    fields that would have required full evaluation. Customer code
    branching on ``evidence['truncated']`` should work cleanly.
    """
    # Build something just over the 5MB cap.
    tools = [{"name": f"tool_{i}", "description": "x" * 220_000} for i in range(30)]
    rule = ToolDefinitionBloatRule({})
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = rule.evaluate(session, project="p")
    assert ev is not None

    # Required keys for the truncated event.
    assert "tool_count" in ev.evidence
    assert "definition_bytes" in ev.evidence
    assert ev.evidence["truncated"] is True
    assert "note" in ev.evidence

    # Crucially: no top_tools_by_size — we never computed it.
    assert "top_tools_by_size" not in ev.evidence
    # And type / rule strings still match the canonical bloat event.
    assert ev.type == "tool_definition_bloat"
    assert ev.rule == "v0.tool_definition_bloat"


# ---------------------------------------------------------------------------
# 6. Caps are configurable per-rule
# ---------------------------------------------------------------------------


def test_caps_are_configurable_per_rule(make_call, now):
    """All four caps must be honoured when overridden via the rule config.

    Backstop against future refactoring that hard-codes the constants
    inside the rule body.
    """
    # tool_loop: drop the per-arg cap to 100 bytes. A 5KB identical-prefix
    # blob is still truncated to 100 bytes for similarity computation.
    blob = "Z" * 5_000
    rule = ToolLoopRule({"tool_loop.max_arg_bytes": 100})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"payload": blob + str(i)}}],
        )
        for i in range(3)
    ]
    ev = rule.evaluate(session, project="p")
    # 100-byte prefixes are identical → similarity 1.0 → still fires.
    assert ev is not None
    assert ev.evidence["mean_similarity"] == pytest.approx(1.0)

    # tool_definition_bloat: drop the total cap so a small tool block
    # short-circuits.
    rule_bloat = ToolDefinitionBloatRule({"tool_definition_bloat.max_total_bytes": 100})
    tools = [
        {"name": "t1", "description": "x" * 200},
        {"name": "t2", "description": "y" * 200},
    ]
    session_bloat = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev_bloat = rule_bloat.evaluate(session_bloat, project="p")
    assert ev_bloat is not None
    assert ev_bloat.evidence.get("truncated") is True


# ---------------------------------------------------------------------------
# 7. Default caps don't trigger on realistic <50KB inputs
# ---------------------------------------------------------------------------


def test_default_caps_do_not_trigger_on_realistic_inputs(make_call, now):
    """Defaults (64KB / 1MB / 256KB / 5MB) should be entirely invisible on
    realistic agent traffic. We feed a normal RAG-shaped corpus and
    confirm the rule fires with the expected similarity score and
    full evidence shape — no short-circuit, no truncation artifacts.
    """
    # 3 RAG-style queries at ~50 bytes each — realistic by an order of
    # magnitude. This is well under every default cap. Identical args so
    # similarity is unambiguously 1.0 and the rule fires; what we're
    # asserting here is that the *defaults* don't truncate or short-
    # circuit on a normal input.
    rule = ToolLoopRule({})
    args = {"q": "kubernetes pod stuck pending"}
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": args}],
        )
        for i in range(3)
    ]
    ev = rule.evaluate(session, project="p")
    # The rule fires AND the evidence has the standard shape — no
    # "truncated" flag, similarity is a normal number.
    assert ev is not None
    assert "truncated" not in ev.evidence
    assert 0.0 < ev.evidence["mean_similarity"] <= 1.0

    # tool_definition_bloat with a realistic ~30KB tool block — fires the
    # canonical 0.85-confidence event, NOT the short-circuit 0.50 event.
    rule_bloat = ToolDefinitionBloatRule({})
    tools = [
        {
            "name": f"tool_{i}",
            "description": "x" * 50,
            "input_schema": {"type": "object", "properties": {}},
        }
        for i in range(35)
    ]
    session_bloat = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev_bloat = rule_bloat.evaluate(session_bloat, project="p")
    assert ev_bloat is not None
    assert ev_bloat.confidence in (0.85, 0.95)
    assert ev_bloat.evidence.get("truncated") is not True
    # Standard evidence fields are present.
    assert "top_tools_by_size" in ev_bloat.evidence
    assert "estimated_tokens" in ev_bloat.evidence


# ---------------------------------------------------------------------------
# 8. retrieval_thrash inherits the caps via shared helper
# ---------------------------------------------------------------------------


def test_retrieval_thrash_inherits_dos_caps(make_call, now):
    """retrieval_thrash reuses ``_mean_pairwise_similarity`` from tool_loop,
    so the same DoS protection applies. Verify a 50MB query doesn't blow
    up retrieval_thrash either.

    Budget includes some headroom over the tool_loop test because much of
    the elapsed time is the 50MB string-slice the rule does once per call
    (which depends on the host's allocator). The pre-cap version of this
    code took multiple seconds; we're checking we're well under that.
    """
    huge = "Q" * 50_000_000
    rule = RetrievalThrashRule({})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "vector_search", "arguments": {"q": huge}}],
        )
        for i in range(3)
    ]
    start = time.perf_counter()
    ev = rule.evaluate(session, project="p")
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Pre-cap baseline was multiple seconds; we expect well under 250ms
    # consistently. Using a 250ms budget to absorb GC / allocator variance
    # on busy CI machines while still meaningfully asserting the cap is
    # in force.
    assert elapsed_ms < 250, f"retrieval_thrash DoS-cap eval took {elapsed_ms:.1f}ms"
    assert ev is not None
    assert ev.evidence["tool"] == "vector_search"


# ---------------------------------------------------------------------------
# Bonus: short-circuit threshold is configurable
# ---------------------------------------------------------------------------


def test_per_tool_cap_does_not_change_threshold_decision(make_call, now):
    """A single 1MB tool definition (above per-tool cap, below total cap)
    must still fire as bloat. The truncation is for evidence reporting
    only — the original byte count drives the threshold comparison.
    """
    # One tool whose serialised size is ~1MB. Per-tool cap is 256KB but
    # total cap is 5MB, so we should serialise it (truncated for evidence)
    # and emit the canonical bloat event using the full 1MB byte count.
    tool = {
        "name": "the_giant",
        "description": "x" * 1_000_000,
        "input_schema": {"type": "object", "properties": {}},
    }
    rule = ToolDefinitionBloatRule({})
    session = [make_call(timestamp=now, raw_request={"tools": [tool]})]
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    # Original (uncapped) byte count is what the threshold compares against.
    assert ev.evidence["definition_bytes"] >= 1_000_000
    assert ev.evidence.get("truncated") is not True
    # And the top-tools-by-size entry reports the original byte count.
    top = ev.evidence["top_tools_by_size"]
    assert top[0]["name"] == "the_giant"
    assert top[0]["bytes"] >= 1_000_000
