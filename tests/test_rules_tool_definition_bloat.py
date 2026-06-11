"""Tests for ``ToolDefinitionBloatRule``.

Defaults: tool_count_threshold=30, tool_definition_bytes_threshold=30000.
Confidence: 0.85 if either threshold tripped, 0.95 if both tripped by ≥50%.
"""

from __future__ import annotations

import json

import pytest

from token_sentinel.rules.tool_definition_bloat import ToolDefinitionBloatRule


def _evaluate(rule, calls, project="proj"):
    return rule.evaluate(calls, project=project)


def _make_tool(name: str, *, schema_size: int = 200) -> dict:
    """Build a tool def whose serialized size we can predict.

    Uses a string of ``schema_size`` bytes inside ``input_schema.description``
    so we can hit byte thresholds deterministically without having to count
    JSON brackets.
    """
    return {
        "name": name,
        "description": "x",
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "x" * schema_size},
            },
        },
    }


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    rule = ToolDefinitionBloatRule({})
    assert _evaluate(rule, []) is None


def test_no_tools_in_request(make_call, now):
    """No ``tools`` key → no signal."""
    rule = ToolDefinitionBloatRule({})
    session = [make_call(timestamp=now, raw_request={"messages": []})]
    assert _evaluate(rule, session) is None


def test_empty_tools_list(make_call, now):
    rule = ToolDefinitionBloatRule({})
    session = [make_call(timestamp=now, raw_request={"tools": []})]
    assert _evaluate(rule, session) is None


def test_single_small_tool_no_fire(make_call, now):
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool("search", schema_size=50)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    assert _evaluate(rule, session) is None


def test_under_count_threshold_small_tools(make_call, now):
    """29 tools, each tiny → both thresholds clear."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(29)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    assert _evaluate(rule, session) is None


def test_just_below_byte_threshold(make_call, now):
    """A handful of tools whose total is just under 30000 bytes."""
    rule = ToolDefinitionBloatRule({})
    # 5 tools × ~5kb each = ~25kb total — under 30kb, well under 30 tools.
    tools = [_make_tool(f"tool_{i}", schema_size=4500) for i in range(5)]
    payload_bytes = len(json.dumps(tools, sort_keys=True).encode())
    assert payload_bytes < 30_000  # sanity-check the fixture
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    assert _evaluate(rule, session) is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_at_exactly_count_threshold_fires(make_call, now):
    """Exactly 30 tools (default threshold) must fire — boundary is inclusive."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.type == "tool_definition_bloat"
    assert ev.rule == "v0.tool_definition_bloat"
    assert ev.evidence["tool_count"] == 30
    assert ev.suggested_action == "reduce_tool_count_or_use_lazy_tool_loading"


def test_count_only_breach(make_call, now):
    """40 small tools — count breached, bytes not — confidence 0.85."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(40)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.confidence == pytest.approx(0.85)
    assert ev.evidence["tool_count"] == 40


def test_bytes_only_breach(make_call, now):
    """5 huge tools — bytes breached, count not — confidence 0.85."""
    rule = ToolDefinitionBloatRule({})
    # 5 tools × ~7kb each = ~35kb — over 30kb, well under 30 tools.
    tools = [_make_tool(f"tool_{i}", schema_size=6500) for i in range(5)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.confidence == pytest.approx(0.85)
    assert ev.evidence["definition_bytes"] >= 30_000


def test_both_breached_severely_confidence_0_95(make_call, now):
    """58 tools at ~55KB — the canonical Claude-Desktop-with-many-MCPs case.

    Both thresholds tripped by >>50% → confidence escalates to 0.95.
    """
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=900) for i in range(58)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.confidence == pytest.approx(0.95)
    assert ev.evidence["tool_count"] == 58
    # 58 × ~950 bytes per tool ≈ 55KB
    assert ev.evidence["definition_bytes"] > 45_000


def test_bedrock_toolconfig_shape(make_call, now):
    """Bedrock Converse uses ``toolConfig.tools`` instead of top-level ``tools``."""
    rule = ToolDefinitionBloatRule({})
    inner = [_make_tool(f"tool_{i}", schema_size=20) for i in range(35)]
    session = [make_call(timestamp=now, raw_request={"toolConfig": {"tools": inner}})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["tool_count"] == 35


def test_top_tools_by_size_returns_5(make_call, now):
    rule = ToolDefinitionBloatRule({})
    # 30 tools, one of which is 10× larger — should appear at top.
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(29)]
    tools.append(_make_tool("hog", schema_size=2000))
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    assert len(top) == 5
    # Hog should be first.
    assert top[0]["name"] == "hog"
    assert top[0]["bytes"] > top[1]["bytes"]


def test_estimated_burn_scales_with_recent_calls(make_call, now):
    """A 10-call session should yield ~10× the burn of a 1-call session."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(40)]

    single = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev_single = _evaluate(rule, single)
    assert ev_single is not None
    burn_single = ev_single.estimated_burn

    many = [make_call(timestamp=now, raw_request={"tools": tools}) for _ in range(10)]
    ev_many = _evaluate(rule, many)
    assert ev_many is not None
    burn_many = ev_many.estimated_burn

    # 10× the calls should give ~10× the burn (within float rounding).
    assert burn_many == pytest.approx(burn_single * 10, rel=0.05)


def test_threshold_overrides(make_call, now):
    """Custom thresholds should be honored."""
    rule = ToolDefinitionBloatRule(
        {
            "tool_definition_bloat.tool_count_threshold": 5,
            "tool_definition_bloat.tool_definition_bytes_threshold": 1_000,
        }
    )
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(6)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["count_threshold"] == 5
    assert ev.evidence["bytes_threshold"] == 1_000


def test_evidence_contains_estimated_tokens(make_call, now):
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(35)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["estimated_tokens"] == ev.evidence["definition_bytes"] // 4


def test_openai_function_shape_top_tools(make_call, now):
    """OpenAI tool def shape: ``{type, function: {name, parameters}}``."""
    rule = ToolDefinitionBloatRule({})
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(35)
    ]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    # Names should be extracted from nested ``function.name`` field.
    top_names = [t["name"] for t in ev.evidence["top_tools_by_size"]]
    assert all(n.startswith("tool_") for n in top_names)


# ---------------------------------------------------------------------------
# False-positive hazards (per docs/04_leak_taxonomy.md §7)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 adds context awareness: a single mass-tool-import call after a quiet session shouldn't fire",
    strict=False,
)
def test_legitimate_one_shot_mass_tool_import_should_not_fire(make_call, now):
    """A manual ``list_tools`` registry dump on a single call.

    V0 has no session-context awareness, so this currently fires. V1 should
    suppress it when it's a one-shot pattern in an otherwise quiet session.
    """
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(40)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    assert _evaluate(rule, session) is None
