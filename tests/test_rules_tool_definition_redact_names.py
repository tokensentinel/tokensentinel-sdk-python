"""Tests for MED-6: ``tool_definition_bloat`` optional tool-name redaction.

The v0.2.0 implementation always shipped literal tool names in
``evidence["top_tools_by_size"]``. For most customers (``"web_search"`` etc.)
this is fine, but customers whose tool names map to internal API surface
(``"create_payroll_run"``, ``"get_employee_ssn"``) leak that surface to
whatever destination their leak handler ships to (Slack, Datadog, etc.).

v0.3.2 adds ``tool_definition_bloat.redact_names`` (default ``False`` — current
behaviour preserved). When ``True``, evidence ships only a 16-hex-char SHA-256
prefix of each tool name plus the byte count.
"""

from __future__ import annotations

import hashlib
import re

from token_sentinel.rules.tool_definition_bloat import ToolDefinitionBloatRule


def _evaluate(rule, calls, project="proj"):
    return rule.evaluate(calls, project=project)


def _make_tool(name: str, *, schema_size: int = 200) -> dict:
    """Build a tool def with predictable serialised size."""
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
# Default (no config): emits names — preserves  shape
# ---------------------------------------------------------------------------


def test_default_emits_literal_tool_names(make_call, now):
    """With no ``redact_names`` config, evidence ships ``{"name": str, "bytes": int}``."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(35)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    assert len(top) == 5
    for entry in top:
        assert "name" in entry
        assert "bytes" in entry
        assert "hash" not in entry
        assert isinstance(entry["name"], str)
        # Must be the literal tool name we passed in.
        assert entry["name"].startswith("tool_")


def test_explicit_false_emits_literal_tool_names(make_call, now):
    """Explicit ``redact_names: false`` is identical to default."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": False})
    tools = [_make_tool(f"sensitive_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    for entry in top:
        assert "name" in entry
        assert "hash" not in entry
        assert entry["name"].startswith("sensitive_")


# ---------------------------------------------------------------------------
# redact_names=True: emits hashes
# ---------------------------------------------------------------------------


def test_redact_names_true_emits_hashes_not_names(make_call, now):
    """With ``redact_names: true``, evidence ships ``{"hash": str, "bytes": int}``."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    tools = [_make_tool(f"create_payroll_run_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    assert len(top) == 5
    for entry in top:
        assert "hash" in entry
        assert "bytes" in entry
        assert "name" not in entry, f"redacted shape must NOT ship 'name' — got {entry!r}"
        # Sensitive substring of the literal tool name must NOT appear in the
        # serialised evidence — that's the entire point of the redaction.
        assert "create_payroll_run" not in entry["hash"]


# ---------------------------------------------------------------------------
# Hash is 16 hex chars
# ---------------------------------------------------------------------------


def test_hash_is_16_hex_chars(make_call, now):
    """The redacted hash must be exactly 16 lowercase hex characters."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    hex_re = re.compile(r"^[0-9a-f]{16}$")
    for entry in ev.evidence["top_tools_by_size"]:
        h = entry["hash"]
        assert hex_re.match(h), f"hash {h!r} is not 16 lowercase hex chars"


# ---------------------------------------------------------------------------
# Same name → same hash (stability)
# ---------------------------------------------------------------------------


def test_same_name_produces_same_hash(make_call, now):
    """The hash must be deterministic so handlers can dedupe by it."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    # 30 tools, all with the same name. _tool_name reads ``tool["name"]``,
    # so they will collapse into the same hash. (Distinct dict identities,
    # same name string.)
    tools = [_make_tool("get_employee_ssn", schema_size=20) for _ in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    assert len(top) == 5
    hashes = {entry["hash"] for entry in top}
    assert len(hashes) == 1, f"expected one unique hash, got {hashes}"
    # And the hash matches the explicit SHA-256 prefix of the name.
    expected = hashlib.sha256(b"get_employee_ssn").hexdigest()[:16]
    assert hashes == {expected}


def test_hash_matches_known_sha256_prefix(make_call, now):
    """Spot-check: hash for a known name == hashlib.sha256(name).hexdigest()[:16]."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    # Build 30 tools where one is the largest so it lands at top of the list.
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(29)]
    tools.append(_make_tool("create_payroll_run", schema_size=2000))
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    # Hog tool sorts to the top by size.
    expected = hashlib.sha256(b"create_payroll_run").hexdigest()[:16]
    assert top[0]["hash"] == expected


# ---------------------------------------------------------------------------
# Bytes still emitted in BOTH shapes
# ---------------------------------------------------------------------------


def test_bytes_emitted_in_default_shape(make_call, now):
    """Default shape: ``bytes`` is present and an int."""
    rule = ToolDefinitionBloatRule({})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    for entry in ev.evidence["top_tools_by_size"]:
        assert "bytes" in entry
        assert isinstance(entry["bytes"], int)
        assert entry["bytes"] > 0


def test_bytes_emitted_in_redacted_shape(make_call, now):
    """Redacted shape: ``bytes`` is present and an int (unchanged)."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(30)]
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    for entry in ev.evidence["top_tools_by_size"]:
        assert "bytes" in entry
        assert isinstance(entry["bytes"], int)
        assert entry["bytes"] > 0


def test_bytes_ranking_preserved_under_redaction(make_call, now):
    """Tools should still be ordered by size when redacted (descending)."""
    rule = ToolDefinitionBloatRule({"tool_definition_bloat.redact_names": True})
    # 29 small, 1 very large — the large one must land at top[0].
    tools = [_make_tool(f"tool_{i}", schema_size=20) for i in range(29)]
    tools.append(_make_tool("hog", schema_size=2000))
    session = [make_call(timestamp=now, raw_request={"tools": tools})]
    ev = _evaluate(rule, session)
    assert ev is not None
    top = ev.evidence["top_tools_by_size"]
    # Descending order by bytes.
    for a, b in zip(top, top[1:], strict=False):
        assert a["bytes"] >= b["bytes"]
    # The hog must be top[0] — verify via its hash, not its name.
    expected_hog = hashlib.sha256(b"hog").hexdigest()[:16]
    assert top[0]["hash"] == expected_hog
