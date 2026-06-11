"""Rule: a single request ships an oversized block of tool definitions.

The MCP-host failure mode this targets:

    Claude Desktop with 12 MCP servers attached → 58 tools merged into the
    top-level ``tools`` array → ~55K tokens of definitions injected on EVERY
    user turn → 72% of a 75K context budget burned before any work begins.

Tool definitions are sent on EVERY call (the model has no memory of them
across requests), so the marginal cost is ``definition_bytes × calls``. A
30-tool, 30KB block on a chatty agent making 100 requests/hour is meaningful
spend. We surface this as a per-call rule so customers can decide whether
to: prune the tool list, split into per-task agents, or move to lazy tool
loading once the host supports it.

This rule is intentionally cheap: serialize once, count, compare. Sub-millisecond
on any realistic input.

Provider keys:
- Anthropic / OpenAI / Google: ``raw_request['tools']`` — list of tool dicts.
- AWS Bedrock Converse: ``raw_request['toolConfig']['tools']`` — same shape
  nested one level deeper. We accept either.

False positives this rule may produce:
- Legitimate one-shot mass-tool-import calls (e.g., a manual ``list_tools``
  for an agent registry). Future releases add context-awareness across the session.
- A single big tool defined with a huge JSON schema (e.g., a single ``submit``
  tool whose schema enumerates 200 enum values). The fix is the same — prune
  the schema — but we may want to surface it as a different signal in the future.

DoS hardening:

- Per-tool serialisation is truncated at ``tool_definition_bloat.max_tool_bytes``
  (default 256KB) for the ``top_tools_by_size`` evidence list. The truncation
  bounds memory at known cost; the rule still records the **original**
  uncapped serialised byte count for the threshold comparison so it
  continues to fire correctly on a single-but-massive tool definition.

- If the cumulative serialised size exceeds
  ``tool_definition_bloat.max_total_bytes`` (default 5MB), the rule
  short-circuits and emits a low-confidence (0.50) event with
  ``evidence={"truncated": True}`` rather than continuing to crunch
  potentially gigabytes of malformed input. The customer still gets a
  signal that something is wrong; the SDK doesn't blow up their process.

Privacy:

- ``tool_definition_bloat.redact_names`` (default ``False``) controls whether
  ``evidence["top_tools_by_size"]`` ships the literal tool name or only a
  16-hex-char SHA-256 prefix of the name. The default preserves the
  shape (``{"name": "...", "bytes": ...}``) so existing handlers keep working.
  Customers whose tool surface uses internally-named tools (e.g.,
  ``"create_payroll_run"``, ``"get_employee_ssn"``) should set this to True;
  the redacted shape is ``{"hash": "abcdef0123456789", "bytes": ...}`` and
  still lets the handler rank/dedupe tools by the (stable) hash without
  exfiltrating the literal name to whatever destination the handler ships
  events to.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

DEFAULT_MAX_TOOL_BYTES = 262_144  # 256 KB per tool serialisation
DEFAULT_MAX_TOTAL_BYTES = 5_242_880  # 5 MB cumulative cap before short-circuit


class ToolDefinitionBloatRule(Rule):
    name = "tool_definition_bloat"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        call = session[-1]
        tools = _extract_tools(call.raw_request)
        if not tools:
            return None

        tool_count_threshold = self.get("tool_count_threshold", 30)
        bytes_threshold = self.get("tool_definition_bytes_threshold", 30_000)
        max_tool_bytes = self.get("max_tool_bytes", DEFAULT_MAX_TOOL_BYTES)
        max_total_bytes = self.get("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES)
        redact_names = bool(self.get("redact_names", False))

        # First pass: bounded per-tool serialisation. We need (a) the per-tool
        # original byte size for accurate threshold comparison and (b) a
        # truncated form for evidence reporting. We compute both in one pass
        # and short-circuit if cumulative cost exceeds the total cap — this
        # protects the customer's process from a malicious / runaway tool
        # block without losing the signal entirely.
        try:
            sized = _serialise_tools_bounded(
                tools,
                max_tool_bytes=max_tool_bytes,
                max_total_bytes=max_total_bytes,
            )
        except Exception:
            return None

        tool_count = len(tools)

        if sized.get("short_circuited"):
            # The tool block was so large we couldn't fully evaluate it.
            # Emit a low-confidence event so the customer is still alerted,
            # but mark it as truncated.
            return LeakEvent(
                type="tool_definition_bloat",
                confidence=0.50,
                project=project,
                session_id=call.session_id,
                rule="v0.tool_definition_bloat",
                evidence={
                    "tool_count": tool_count,
                    "definition_bytes": sized["definition_bytes"],
                    "truncated": True,
                    "note": "tool block too large to evaluate",
                },
                estimated_burn=0.0,
                suggested_action="reduce_tool_count_or_use_lazy_tool_loading",
            )

        definition_bytes = sized["definition_bytes"]
        # Industry rule of thumb: ~4 bytes per token for English JSON. Calibrated
        # empirically against tiktoken on a sample of public MCP server tool
        # definitions — within ~8% across the sample.
        estimated_tokens = definition_bytes // 4

        count_breached = tool_count >= tool_count_threshold
        bytes_breached = definition_bytes >= bytes_threshold
        if not (count_breached or bytes_breached):
            return None

        # Confidence ladder:
        # - 0.85 baseline if either threshold tripped
        # - 0.95 if BOTH are tripped by >50% (e.g., 45 tools at 45KB)
        confidence = 0.85
        count_severe = tool_count >= int(tool_count_threshold * 1.5)
        bytes_severe = definition_bytes >= int(bytes_threshold * 1.5)
        if count_severe and bytes_severe:
            confidence = 0.95

        recent_calls = _recent_call_count(session)
        # Tools are re-sent on every call. Even if just one call carries them
        # right now, the instrumented agent will keep doing so — extrapolate.
        # Use the recent call count as the multiplier to give the customer a
        # near-term burn estimate they can act on. 9e-6 USD/token matches the
        # placeholder used elsewhere (see tool_loop._estimate_burn).
        estimated_burn = round(estimated_tokens * 9e-6 * max(recent_calls, 1), 4)

        return LeakEvent(
            type="tool_definition_bloat",
            confidence=confidence,
            project=project,
            session_id=call.session_id,
            rule="v0.tool_definition_bloat",
            evidence={
                "tool_count": tool_count,
                "definition_bytes": definition_bytes,
                "estimated_tokens": estimated_tokens,
                "top_tools_by_size": _top_tools_by_size(
                    sized["per_tool"], k=5, redact_names=redact_names
                ),
                "count_threshold": tool_count_threshold,
                "bytes_threshold": bytes_threshold,
                "recent_call_count": recent_calls,
            },
            estimated_burn=estimated_burn,
            suggested_action="reduce_tool_count_or_use_lazy_tool_loading",
        )


def _extract_tools(raw_request: dict[str, Any]) -> list[Any]:
    """Pull tool definitions out of either Anthropic/OpenAI- or Bedrock-shaped
    requests.

    - Anthropic / OpenAI / Gemini / Vertex: ``raw_request['tools']``
    - Bedrock Converse: ``raw_request['toolConfig']['tools']``
    """
    tools = raw_request.get("tools")
    if isinstance(tools, list):
        return tools
    tool_config = raw_request.get("toolConfig")
    if isinstance(tool_config, dict):
        nested = tool_config.get("tools")
        if isinstance(nested, list):
            return nested
    return []


def _serialise_tools_bounded(
    tools: list[Any],
    *,
    max_tool_bytes: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    """Serialise each tool with a per-tool truncation cap and a total budget.

    Returns a dict with:

    - ``per_tool``: list of ``(name, original_bytes, truncated_str)`` triples
      for the tools that fit under the budget.
    - ``definition_bytes``: sum of **original** (uncapped) byte sizes —
      this is what the threshold comparison should use so a single huge
      tool still fires.
    - ``short_circuited``: True iff cumulative cost exceeded
      ``max_total_bytes`` and we bailed early. When True, ``definition_bytes``
      is the partial sum at the bail point.

    The per-tool truncation is intentional. We need the *original* byte count
    for accurate threshold comparison (a single 1MB tool def is a bloat
    signal even if we truncated it for evidence), but we don't want to keep
    the 1MB string sitting in evidence for a customer's leak handler.
    """
    per_tool: list[tuple[str, int, str]] = []
    total = 0
    for tool in tools:
        try:
            full = json.dumps(tool, sort_keys=True, default=str)
        except Exception:
            # Unserialisable tool def: skip it for evidence but don't crash.
            continue
        original_bytes = len(full.encode("utf-8"))
        # Bound cumulative cost. If we'd blow past the total budget by
        # serialising this tool we bail entirely — the rule emits a low-
        # confidence "too big to evaluate" event in the caller.
        if total + original_bytes > max_total_bytes:
            return {
                "per_tool": per_tool,
                "definition_bytes": total + original_bytes,
                "short_circuited": True,
            }
        # Truncate the kept string for evidence. We keep the original length
        # for the threshold comparison.
        truncated = full if len(full) <= max_tool_bytes else full[:max_tool_bytes]
        per_tool.append((_tool_name(tool), original_bytes, truncated))
        total += original_bytes
    return {
        "per_tool": per_tool,
        "definition_bytes": total,
        "short_circuited": False,
    }


def _top_tools_by_size(
    per_tool: list[tuple[str, int, str]],
    *,
    k: int = 5,
    redact_names: bool = False,
) -> list[dict[str, Any]]:
    """Return the top-``k`` tools by serialised byte cost.

    Output is a list of ``{"name": str, "bytes": int}`` (default) or
    ``{"hash": str, "bytes": int}`` (when ``redact_names=True``) so the
    customer's leak handler can render "these are the tools to prune first"
    without reaching back into raw payloads. ``bytes`` is the **original**
    uncapped size so customers get accurate ranking even when truncation
    kicked in.

    The ``hash`` form is a 16-hex-char SHA-256 prefix of the literal tool
    name. It is stable across calls (same name → same hash) so handlers can
    still group / dedupe by tool, but never see the name itself. This is
    intended for customers whose tool names map to sensitive internal API
    surface (e.g., ``"create_payroll_run"``).
    """
    sized: list[tuple[str, int]] = [(name, original) for name, original, _ in per_tool]
    sized.sort(key=lambda x: x[1], reverse=True)
    if redact_names:
        return [{"hash": _hash_name(n), "bytes": b} for n, b in sized[:k]]
    return [{"name": n, "bytes": b} for n, b in sized[:k]]


def _hash_name(name: str) -> str:
    """16-hex-char SHA-256 prefix of ``name`` for the redacted evidence shape.

    16 hex chars = 64 bits of entropy — collision-resistant enough for the
    "rank top-5 tools by size" use case without leaking the literal name.
    """
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:16]


def _tool_name(tool: Any) -> str:
    """Best-effort tool-name extraction across Anthropic/OpenAI/Bedrock shapes.

    - Anthropic: ``{"name": "search", "input_schema": {...}}``
    - OpenAI: ``{"type": "function", "function": {"name": "search", ...}}``
    - Bedrock: ``{"toolSpec": {"name": "search", ...}}``
    - Pydantic AI / others: a plain object with a ``.name`` attribute.
    """
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            return name
        function = tool.get("function")
        if isinstance(function, dict):
            fn_name = function.get("name")
            if isinstance(fn_name, str):
                return fn_name
        spec = tool.get("toolSpec")
        if isinstance(spec, dict):
            spec_name = spec.get("name")
            if isinstance(spec_name, str):
                return spec_name
    name_attr = getattr(tool, "name", None)
    if isinstance(name_attr, str):
        return name_attr
    return "unknown"


def _recent_call_count(session: list[CallRecord]) -> int:
    """Number of calls observed in this session so far.

    We deliberately use the full session, not a time window: the cost of
    tool-def bloat is ``defs × calls``, and an agent that's already done 200
    calls in a session is likely to do another 200. Using the full count gives
    the leak handler a more honest dollar number than a 60-second window
    would.
    """
    return len(session)
