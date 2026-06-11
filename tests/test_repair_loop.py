"""Tests for ``RepairLoopRule`` .

Defaults: ``min_corrections=2``, ``window_turns=10``,
``similarity_threshold=0.7``, ``length_ratio=0.8``. Confidence is
``0.65 + 0.05 * (corrections - min_corrections)`` capped at ``0.9``.

The rule walks ``session[-1].raw_request["messages"]`` so each test
constructs a single ``CallRecord`` whose ``raw_request`` carries the
conversation history under test. Live LLM calls are NOT used —
everything is synthetic chat history.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from token_sentinel.events import CallRecord
from token_sentinel.rules.repair_loop import RepairLoopRule


def _evaluate(rule: RepairLoopRule, calls: list[CallRecord], project: str = "proj") -> Any:
    return rule.evaluate(calls, project=project)


def _make_call_with_messages(
    make_call: Any,
    now: Any,
    messages: list[dict[str, Any]],
    *,
    offset_s: float = 0.0,
    prompt_tokens: int = 1500,
) -> CallRecord:
    """Build a CallRecord whose raw_request carries ``messages``.

    Mirrors the Anthropic / OpenAI wrapper shape — a single chat-shape
    call with a populated history.
    """
    return make_call(
        timestamp=now + timedelta(seconds=offset_s),
        prompt_tokens=prompt_tokens,
        raw_request={"messages": messages, "tools": [], "max_tokens": 1024},
        user_facing_output=True,
    )


# A long agent answer (~280 chars) used as the regenerated content
# across multiple turns. Repeating it keeps TF-IDF char-3-gram cosine
# at ~1.0, which is well above the 0.7 default threshold.
_LONG_AGENT_ANSWER = (
    "The revenue for the second quarter was approximately $4.2 million dollars, "
    "with the majority of growth coming from the enterprise segment in North "
    "America. The product line that contributed most was the analytics platform, "
    "followed by the data warehouse integration product line."
)

_LONG_AGENT_ANSWER_VARIANT = (
    "The revenue for the second quarter was approximately $4.2 million dollars, "
    "with the majority of growth driven by the enterprise segment in North "
    "America. The leading product was the analytics platform, followed by the "
    "data warehouse integration product."
)

# A genuinely different agent response (below similarity threshold).
_DIFFERENT_AGENT_ANSWER = (
    "Q3 totaled $5.8 million in revenue. The growth was primarily fueled by our "
    "Asia-Pacific operations, with new partnerships in Japan and Singapore. The "
    "managed services line led the way, followed by training revenue."
)


# ---------------------------------------------------------------------------
# Positive cases (3+)
# ---------------------------------------------------------------------------


def test_two_corrections_fire_at_base_confidence(make_call: Any, now: Any) -> None:
    """Exactly 2 corrections with similar regenerations fires at 0.65."""
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "what was the revenue last quarter"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, that's wrong, I asked about Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
        {"role": "user", "content": "actually no, I meant Q3 not Q2"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.type == "repair_loop"
    assert ev.rule == "v0.repair_loop"
    assert ev.evidence["correction_count"] == 2
    assert ev.evidence["window_turns"] == 10
    assert ev.evidence["mean_similarity"] >= 0.7
    assert ev.suggested_action == "surface_correction_pattern_to_engineer"
    assert ev.confidence == pytest.approx(0.65)


def test_three_corrections_scale_confidence(make_call: Any, now: Any) -> None:
    """3 corrections (min+1) → confidence 0.7."""
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no that's not right"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
        {"role": "user", "content": "actually I meant Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, wrong again"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["correction_count"] == 3
    assert ev.confidence == pytest.approx(0.70)


def test_six_corrections_capped_at_0_9(make_call: Any, now: Any) -> None:
    """Many corrections must clamp the confidence at 0.9."""
    rule = RepairLoopRule({})
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    # Append 6 correction+regen pairs.
    for _ in range(6):
        messages.append({"role": "user", "content": "no, wrong"})
        messages.append({"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT})
    rule = RepairLoopRule({"repair_loop.window_turns": 20})
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["correction_count"] == 6
    # 0.65 + 0.05 * (6 - 2) = 0.85; with 6 it would be 0.85 — still < 0.9.
    # The cap test is exercised at higher repetition counts:
    assert ev.confidence <= 0.9


def test_eight_corrections_hits_the_cap(make_call: Any, now: Any) -> None:
    """8 corrections → 0.65 + 0.05 * 6 = 0.95 → clamped to 0.9."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    for _ in range(8):
        messages.append({"role": "user", "content": "no, wrong"})
        messages.append({"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT})
    rule = RepairLoopRule({"repair_loop.window_turns": 25})
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Negative cases (4+)
# ---------------------------------------------------------------------------


def test_single_correction_does_not_fire(make_call: Any, now: Any) -> None:
    """min_corrections defaults to 2 — a lone correction is normal."""
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, that's wrong, I asked about Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


def test_corrections_outside_window_dont_fire(make_call: Any, now: Any) -> None:
    """Corrections older than ``window_turns`` are excluded.

    Build a conversation where the only correction turns sit at the
    BEGINNING, then pad with many non-correction turns so they fall out
    of the 10-turn tail window the rule examines.
    """
    rule = RepairLoopRule({})
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, that's wrong"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
        {"role": "user", "content": "actually I meant Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    # Pad with 12 collaborative turns — pushes the early corrections
    # out of the 10-turn tail.
    for i in range(6):
        messages.append({"role": "user", "content": f"please add detail about region {i}"})
        messages.append(
            {"role": "assistant", "content": f"Region {i} grew by {10 + i}% year over year."}
        )
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


def test_different_sessions_dont_link(make_call: Any, now: Any) -> None:
    """Rule operates on the latest call's history — earlier sessions don't bleed in."""
    # The rule only reads ``session[-1].raw_request["messages"]``. A
    # prior CallRecord in the list that BELONGS to a different
    # session must NOT influence the result. Set up two records: a
    # noisy "other session" record first, then a clean current
    # record with no corrections.
    rule = RepairLoopRule({})
    other_session_messages = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no that's wrong"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
        {"role": "user", "content": "actually I meant Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    current_session_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
    ]
    other = make_call(
        session_id="other",
        timestamp=now,
        raw_request={"messages": other_session_messages, "tools": [], "max_tokens": 100},
        user_facing_output=True,
    )
    current = make_call(
        session_id="current",
        timestamp=now + timedelta(seconds=10),
        raw_request={"messages": current_session_messages, "tools": [], "max_tokens": 100},
        user_facing_output=True,
    )
    session = [other, current]
    assert _evaluate(rule, session) is None


def test_below_similarity_threshold_doesnt_fire(make_call: Any, now: Any) -> None:
    """If the agent's regeneration is materially different, no fire.

    Two corrections present, but the agent's outputs diverge wildly —
    the customer IS getting value from the back-and-forth, even if
    their wording sounds corrective.
    """
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, I meant Q3"},
        {"role": "assistant", "content": _DIFFERENT_AGENT_ANSWER},
        {"role": "user", "content": "actually no, I meant Q4"},
        {
            "role": "assistant",
            "content": (
                "Q4 closed at $7.1M, driven by EMEA enterprise renewals. "
                "Top contributor was the security suite, followed by the data "
                "platform expansion in the German market."
            ),
        },
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


def test_no_user_facing_output_doesnt_fire(make_call: Any, now: Any) -> None:
    """Tool-only sessions (no agent text turns) cannot exhibit repair loops.

    With no assistant text content there is no regeneration to compare
    against — the similarity gate degrades to <2 documents and the
    rule short-circuits to None.
    """
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "what was Q2 revenue"},
        # Assistant turn carries only a tool_use block — no text content.
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "search", "input": {"q": "revenue Q2"}}],
        },
        {"role": "user", "content": "no that's wrong"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "search", "input": {"q": "revenue Q3"}}],
        },
        {"role": "user", "content": "no still wrong"},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


def test_collaborative_refinement_doesnt_fire(make_call: Any, now: Any) -> None:
    """A user writing LONGER clarifications than the agent's terse answer is collaborating.

    The length-ratio gate (user must be < 80% of agent length) drops
    these turns even though they contain "actually" / "instead".
    """
    rule = RepairLoopRule({})
    messages = [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "Brief."},
        {
            "role": "user",
            "content": (
                "actually I would like a much more detailed summary instead "
                "covering Q1 Q2 Q3 and Q4 with regional breakdowns and a "
                "discussion of the product mix shift over the year"
            ),
        },
        {"role": "assistant", "content": "Brief."},
        {
            "role": "user",
            "content": (
                "no instead can you also cover the cost side and the segment "
                "margins and discuss the headcount changes that drove the "
                "operating expense growth"
            ),
        },
        {"role": "assistant", "content": "Brief."},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


# ---------------------------------------------------------------------------
# Edge cases (2+)
# ---------------------------------------------------------------------------


def test_empty_session_returns_none() -> None:
    """An empty session list returns None cleanly."""
    rule = RepairLoopRule({})
    assert _evaluate(rule, []) is None


def test_single_turn_doesnt_fire(make_call: Any, now: Any) -> None:
    """A conversation with just one user turn cannot exhibit a repair loop."""
    rule = RepairLoopRule({})
    messages = [{"role": "user", "content": "hello"}]
    session = [_make_call_with_messages(make_call, now, messages)]
    assert _evaluate(rule, session) is None


def test_missing_raw_request_doesnt_crash(make_call: Any, now: Any) -> None:
    """A CallRecord with empty raw_request degrades silently."""
    rule = RepairLoopRule({})
    call = make_call(timestamp=now, raw_request={}, user_facing_output=True)
    assert _evaluate(rule, [call]) is None


def test_messages_field_missing_doesnt_crash(make_call: Any, now: Any) -> None:
    """raw_request without a 'messages' key is treated as no history."""
    rule = RepairLoopRule({})
    call = make_call(
        timestamp=now,
        raw_request={"tools": [], "max_tokens": 100},
        user_facing_output=True,
    )
    assert _evaluate(rule, [call]) is None


def test_malformed_message_entries_skipped(make_call: Any, now: Any) -> None:
    """Non-dict / wrong-role / no-content entries don't break the parser."""
    rule = RepairLoopRule({})
    messages: list[Any] = [
        "not a dict",
        {"role": "system", "content": "ignored"},
        {"role": "user"},  # no content key
        {"role": "user", "content": "what was Q2 revenue"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        {"role": "user", "content": "no, that's wrong"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT},
        {"role": "user", "content": "actually I meant Q3"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    assert ev.evidence["correction_count"] == 2


# ---------------------------------------------------------------------------
# Confidence-calculation correctness (1+)
# ---------------------------------------------------------------------------


def test_confidence_formula_exact_values(make_call: Any, now: Any) -> None:
    """Pin the exact confidence at 2, 3, 4, 5 corrections.

    Formula: ``min(0.65 + 0.05 * (n - min_corrections), 0.9)``.
    """
    expected = {2: 0.65, 3: 0.70, 4: 0.75, 5: 0.80}
    for n, want in expected.items():
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "what was Q2 revenue"},
            {"role": "assistant", "content": _LONG_AGENT_ANSWER},
        ]
        for _ in range(n):
            messages.append({"role": "user", "content": "no, wrong"})
            messages.append({"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT})
        # Bump the window so all corrections stay in scope.
        rule = RepairLoopRule({"repair_loop.window_turns": 30})
        session = [_make_call_with_messages(make_call, now, messages)]
        ev = _evaluate(rule, session)
        assert ev is not None, f"expected fire at n={n}"
        assert ev.evidence["correction_count"] == n
        assert ev.confidence == pytest.approx(want), (
            f"confidence mismatch at n={n}: got {ev.confidence}, want {want}"
        )


def test_evidence_redacts_raw_text(make_call: Any, now: Any) -> None:
    """Verify the evidence dict never contains the raw user / agent text.

    The privacy contract from the sample_args precedent: evidence
    ships only structural summaries.
    """
    rule = RepairLoopRule({})
    secret = "MY-SECRET-PROMPT-9X8Y7Z"
    messages = [
        {"role": "user", "content": f"please tell me about {secret}"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER + " " + secret},
        {"role": "user", "content": "no, that's wrong"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER_VARIANT + " " + secret},
        {"role": "user", "content": "actually I meant something else"},
        {"role": "assistant", "content": _LONG_AGENT_ANSWER + " " + secret},
    ]
    session = [_make_call_with_messages(make_call, now, messages)]
    ev = _evaluate(rule, session)
    assert ev is not None
    blob = str(ev.evidence)
    assert secret not in blob, "raw text must never appear in evidence"
    # Structural fields are present:
    assert "correction_count" in ev.evidence
    assert "mean_similarity" in ev.evidence
    assert "matched_keywords" in ev.evidence
