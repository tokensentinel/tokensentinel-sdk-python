"""Tests for ``VisionHighDetailMisrouteRule``.

Defaults: max_completion_tokens=50, fires on provider=openai with
detail in {"high", "auto", None} on a classify-shaped prompt.

The rule re-uses the keyword regex from ``model_misroute`` so a single
``"classify"`` / ``"yes or no"`` / ``"label this"`` keyword anywhere in
the flattened prompt is enough — these tests pin the boundary cases.
"""

from __future__ import annotations

import pytest

from token_sentinel.rules.vision_high_detail_misroute import (
    VisionHighDetailMisrouteRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_request(
    prompt_text: str = "Classify this as cat or dog",
    *,
    detail: str | None = "high",
    image_count: int = 1,
) -> dict:
    """Build an OpenAI ``raw_request`` with N images at the chosen detail."""
    content = [{"type": "text", "text": prompt_text}]
    for _ in range(image_count):
        image_url: dict = {"url": "data:image/png;base64,iVBORw0KG"}
        if detail is not None:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return {
        "messages": [{"role": "user", "content": content}],
        "tools": [],
        "max_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert VisionHighDetailMisrouteRule({}).evaluate([], project="p") is None


def test_non_openai_provider_no_fire(make_call):
    """Only OpenAI has a ``detail`` knob. Anthropic / Gemini calls must
    short-circuit without firing — they have no equivalent waste."""
    session = [
        make_call(
            provider="anthropic",
            model="claude-sonnet-4-6",
            completion_tokens=5,
            raw_request=_openai_request(),
        )
    ]
    assert VisionHighDetailMisrouteRule({}).evaluate(session, project="p") is None


def test_detail_low_no_fire(make_call):
    """``detail="low"`` is the recommended state — never fire."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request=_openai_request(detail="low"),
        )
    ]
    assert VisionHighDetailMisrouteRule({}).evaluate(session, project="p") is None


def test_no_image_no_fire(make_call):
    """Text-only OpenAI call → no image → no fire even with a classify keyword."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": "Classify hello"}],
                "tools": [],
                "max_tokens": 0,
            },
        )
    ]
    assert VisionHighDetailMisrouteRule({}).evaluate(session, project="p") is None


def test_no_classify_keyword_no_fire(make_call):
    """High-detail image is fine when the prompt isn't classification-shaped
    (e.g., the customer genuinely needs the tiles for a description)."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request=_openai_request(
                prompt_text="Describe the architecture diagram in detail.",
                detail="high",
            ),
        )
    ]
    assert VisionHighDetailMisrouteRule({}).evaluate(session, project="p") is None


def test_long_completion_no_fire(make_call):
    """``completion_tokens > 50`` means the model generated a real
    response — the image-detail cost was probably justified."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=200,
            raw_request=_openai_request(detail="high"),
        )
    ]
    assert VisionHighDetailMisrouteRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detail", ["high", "auto", None])
def test_high_or_auto_or_missing_detail_fires(make_call, detail):
    """All three "the SDK is paying the high-detail cost" values fire."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request=_openai_request(detail=detail),
        )
    ]
    ev = VisionHighDetailMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "vision_high_detail_misroute"
    assert ev.rule == "v0.vision_high_detail_misroute"
    assert ev.confidence == 0.75
    assert ev.suggested_action == "use_image_detail_low_for_classification"
    assert ev.evidence["image_count"] == 1
    assert ev.evidence["detail_values"] == [detail]


def test_multiple_high_detail_images_all_counted(make_call):
    """Three images at ``detail="high"`` → ``image_count=3`` in evidence."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request=_openai_request(detail="high", image_count=3),
        )
    ]
    ev = VisionHighDetailMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["image_count"] == 3
    assert ev.evidence["detail_values"] == ["high", "high", "high"]
    # Burn scales linearly with image count.
    assert ev.estimated_burn == pytest.approx(0.0102, abs=1e-4)


def test_mixed_low_and_high_only_high_counts(make_call):
    """One ``detail="low"`` + one ``detail="high"`` → ``image_count=1``
    (the rule only counts the high/auto/missing images)."""
    content = [
        {"type": "text", "text": "Classify these"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,A", "detail": "low"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,B", "detail": "high"},
        },
    ]
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": content}],
                "tools": [],
                "max_tokens": 0,
            },
        )
    ]
    ev = VisionHighDetailMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["image_count"] == 1
    assert ev.evidence["detail_values"] == ["high"]


@pytest.mark.parametrize(
    "prompt_text",
    [
        "Classify this picture as cat or dog",
        "Yes or no: is this a screenshot of an error?",
        "Is this an example of clean code?",
        "Label this image with one word",
        "Which category does this belong to?",
        "True or false: this is a UI mockup",
    ],
)
def test_classify_keywords_fire(make_call, prompt_text):
    """Reuses ``model_misroute._KEYWORD_PATTERN`` — every keyword still
    matches at word boundary the same way."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=5,
            raw_request=_openai_request(prompt_text=prompt_text, detail="high"),
        )
    ]
    ev = VisionHighDetailMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert len(ev.evidence["matched_keywords"]) >= 1


def test_custom_max_completion_tokens(make_call):
    """``max_completion_tokens=10`` config — 12 completion tokens no longer fires."""
    session = [
        make_call(
            provider="openai",
            model="gpt-4o",
            completion_tokens=12,
            raw_request=_openai_request(detail="high"),
        )
    ]
    rule = VisionHighDetailMisrouteRule({"vision_high_detail_misroute.max_completion_tokens": 10})
    assert rule.evaluate(session, project="p") is None


def test_only_last_call_evaluated(make_call):
    """Like ``model_misroute``, the rule operates on ``session[-1]``."""
    earlier = make_call(
        provider="openai",
        model="gpt-4o",
        completion_tokens=5,
        raw_request=_openai_request(detail="high"),
    )
    latest = make_call(
        provider="openai",
        model="gpt-4o",
        completion_tokens=5,
        # Latest is non-classification → no fire.
        raw_request=_openai_request(
            prompt_text="Tell me a long story",
            detail="high",
        ),
    )
    assert VisionHighDetailMisrouteRule({}).evaluate([earlier, latest], project="p") is None
