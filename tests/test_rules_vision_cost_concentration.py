"""Tests for ``VisionCostConcentrationRule``.

Defaults: image_share_threshold=0.80, max_completion_tokens=50, Gemini-only.

Reads the per-modality breakdown from
``CallRecord.raw_response_meta["prompt_tokens_details"]`` as surfaced by
the  Gemini wrapper enhancement (``_extract_prompt_tokens_details``).
"""

from __future__ import annotations

from typing import Any

import pytest

from token_sentinel.rules.vision_cost_concentration import (
    VisionCostConcentrationRule,
    _extract_image_token_count,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta_with_modalities(
    text_tokens: int = 0,
    image_tokens: int = 0,
    video_tokens: int = 0,
) -> dict[str, Any]:
    """Build a ``raw_response_meta`` carrying the  modality breakdown."""
    return {
        "finish_reason": "STOP",
        "prompt_tokens_details": [
            {"modality": "TEXT", "token_count": text_tokens},
            {"modality": "IMAGE", "token_count": image_tokens},
            {"modality": "VIDEO", "token_count": video_tokens},
        ],
    }


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert VisionCostConcentrationRule({}).evaluate([], project="p") is None


def test_non_gemini_provider_no_fire(make_call):
    """Only Gemini ships per-modality breakdowns — short-circuit elsewhere
    so we don't false-positive on a fabricated breakdown on a wrong wrapper."""
    session = [
        make_call(
            provider="openai",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=100, image_tokens=900),
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


def test_no_modality_breakdown_no_fire(make_call):
    """Gemini call but ``prompt_tokens_details`` absent (older wrapper /
    text-only chunks before ) → no fire."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta={"finish_reason": "STOP"},
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


def test_no_image_tokens_no_fire(make_call):
    """Modality breakdown present, all-text prompt — no image waste to flag."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=1000, image_tokens=0),
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


def test_image_share_below_threshold_no_fire(make_call):
    """Image tokens at 50% of prompt → under the 80% default."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=500, image_tokens=500),
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


def test_long_completion_no_fire(make_call):
    """The model returned a meaningful response → images were used → no fire."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=2000,
            completion_tokens=300,
            raw_response_meta=_meta_with_modalities(text_tokens=200, image_tokens=1800),
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_image_heavy_prompt_short_answer_fires(make_call):
    """1800/2000 image-token share with a 5-token answer → fire."""
    session = [
        make_call(
            provider="gemini",
            model="gemini-2.5-pro",
            prompt_tokens=2000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=200, image_tokens=1800),
        )
    ]
    ev = VisionCostConcentrationRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "vision_cost_concentration"
    assert ev.rule == "v0.vision_cost_concentration"
    assert ev.confidence == 0.65
    assert ev.suggested_action == "reduce_image_count_or_resolution"
    # Image share evidence is rounded to 3 decimals.
    assert ev.evidence["image_share"] == 0.9
    assert ev.evidence["image_tokens"] == 1800
    assert ev.evidence["prompt_tokens"] == 2000
    assert ev.evidence["completion_tokens"] == 5
    assert ev.evidence["model"] == "gemini-2.5-pro"


def test_exactly_at_threshold_fires(make_call):
    """80% image share is the threshold — at boundary, fires (>=)."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=200, image_tokens=800),
        )
    ]
    ev = VisionCostConcentrationRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["image_share"] == 0.8


def test_custom_image_share_threshold(make_call):
    """Lower the threshold to 0.5 → a 60% share fires."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=400, image_tokens=600),
        )
    ]
    rule = VisionCostConcentrationRule({"vision_cost_concentration.image_share_threshold": 0.5})
    ev = rule.evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["image_share"] == 0.6


def test_enum_prefixed_modality_string_recognised(make_call):
    """The wrapper strips ``MediaModality.`` enum prefixes, but defend
    against an enum-bearing breakdown landing as-is — the rule's
    case-insensitive ``"image"`` check must still recognise it after
    the wrapper-side normalisation."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta={
                "finish_reason": "STOP",
                "prompt_tokens_details": [
                    {"modality": "IMAGE", "token_count": 850},
                    {"modality": "TEXT", "token_count": 150},
                ],
            },
        )
    ]
    ev = VisionCostConcentrationRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["image_tokens"] == 850


def test_attribute_bearing_objects_extracted(make_call):
    """``prompt_tokens_details`` items expose ``modality``/``token_count``
    as attributes (the SDK object shape) — extractor handles both."""

    class _ModalityCount:
        def __init__(self, modality: str, token_count: int):
            self.modality = modality
            self.token_count = token_count

    meta = {
        "prompt_tokens_details": [
            _ModalityCount("IMAGE", 900),
            _ModalityCount("TEXT", 100),
        ]
    }
    assert _extract_image_token_count(meta) == 900


def test_extract_returns_none_when_field_absent():
    """Distinguishing "no breakdown" (None) from "zero image tokens" (0)
    is what stops the rule firing on text-only Gemini calls."""
    assert _extract_image_token_count({"finish_reason": "STOP"}) is None
    assert _extract_image_token_count(None) is None
    assert _extract_image_token_count({"prompt_tokens_details": None}) is None


def test_estimated_burn_proportional_to_image_tokens(make_call):
    """Sanity: burn estimate is positive and scales with image tokens."""
    session_small = [
        make_call(
            provider="gemini",
            prompt_tokens=1000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=100, image_tokens=900),
        )
    ]
    session_large = [
        make_call(
            provider="gemini",
            prompt_tokens=10000,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(text_tokens=1000, image_tokens=9000),
        )
    ]
    ev_small = VisionCostConcentrationRule({}).evaluate(session_small, project="p")
    ev_large = VisionCostConcentrationRule({}).evaluate(session_large, project="p")
    assert ev_small is not None and ev_large is not None
    assert ev_large.estimated_burn > ev_small.estimated_burn


def test_only_last_call_evaluated(make_call):
    """Like other one-call rules, evaluates ``session[-1]`` only."""
    earlier = make_call(
        provider="gemini",
        prompt_tokens=2000,
        completion_tokens=5,
        raw_response_meta=_meta_with_modalities(text_tokens=200, image_tokens=1800),
    )
    latest = make_call(
        provider="gemini",
        prompt_tokens=2000,
        # Latest has a long response → no fire even though earlier would have.
        completion_tokens=300,
        raw_response_meta=_meta_with_modalities(text_tokens=200, image_tokens=1800),
    )
    assert VisionCostConcentrationRule({}).evaluate([earlier, latest], project="p") is None


def test_zero_prompt_tokens_no_crash(make_call):
    """Defensive: ``prompt_tokens=0`` (degenerate case) → no division-by-zero crash."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=0,
            completion_tokens=5,
            raw_response_meta=_meta_with_modalities(image_tokens=0),
        )
    ]
    assert VisionCostConcentrationRule({}).evaluate(session, project="p") is None


@pytest.mark.parametrize(
    "image_tokens, prompt_tokens, completion_tokens, should_fire",
    [
        (900, 1000, 5, True),  # 0.9 share, short answer
        (800, 1000, 5, True),  # 0.8 share at threshold
        (700, 1000, 5, False),  # 0.7 share below threshold
        (900, 1000, 50, False),  # at completion_tokens limit (>=)
        (900, 1000, 49, True),  # one below the limit
    ],
)
def test_fire_matrix(make_call, image_tokens, prompt_tokens, completion_tokens, should_fire):
    """Parametrised boundary matrix."""
    session = [
        make_call(
            provider="gemini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response_meta=_meta_with_modalities(
                text_tokens=prompt_tokens - image_tokens,
                image_tokens=image_tokens,
            ),
        )
    ]
    ev = VisionCostConcentrationRule({}).evaluate(session, project="p")
    if should_fire:
        assert ev is not None
    else:
        assert ev is None
