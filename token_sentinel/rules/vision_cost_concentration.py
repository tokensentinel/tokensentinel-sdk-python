"""Rule: Gemini call where image tokens dominate while the output is tiny.

Gemini is the only major provider that returns a per-modality breakdown
of prompt tokens via ``usage_metadata.prompt_tokens_details`` (a list of
``ModalityTokenCount`` entries, one per modality). Per internal research:

    Because Gemini *does* expose per-modality token counts, we can
    implement ``vision_cost_concentration`` — fires if >X% of session
    tokens came from images while the agent's output is text-only
    (suggests the images are not informing the output).

The fire condition: ``image_tokens > 80% of total prompt_tokens`` AND
``completion_tokens < 50``. The second clause is what distinguishes
"image-heavy prompt drove a real answer" from "stuffed 30 images into a
prompt that only needed a yes/no answer." In the former case the model
genuinely consumed the images to write a multi-paragraph description; in
the latter the customer is paying for image processing the model never
used.

Why Gemini-only
===============

Anthropic and OpenAI return only the combined ``input_tokens`` /
``prompt_tokens`` count — no text/image split. The same heuristic could
in principle approximate-fire on Anthropic by counting ``image`` blocks
× ~1,600 (Anthropic's per-1568px-image rate), but the approximation is
brittle (image dimension affects the count) and the rule would be
firing on a re-derivation of Anthropic's formula. We defer that to
and keep  Gemini-only where we have authoritative numbers.

Source of the modality breakdown
================================

The Gemini wrapper at ``wrappers/gemini.py`` currently captures
``usage_metadata.prompt_token_count`` + ``.candidates_token_count`` as
the combined token totals. **It does not yet capture the per-modality
breakdown.** This rule reads the breakdown from
``CallRecord.raw_response_meta['prompt_tokens_details']`` when the
wrapper surfaces it; without that field present the rule short-circuits
to ``None`` and never fires. The wrapper enhancement is included in
this same  deliverable so the rule has a data source to read from.

The shape we expect on ``raw_response_meta``:

    raw_response_meta = {
        ...,
        "prompt_tokens_details": [
            {"modality": "TEXT",  "token_count": 24},
            {"modality": "IMAGE", "token_count": 1560},
            {"modality": "VIDEO", "token_count": 0},
        ],
    }

Confidence
==========

0.65. Lower than ``vision_high_detail_misroute`` (0.75) because the
"images that didn't inform the output" verdict is genuinely fuzzy: the
model may have used the images to confirm a negative ("the screenshot
shows no errors → answer is 'OK'") which is a legitimate short answer.
We bias toward fewer false-positives by clearing the high bar of
"image_tokens > 80% of prompt".
"""

from __future__ import annotations

from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Modality string we look for. Gemini emits ``"IMAGE"`` in uppercase
# (matching the ``Modality`` enum's wire format) but the SDK occasionally
# downcases on Python-side normalisation; we match case-insensitively.
_IMAGE_MODALITY_NAMES: frozenset[str] = frozenset({"image", "images"})


class VisionCostConcentrationRule(Rule):
    name = "vision_cost_concentration"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None
        c = session[-1]

        # Gemini-only. Other providers don't ship the per-modality
        # breakdown so we can't run the same heuristic without an
        # approximation we'd rather not bake in.
        if c.provider != "gemini":
            return None

        # Read the modality breakdown. Wrapper-captured location; falls
        # back to None when the wrapper hasn't surfaced the field yet.
        image_tokens = _extract_image_token_count(c.raw_response_meta)
        if image_tokens is None:
            return None

        image_share_threshold = self.get("image_share_threshold", 0.80)
        max_completion_tokens = self.get("max_completion_tokens", 50)

        total_prompt = c.prompt_tokens
        if total_prompt <= 0:
            return None
        share = image_tokens / total_prompt
        if share < image_share_threshold:
            return None
        if c.completion_tokens >= max_completion_tokens:
            return None

        return LeakEvent(
            type="vision_cost_concentration",
            confidence=0.65,
            project=project,
            session_id=c.session_id,
            rule="v0.vision_cost_concentration",
            evidence={
                "model": c.model,
                "image_tokens": image_tokens,
                "prompt_tokens": total_prompt,
                "completion_tokens": c.completion_tokens,
                "image_share": round(share, 3),
            },
            # ~$0.000275 per Gemini 2.5 Pro input-token at May 2026
            # pricing. Image tokens that didn't inform the output are
            # the wasted fraction we want to surface.
            estimated_burn=round(image_tokens * 2.75e-7, 4),
            suggested_action="reduce_image_count_or_resolution",
        )


def _extract_image_token_count(raw_response_meta: Any) -> int | None:
    """Pull the image-modality token count out of ``raw_response_meta``.

    Returns ``None`` if the field is absent / malformed. Callers should
    short-circuit on ``None`` rather than treating it as zero — the
    distinction matters: zero means "Gemini said no image tokens", which
    is a valid signal we DON'T want to fire the rule on (the call had no
    images), while None means "we don't know" (the wrapper didn't
    surface the breakdown).

    Shape expected:

        raw_response_meta["prompt_tokens_details"] = [
            {"modality": "IMAGE", "token_count": 1560},
            {"modality": "TEXT",  "token_count": 24},
            ...
        ]

    Items may be dicts or attribute-bearing SDK objects; we handle both.
    """
    if not isinstance(raw_response_meta, dict):
        return None
    details = raw_response_meta.get("prompt_tokens_details")
    if details is None:
        return None
    if not isinstance(details, list):
        return None

    total = 0
    found_any = False
    for entry in details:
        modality = _get_field(entry, "modality")
        if not isinstance(modality, str):
            continue
        if modality.lower() not in _IMAGE_MODALITY_NAMES:
            continue
        count = _get_field(entry, "token_count")
        if count is None:
            count = _get_field(entry, "tokenCount")
        if isinstance(count, int):
            total += count
            found_any = True
    if not found_any:
        return None
    return total


def _get_field(entry: Any, name: str) -> Any:
    """Read ``name`` off ``entry`` whether it's a dict or an attribute object."""
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name, None)
