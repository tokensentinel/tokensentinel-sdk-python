"""Rule: OpenAI ``detail="high"`` on a classification-shaped vision prompt.

OpenAI's vision pricing scales the tile-based token count by the
``detail`` setting on each ``image_url``:

  - ``detail="low"``:  fixed 85 tokens per image (no tiling — global only)
  - ``detail="high"``: ~170 per tile × N tiles + 85 base = ~765 for a
                       1024×1024 image (roughly **4×** the low-detail cost)
  - ``detail="auto"``: OpenAI picks; usually defaults to ``high`` for
                       images >512px in either dimension

Per internal research:

    Multiple OpenAI dev-forum threads report unexpectedly high vision
    token costs because customers don't realize ``detail="high"`` (the
    default for many SDK paths) is roughly 4× the cost of
    ``detail="low"``. This is a perfect TokenSentinel detection
    opportunity.

The pattern: a customer ships a screenshot or photo at
``detail="high"`` and asks ``"Is there a cat in this image?"``. The
prompt only needs the 85-token low-detail global view; the high-detail
tiles are pure waste.

We reuse the classification-keyword regex from :mod:`model_misroute` so
the firing semantics stay consistent between text-only misrouting and
vision-detail misrouting. The match logic is otherwise straightforward:

  1. Provider is OpenAI (only OpenAI exposes a ``detail`` knob).
  2. At least one ``image_url`` block has ``detail in {"high", "auto"}``
     OR no ``detail`` field at all (OpenAI's default is "auto" → usually
     "high").
  3. The prompt text matches a classify keyword at word boundary.

Confidence: 0.75. Slightly higher than ``model_misroute`` (0.7) because
the fix is more mechanical — there's no judgement call about whether
the model is appropriate; the only question is whether the image
resolution is needed, and classification prompts almost never need it.
"""

from __future__ import annotations

from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Re-use the classification-keyword regex from ``model_misroute`` so the
# two rules can't drift on keyword definitions. The regex is pre-compiled
# at module-load there and is private-but-stable ( contract: do not
# rename without updating both rules).
from token_sentinel.rules.model_misroute import _KEYWORD_PATTERN, _flatten_messages

# Detail values that mean "use the expensive high-resolution tiles":
# explicit ``"high"`` plus ``"auto"`` (which OpenAI documents as
# "auto-select; usually high for large images") plus missing (which
# defaults to ``"auto"`` per the OpenAI Python SDK).
_HIGH_DETAIL_VALUES: frozenset[str | None] = frozenset({"high", "auto", None})


class VisionHighDetailMisrouteRule(Rule):
    name = "vision_high_detail_misroute"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None
        c = session[-1]

        # OpenAI is the only provider with a ``detail`` knob — short-
        # circuit for everything else so we don't false-positive on
        # Anthropic / Gemini calls. The provider check is done by
        # string-equality on ``CallRecord.provider`` rather than
        # parsing the model name; the wrapper sets this authoritatively.
        if c.provider != "openai":
            return None

        max_completion_tokens = self.get("max_completion_tokens", 50)
        if c.completion_tokens > max_completion_tokens:
            return None

        # Walk the request and find image_url blocks with high-detail.
        image_blocks = _collect_high_detail_images(c.raw_request)
        if not image_blocks:
            return None

        # Now check the classification-keyword shape against the prompt.
        prompt_text = _flatten_messages(c.raw_request.get("messages") or [])
        raw_matches = _KEYWORD_PATTERN.findall(prompt_text)
        if not raw_matches:
            return None
        seen: set[str] = set()
        matched: list[str] = []
        for m in raw_matches:
            normalised = m.lower()
            if normalised not in seen:
                seen.add(normalised)
                matched.append(normalised)

        return LeakEvent(
            type="vision_high_detail_misroute",
            confidence=0.75,
            project=project,
            session_id=c.session_id,
            rule="v0.vision_high_detail_misroute",
            evidence={
                "model": c.model,
                "image_count": len(image_blocks),
                # Per-image detail value as the SDK saw it. ``None``
                # means the customer omitted the field; OpenAI then
                # defaults to ``"auto"`` (which typically resolves to
                # high). Keeping the raw value tells the dashboard
                # exactly what the customer is doing.
                "detail_values": [b.get("detail") for b in image_blocks],
                "matched_keywords": matched[:3],
            },
            # Burn estimate: ~4× delta per image at OpenAI's
            # ~$5/MTok vision rate ≈ 765 - 85 = 680 wasted tokens per
            # image. ($0.0034 per image at GPT-4o pricing.)
            estimated_burn=round(len(image_blocks) * 0.0034, 4),
            suggested_action="use_image_detail_low_for_classification",
        )


def _collect_high_detail_images(raw_request: Any) -> list[dict[str, Any]]:
    """Return a list of ``image_url`` blocks whose detail is high (or auto/missing).

    Returns an empty list if ``raw_request`` doesn't look like an OpenAI
    chat-completion shape — defensive guard so the rule short-circuits
    cleanly on unexpected inputs rather than crashing.
    """
    if not isinstance(raw_request, dict):
        return []
    out: list[dict[str, Any]] = []
    for message in raw_request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image_url":
                continue
            image_url = block.get("image_url")
            if not isinstance(image_url, dict):
                continue
            detail = image_url.get("detail")
            # Normalise: empty string == None for the lookup. The
            # SDK accepts both spellings as "unset".
            normalised = detail if (detail is None or detail) else None
            if normalised in _HIGH_DETAIL_VALUES:
                out.append({"detail": detail})
    return out
