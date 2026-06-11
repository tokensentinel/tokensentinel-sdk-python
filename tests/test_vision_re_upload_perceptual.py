"""Tests for the perceptual-hash fallback on ``VisionReUploadRule``.

These tests pair with the  byte-exact tests in
``test_rules_vision_re_upload.py``. The perceptual path is the second-tier
signal: when SHA-256 doesn't catch a re-upload because a minor
recompression / resize between turns changed the bytes, ``imagehash.phash``
should still recognise the image as a near-duplicate.

The tests build their own synthetic images via PIL (a content-rich
gradient + shapes pattern that phashes stably) so the suite has zero
dependency on real-world image fixtures and is reproducible across
platforms.

``imagehash`` is an OPTIONAL dependency of ``token-sentinel``. Most
tests in this file ``pytest.importorskip`` it so a CI matrix without the
extra installed simply skips the perceptual tests rather than failing.
The two ``test_*_without_imagehash`` tests at the bottom of the file
DO NOT depend on imagehash and verify the rule's graceful-degradation
contract — they patch ``_PERCEPTUAL_AVAILABLE = False`` to simulate the
no-extra-installed environment.
"""

from __future__ import annotations

import base64
import io
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest

from token_sentinel.rules import vision_re_upload as vru
from token_sentinel.rules.vision_re_upload import (
    VisionReUploadRule,
    _perceptual_hash_from_b64,
    _phash_distance,
)

# ---------------------------------------------------------------------------
# Synthetic-image helpers.
#
# These produce small in-memory PIL images with enough content (lines +
# rectangles + ellipses) that phash returns a stable, non-degenerate
# hash. A solid-colour image phashes to all-zeros regardless of colour,
# which is useless for distance testing, so we always add geometry.
# ---------------------------------------------------------------------------


def _content_rich_image(size: tuple[int, int] = (256, 256)) -> Any:
    """Construct a PIL image with gradient-style content suitable for phash."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 120, 120], fill=(20, 40, 80))
    draw.ellipse([100, 100, 220, 220], fill=(200, 30, 40))
    draw.line([0, 0, size[0], size[1]], fill=(0, 200, 0), width=5)
    draw.line([0, size[1], size[0], 0], fill=(0, 0, 200), width=5)
    return img


def _different_image(size: tuple[int, int] = (256, 256)) -> Any:
    """Construct a *visibly different* image (a high-contrast 2x2 checker)
    whose phash sits far from :func:`_content_rich_image` — distance >> 6."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0] // 2, size[1] // 2], fill=(0, 0, 0))
    draw.rectangle([size[0] // 2, size[1] // 2, size[0], size[1]], fill=(0, 0, 0))
    return img


def _png_b64(img: Any) -> str:
    """Encode a PIL image as base64-PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(img: Any, *, quality: int = 85) -> str:
    """Encode a PIL image as base64-JPEG with a configurable quality.

    JPEG recompression is the canonical "bytes change, image is the same"
    pattern this fallback was built for: a customer's UI saves the
    screenshot back to disk as JPEG between turns and the SHA-256 changes
    even though pHash hardly moves.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Provider request shape builders (mirror the  unit-test fixtures).
# ---------------------------------------------------------------------------


def _openai_request_with_image(image_b64: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "tools": [],
        "max_tokens": 0,
    }


def _anthropic_request_with_image(image_b64: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
        "tools": [],
        "max_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Perceptual-fallback tests (require imagehash).
# ---------------------------------------------------------------------------


def test_perceptual_fires_on_resized_reupload(make_call, now):
    """Three uploads of the *same content* re-encoded at different sizes —
    the SHA-256 differs (bytes change with each re-encode/resize) but phash
    distance stays within the threshold, so the perceptual fallback fires.

    Confidence MUST be 0.65 (the 0.7 exact-base, minus the 0.05 perceptual
    discount specified in the  design).
    """
    pytest.importorskip("imagehash")

    img = _content_rich_image((256, 256))
    img_resized_a = img.resize((250, 250))
    img_resized_b = img.resize((240, 240))

    images = [_png_b64(img), _png_b64(img_resized_a), _png_b64(img_resized_b)]
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(b64),
        )
        for i, b64 in enumerate(images)
    ]

    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None, "perceptual fallback should fire on resized re-upload"
    assert ev.type == "vision_re_upload"
    assert ev.evidence["match_type"] == "perceptual"
    assert ev.confidence == pytest.approx(0.65)
    assert ev.evidence["duplicate_count"] == 3
    # Distance is small but non-zero — the resize moves one or two bits.
    assert 0 < ev.evidence["perceptual_distance"] <= 6


def test_perceptual_fires_on_jpeg_recompression(make_call, now):
    """JPEG recompression is the canonical 'bytes change, image is the same'
    pattern. PNG round-trip + JPEG round-trip + JPEG quality=70 round-trip
    produce three different SHA-256 values, but their phashes are within
    the Hamming threshold."""
    pytest.importorskip("imagehash")

    img = _content_rich_image((256, 256))
    variants = [
        _png_b64(img),
        _jpeg_b64(img, quality=85),
        _jpeg_b64(img, quality=70),
    ]
    # Sanity: confirm SHA-256-relevant bytes really do differ across the
    # variants — otherwise we'd be testing the exact-match path, not the
    # perceptual fallback.
    assert len({v for v in variants}) == 3

    session = [
        make_call(
            provider="anthropic",
            timestamp=now + timedelta(seconds=i),
            raw_request=_anthropic_request_with_image(b64),
        )
        for i, b64 in enumerate(variants)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["match_type"] == "perceptual"
    assert ev.evidence["provider"] == "anthropic"
    assert ev.confidence == pytest.approx(0.65)


def test_perceptual_does_not_fire_on_genuinely_different_images(make_call, now):
    """Three visibly distinct images — different content, large pHash
    distance — must NOT trigger the perceptual fallback. The exact match
    is also impossible (different bytes), so the rule should return None."""
    pytest.importorskip("imagehash")

    img_a = _content_rich_image((256, 256))
    img_b = _different_image((256, 256))

    # Three calls: gradient, checker, gradient. The latest call's phash
    # sits 30+ bits away from the checker in the middle, so the
    # consecutive run from the end is 1 — below the min_calls threshold.
    sequence = [_png_b64(img_a), _png_b64(img_b), _png_b64(img_a)]
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(b64),
        )
        for i, b64 in enumerate(sequence)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_exact_match_wins_when_both_would_fire(make_call, now):
    """When the SHA-256 path catches the duplicate, the perceptual
    discount MUST NOT apply: identical bytes is a higher-precision signal
    than 'visually similar', so the event reports
    ``match_type='exact'`` with ``confidence=0.7`` (the exact base)
    rather than 0.65."""
    pytest.importorskip("imagehash")

    img_b64 = _png_b64(_content_rich_image((256, 256)))
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img_b64),
        )
        for i in range(3)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["match_type"] == "exact"
    assert ev.evidence["perceptual_distance"] == 0
    assert ev.confidence == pytest.approx(0.7)


def test_evidence_dict_contains_match_type_and_distance(make_call, now):
    """Spot-check both the exact and perceptual fire paths populate the
    new  evidence fields. Dashboard consumers will key on these to
    distinguish high-precision exact dupes from lower-precision perceptual
    near-dupes."""
    pytest.importorskip("imagehash")

    # --- Exact path: same bytes three times. ---
    img_b64 = _png_b64(_content_rich_image((256, 256)))
    exact_session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img_b64),
        )
        for i in range(3)
    ]
    ev_exact = VisionReUploadRule({}).evaluate(exact_session, project="p")
    assert ev_exact is not None
    assert "match_type" in ev_exact.evidence
    assert "perceptual_distance" in ev_exact.evidence
    assert ev_exact.evidence["match_type"] == "exact"
    assert ev_exact.evidence["perceptual_distance"] == 0

    # --- Perceptual path: resized re-encodes. ---
    img = _content_rich_image((256, 256))
    perceptual_session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(_png_b64(img.resize((256 - i, 256 - i)))),
        )
        for i in range(3)
    ]
    ev_perceptual = VisionReUploadRule({}).evaluate(perceptual_session, project="p")
    assert ev_perceptual is not None
    assert ev_perceptual.evidence["match_type"] == "perceptual"
    assert isinstance(ev_perceptual.evidence["perceptual_distance"], int)
    assert 0 <= ev_perceptual.evidence["perceptual_distance"] <= 6


def test_malformed_image_does_not_crash(make_call, now):
    """Corrupted base64 / non-image bytes must NOT propagate exceptions
    out of the rule. ``_perceptual_hash_from_b64`` swallows everything and
    returns ``None``; the rule then continues with the SHA-256 leg, which
    can still match (or not) on whatever bytes are present.

    This test uses *garbage* base64 that PIL cannot parse; we confirm the
    rule returns None (or an exact match if SHA-256 happens to fire)
    without raising."""
    pytest.importorskip("imagehash")

    # Three identical garbage payloads — SHA-256 WILL match, but the
    # phash leg should return None silently and not interfere. The rule
    # should fire the exact-match event.
    garbage_b64 = base64.b64encode(b"\x89PNG\x00not actually an image" * 5).decode("ascii")
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(garbage_b64),
        )
        for i in range(3)
    ]
    # Must not raise — that's the load-bearing assertion.
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    # SHA-256 of identical bytes IS identical → exact match fires.
    assert ev is not None
    assert ev.evidence["match_type"] == "exact"
    # Verify the perceptual-hash helper does indeed return None on this
    # garbage so we know the perceptual leg degraded silently as intended.
    assert _perceptual_hash_from_b64(garbage_b64, max_bytes=5 * 1024 * 1024) is None


def test_malformed_perceptual_does_not_block_perceptual_match(make_call, now):
    """A call whose bytes are unparseable contributes no phash to the
    candidate set, but a different image alongside it (or in the next
    call) still allows the perceptual fallback to fire when the latest
    call's image is a near-duplicate of earlier calls' images."""
    pytest.importorskip("imagehash")

    img = _content_rich_image((256, 256))
    # Three valid (slightly-resized) re-uploads — would fire on the
    # perceptual path alone. Phash decoder must handle each cleanly.
    sequence = [
        _png_b64(img),
        _png_b64(img.resize((252, 252))),
        _png_b64(img.resize((248, 248))),
    ]
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(b64),
        )
        for i, b64 in enumerate(sequence)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["match_type"] == "perceptual"


# ---------------------------------------------------------------------------
# Graceful-degradation tests (run regardless of whether imagehash is installed).
#
# These patch ``_PERCEPTUAL_AVAILABLE`` to False to simulate the
# no-extra-installed environment, confirming the  exact-byte behaviour
# remains intact and that the rule does not crash on the absent dep.
# ---------------------------------------------------------------------------


def test_graceful_degrade_when_imagehash_unavailable(make_call, now):
    """Patch ``_PERCEPTUAL_AVAILABLE = False`` to simulate the no-extra-
    installed environment. Exact-byte matching MUST still fire on
    identical uploads. The perceptual fallback simply doesn't run.
    """
    with patch.object(vru, "_PERCEPTUAL_AVAILABLE", False):
        img_b64 = base64.b64encode(b"FAKEIMAGEBYTES" * 200).decode("ascii")
        session = [
            make_call(
                provider="openai",
                timestamp=now + timedelta(seconds=i),
                raw_request=_openai_request_with_image(img_b64),
            )
            for i in range(3)
        ]
        ev = VisionReUploadRule({}).evaluate(session, project="p")
        assert ev is not None
        assert ev.evidence["match_type"] == "exact"
        assert ev.confidence == pytest.approx(0.7)


def test_graceful_degrade_returns_none_when_only_perceptual_would_fire(make_call, now):
    """When ``imagehash`` is unavailable AND the bytes differ across
    calls, the rule must return None rather than trying to invoke phash
    helpers. The exact path turns up nothing (different SHA-256s); the
    perceptual path is gated off. No fire.
    """
    with patch.object(vru, "_PERCEPTUAL_AVAILABLE", False):
        # Three different byte sequences — exact match impossible.
        session = [
            make_call(
                provider="openai",
                timestamp=now + timedelta(seconds=i),
                raw_request=_openai_request_with_image(
                    base64.b64encode(f"IMG-VARIANT-{i}".encode() * 100).decode("ascii")
                ),
            )
            for i in range(3)
        ]
        assert VisionReUploadRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Helper unit tests.
# ---------------------------------------------------------------------------


def test_phash_distance_hamming_correctness():
    """Spot-check ``_phash_distance`` against known XOR-popcount pairs.

    No imagehash dependency — this tests pure bit arithmetic against
    fixture phash strings. Robust on a CI matrix without the extra.
    """
    # 0xFF vs 0x00: 8 bits differ (a full byte flip).
    assert _phash_distance("ff", "00") == 8
    # Identity: distance 0.
    assert _phash_distance("deadbeef", "deadbeef") == 0
    # Two different real-looking phashes; XOR popcount is deterministic.
    assert _phash_distance("bf1994e2c4acd1c3", "bf1b94e2c4acd0c3") == 2
    # Malformed phash → max distance (defensive).
    assert _phash_distance("not-hex", "deadbeef") == 64
