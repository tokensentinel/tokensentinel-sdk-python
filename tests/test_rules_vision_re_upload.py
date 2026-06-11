"""Tests for ``VisionReUploadRule``.

Defaults: min_calls=3, window_seconds=60, max_image_bytes=5MB.

The image-byte extraction handles three provider request shapes:

  - OpenAI: ``messages[].content[].image_url.url`` (``data:image/...;base64,...``)
  - Anthropic: ``messages[].content[].source.data``
  - Gemini: ``contents[].parts[].inline_data.data``
                  OR ``contents[].inline_data.data``

Each test pins the exact shape so a wrapper change is caught immediately.
"""

from __future__ import annotations

import base64
from datetime import timedelta

import pytest

from token_sentinel.rules.vision_re_upload import (
    VisionReUploadRule,
    _extract_image_hashes,
)

# ---------------------------------------------------------------------------
# Helpers — construct provider-specific image-bearing raw_request shapes.
# ---------------------------------------------------------------------------


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _openai_request_with_image(image_b64: str, *, detail: str = "auto") -> dict:
    """OpenAI chat-completion ``raw_request`` with one image attachment."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": detail,
                        },
                    },
                ],
            }
        ],
        "tools": [],
        "max_tokens": 0,
    }


def _anthropic_request_with_image(image_b64: str) -> dict:
    """Anthropic ``raw_request`` with one image source block."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this screenshot?"},
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


def _gemini_request_with_image(image_b64: str) -> dict:
    """Gemini ``raw_request`` with one inline_data part."""
    return {
        "model": "gemini-2.5-pro",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this."},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                ],
            }
        ],
        "tools": [],
    }


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert VisionReUploadRule({}).evaluate([], project="p") is None


def test_below_min_calls_no_fire(make_call, now):
    """Two consecutive identical uploads — below default min_calls=3."""
    img = _b64(b"PNGBYTES" * 100)
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img),
        )
        for i in range(2)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_three_calls_different_images_no_fire(make_call, now):
    """Three calls, three distinct images — no duplication signal."""
    session = [
        make_call(
            provider="anthropic",
            timestamp=now + timedelta(seconds=i),
            raw_request=_anthropic_request_with_image(_b64(f"IMG-{i}".encode() * 50)),
        )
        for i in range(3)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_text_only_calls_no_fire(make_call, now):
    """Text-only calls (no image_url / source / inline_data) → no fire."""
    session = [
        make_call(
            provider="anthropic",
            timestamp=now + timedelta(seconds=i),
            raw_request={
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
                "max_tokens": 0,
            },
        )
        for i in range(5)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_remote_http_image_url_not_hashed(make_call, now):
    """``http://`` URLs cannot be hashed client-side — three remote-URL
    calls don't fire even when the URL repeats."""
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            }
                        ],
                    }
                ],
                "tools": [],
                "max_tokens": 0,
            },
        )
        for i in range(3)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_dupes_outside_window_no_fire(make_call, now):
    """Three identical uploads spaced 100s apart — only the latest is in
    the default 60s window so the consecutive run is just 1."""
    img = _b64(b"PNGBYTES" * 100)
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i * 100),
            raw_request=_openai_request_with_image(img),
        )
        for i in range(3)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases (fires)
# ---------------------------------------------------------------------------


def test_three_consecutive_openai_uploads_fire(make_call, now):
    img = _b64(b"OPENAIIMG" * 80)
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i * 5),
            raw_request=_openai_request_with_image(img),
        )
        for i in range(3)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "vision_re_upload"
    assert ev.rule == "v0.vision_re_upload"
    assert ev.evidence["duplicate_count"] == 3
    assert ev.evidence["provider"] == "openai"
    # Base confidence at the threshold.
    assert ev.confidence == 0.7
    assert ev.suggested_action == "cache_image_locally_or_reuse_attachment_id"


def test_three_consecutive_anthropic_uploads_fire(make_call, now):
    img = _b64(b"ANTHROPICIMG" * 80)
    session = [
        make_call(
            provider="anthropic",
            timestamp=now + timedelta(seconds=i),
            raw_request=_anthropic_request_with_image(img),
        )
        for i in range(3)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["provider"] == "anthropic"


def test_three_consecutive_gemini_uploads_fire(make_call, now):
    img = _b64(b"GEMINIIMG" * 80)
    session = [
        make_call(
            provider="gemini",
            timestamp=now + timedelta(seconds=i),
            raw_request=_gemini_request_with_image(img),
        )
        for i in range(3)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["provider"] == "gemini"


def test_confidence_scales_with_duplicate_count(make_call, now):
    """4 dupes → 0.8 confidence; 5 → 0.9; 6+ → 0.99 (clamped)."""
    img = _b64(b"X" * 100)
    for count, expected in [(4, 0.8), (5, 0.9), (6, 0.99), (10, 0.99)]:
        session = [
            make_call(
                provider="openai",
                timestamp=now + timedelta(seconds=i),
                raw_request=_openai_request_with_image(img),
            )
            for i in range(count)
        ]
        ev = VisionReUploadRule({}).evaluate(session, project="p")
        assert ev is not None, f"count={count} should fire"
        assert ev.confidence == pytest.approx(expected), (
            f"count={count}, got {ev.confidence}, expected {expected}"
        )


def test_consecutive_run_breaks_on_different_image(make_call, now):
    """A, B, A, A, A → consecutive run from the end is 3 → fires."""
    img_a = _b64(b"AAAA" * 80)
    img_b = _b64(b"BBBB" * 80)
    sequence = [img_a, img_b, img_a, img_a, img_a]
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img),
        )
        for i, img in enumerate(sequence)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    # 3 consecutive from the latest end.
    assert ev.evidence["duplicate_count"] == 3


def test_consecutive_run_breaks_no_fire(make_call, now):
    """A, A, B → latest call has B which doesn't match earlier → no fire."""
    img_a = _b64(b"AAAA" * 80)
    img_b = _b64(b"BBBB" * 80)
    sequence = [img_a, img_a, img_b]
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img),
        )
        for i, img in enumerate(sequence)
    ]
    assert VisionReUploadRule({}).evaluate(session, project="p") is None


def test_event_fields(make_call, now):
    """Spot-check the LeakEvent shape so dashboard consumers stay happy."""
    img = _b64(b"X" * 200)
    session = [
        make_call(
            provider="anthropic",
            timestamp=now + timedelta(seconds=i),
            raw_request=_anthropic_request_with_image(img),
        )
        for i in range(3)
    ]
    ev = VisionReUploadRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.project == "p"
    # 16-hex-char prefix mirrors embedding_waste.
    assert len(ev.evidence["image_hash"]) == 16
    assert ev.evidence["window_seconds"] == 60
    assert ev.estimated_burn >= 0  # 2 extra dupes × 5e-3 = 0.01


def test_custom_min_calls(make_call, now):
    """``vision_re_upload.min_calls=2`` fires on 2 consecutive uploads."""
    img = _b64(b"X" * 100)
    session = [
        make_call(
            provider="openai",
            timestamp=now + timedelta(seconds=i),
            raw_request=_openai_request_with_image(img),
        )
        for i in range(2)
    ]
    ev = VisionReUploadRule({"vision_re_upload.min_calls": 2}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["duplicate_count"] == 2


# ---------------------------------------------------------------------------
# Image-byte extraction unit tests
# ---------------------------------------------------------------------------


def test_extract_handles_openai_data_url(make_call, now):
    """The OpenAI ``data:image/...;base64,...`` shape extracts cleanly."""
    img = _b64(b"OPENAI" * 50)
    call = make_call(
        provider="openai",
        raw_request=_openai_request_with_image(img),
    )
    hashes = _extract_image_hashes(call, max_bytes=5 * 1024 * 1024)
    assert len(hashes) == 1


def test_extract_handles_anthropic_source(make_call):
    """The Anthropic ``content[].source.data`` shape extracts cleanly."""
    img = _b64(b"ANTHROPIC" * 50)
    call = make_call(
        provider="anthropic",
        raw_request=_anthropic_request_with_image(img),
    )
    hashes = _extract_image_hashes(call, max_bytes=5 * 1024 * 1024)
    assert len(hashes) == 1


def test_extract_handles_gemini_inline_data(make_call):
    """The Gemini ``parts[].inline_data.data`` shape extracts cleanly."""
    img = _b64(b"GEMINI" * 50)
    call = make_call(
        provider="gemini",
        raw_request=_gemini_request_with_image(img),
    )
    hashes = _extract_image_hashes(call, max_bytes=5 * 1024 * 1024)
    assert len(hashes) == 1


def test_extract_handles_top_level_gemini_dict(make_call):
    """``contents = {"inline_data": {...}}`` (dict, not list) also works."""
    img = _b64(b"GEMINI-FLAT" * 30)
    call = make_call(
        provider="gemini",
        raw_request={
            "model": "gemini-2.5-pro",
            "contents": {
                "inline_data": {"mime_type": "image/png", "data": img},
            },
        },
    )
    hashes = _extract_image_hashes(call, max_bytes=5 * 1024 * 1024)
    assert len(hashes) == 1


def test_extract_multiple_images_in_single_call(make_call):
    """Multiple images in one Anthropic content block → multiple hashes."""
    img_a = _b64(b"AAAA" * 30)
    img_b = _b64(b"BBBB" * 30)
    call = make_call(
        provider="anthropic",
        raw_request={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": img_a},
                        },
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": img_b},
                        },
                    ],
                }
            ]
        },
    )
    hashes = _extract_image_hashes(call, max_bytes=5 * 1024 * 1024)
    assert len(hashes) == 2


def test_extract_returns_empty_on_unknown_shape(make_call):
    """Non-dict ``raw_request`` → empty set, no crash."""
    call = make_call(raw_request={})
    assert _extract_image_hashes(call, max_bytes=5 * 1024 * 1024) == set()


def test_extract_truncates_oversized_images(make_call):
    """Two images that share a 100-byte prefix produce the SAME hash under
    a 100-byte cap (the trailing bytes are truncated before hashing)."""
    shared_prefix = "AAAAAAAAAA" * 10  # 100 chars
    img_a = shared_prefix + "DIFFERENT_A"
    img_b = shared_prefix + "DIFFERENT_B"
    call_a = make_call(
        provider="anthropic",
        raw_request={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"data": img_a}},
                    ],
                }
            ]
        },
    )
    call_b = make_call(
        provider="anthropic",
        raw_request={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"data": img_b}},
                    ],
                }
            ]
        },
    )
    h_a = _extract_image_hashes(call_a, max_bytes=100).pop()
    h_b = _extract_image_hashes(call_b, max_bytes=100).pop()
    # With a 100-byte cap, the trailing differences are stripped.
    assert h_a == h_b
