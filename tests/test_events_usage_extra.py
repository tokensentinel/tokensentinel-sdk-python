"""Tests for :attr:`CallRecord.usage_extra` .

The  schema convention is:

    usage_extra: {
        "dimension_kind": "per_image" | "per_pixel" | "per_second"
                          | "per_character" | "per_token",
        "dimension_value": float,
        "model_specific_meta": {...},  # optional, free-form
    }

These tests pin the contract structurally (no enum enforcement at the
SDK boundary — the cloud's :data:`NON_TOKEN_PRICES` dispatch is the
single source of truth for which dimension_kind values map to a price)
and verify the additive nature of the field: pre- callers that
omit usage_extra get an empty dict and everything else still works.
"""

from __future__ import annotations

from datetime import datetime, timezone

from token_sentinel.events import CallRecord

# ---------------------------------------------------------------------------
# 1. Default value: empty dict for token-priced providers
# ---------------------------------------------------------------------------


def test_usage_extra_defaults_to_empty_dict() -> None:
    """A CallRecord built without usage_extra has an empty dict.

    Empty dict (not None) so callers can safely do ``record.usage_extra.get(...)``
    without an isinstance check. This matches the field's
    ``field(default_factory=dict)`` declaration.
    """
    rec = CallRecord(
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        method="messages.create",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=120.0,
        request_hash="abc",
    )
    assert rec.usage_extra == {}
    # New CallRecord instances must NOT share the same dict (mutable
    # default factory regression test).
    rec2 = CallRecord(
        session_id="s2",
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        method="messages.create",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=120.0,
        request_hash="abc",
    )
    assert rec.usage_extra is not rec2.usage_extra


# ---------------------------------------------------------------------------
# 2. Populated per_image shape (Replicate Flux et al.)
# ---------------------------------------------------------------------------


def test_usage_extra_per_image_shape() -> None:
    """The per_image dimension is structurally a dict with the documented keys."""
    extra = {
        "dimension_kind": "per_image",
        "dimension_value": 4.0,
        "model_specific_meta": {"source": "prediction_output_count"},
    }
    rec = CallRecord(
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        provider="replicate",
        model="black-forest-labs/flux-schnell",
        method="run",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=2400.0,
        request_hash="h",
        usage_extra=extra,
    )
    assert rec.usage_extra["dimension_kind"] == "per_image"
    assert rec.usage_extra["dimension_value"] == 4.0
    assert rec.usage_extra["model_specific_meta"]["source"] == "prediction_output_count"


# ---------------------------------------------------------------------------
# 3. Populated per_second shape (Replicate video)
# ---------------------------------------------------------------------------


def test_usage_extra_per_second_shape() -> None:
    extra = {
        "dimension_kind": "per_second",
        "dimension_value": 12.5,
        "model_specific_meta": {"source": "metrics.predict_time"},
    }
    rec = CallRecord(
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        provider="replicate",
        model="tencent/hunyuan-video",
        method="predictions.get",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=45000.0,
        request_hash="h",
        usage_extra=extra,
    )
    assert rec.usage_extra["dimension_kind"] == "per_second"
    assert rec.usage_extra["dimension_value"] == 12.5


# ---------------------------------------------------------------------------
# 4. No enum enforcement — arbitrary dimension_kind round-trips
# ---------------------------------------------------------------------------


def test_usage_extra_no_enum_enforcement() -> None:
    """may add new dimension_kind values (e.g. ``"per_megapixel"``).

    The SDK boundary is intentionally permissive: we store whatever the
    wrapper passed and let the cloud's ``NON_TOKEN_PRICES`` dispatch
    decide whether the kind is recognised. Crash-on-unknown-enum would
    require an SDK release every time the cloud added a price dimension,
    which defeats the point of the abstraction.
    """
    extra = {
        "dimension_kind": "per_future_dimension_we_havent_invented",
        "dimension_value": 1.0,
    }
    rec = CallRecord(
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        provider="future-provider",
        model="future-model",
        method="run",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=10.0,
        request_hash="h",
        usage_extra=extra,
    )
    assert rec.usage_extra["dimension_kind"] == "per_future_dimension_we_havent_invented"


# ---------------------------------------------------------------------------
# 5. Token-priced records keep usage_extra empty by default
# ---------------------------------------------------------------------------


def test_token_priced_records_leave_usage_extra_empty() -> None:
    """Existing wrappers (Anthropic/OpenAI/Gemini/Bedrock) don't set
    usage_extra; the field default must be ``{}`` so older code paths
    don't need a migration.
    """
    rec = CallRecord(
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model="gpt-5",
        method="chat.completions.create",
        prompt_tokens=1500,
        completion_tokens=300,
        latency_ms=850.0,
        request_hash="h",
    )
    # Token-priced fields populated, usage_extra empty.
    assert rec.prompt_tokens == 1500
    assert rec.completion_tokens == 300
    assert rec.usage_extra == {}
    # Field IS present (i.e. not a missing attribute) so consumers can
    # do ``record.usage_extra`` without a hasattr() guard.
    assert hasattr(rec, "usage_extra")
