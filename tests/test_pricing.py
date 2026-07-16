"""Tests for model-aware pricing and optional tiktoken fill (1.0.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from token_sentinel.events import CallRecord
from token_sentinel.pricing import (
    FALLBACK_USD_PER_TOKEN,
    ModelRate,
    default_pricing_table,
    estimate_call_usd,
    estimate_usd,
    extract_anthropic_cache_read,
    extract_openai_cache_read,
    lookup_rate,
    maybe_fill_missing_tokens,
    normalize_model_name,
)
from token_sentinel.sentinel import Sentinel, _estimate_call_burn_usd


def _call(
    *,
    model: str = "gpt-4o",
    prompt: int = 1000,
    completion: int = 500,
    usage_extra: dict | None = None,
    messages: list | None = None,
) -> CallRecord:
    return CallRecord(
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model=model,
        method="chat.completions.create",
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=10.0,
        request_hash="h",
        raw_request={"messages": messages or []},
        usage_extra=usage_extra or {},
    )


def test_haiku_cheaper_than_opus_same_tokens() -> None:
    haiku = estimate_usd("claude-haiku-4-5", 100_000, 10_000)
    opus = estimate_usd("claude-opus-4", 100_000, 10_000)
    assert haiku < opus
    assert haiku > 0
    assert opus > 0


def test_gpt4o_mini_cheaper_than_gpt4o() -> None:
    mini = estimate_usd("gpt-4o-mini", 50_000, 5_000)
    full = estimate_usd("gpt-4o", 50_000, 5_000)
    assert mini < full


def test_unknown_model_falls_back_to_flat_rate() -> None:
    usd = estimate_usd("totally-unknown-model-xyz", 1000, 0)
    assert abs(usd - 1000 * FALLBACK_USD_PER_TOKEN) < 1e-12


def test_gateway_prefix_normalizes() -> None:
    assert normalize_model_name("anthropic/claude-sonnet-4-20250514").startswith("claude-sonnet")
    rate = lookup_rate("openai/gpt-4o-mini")
    assert rate is not None
    assert rate.input_per_mtok < lookup_rate("gpt-4o").input_per_mtok  # type: ignore[union-attr]


def test_cache_read_reduces_burn() -> None:
    full = estimate_usd("gpt-4o", 10_000, 0, cache_read_tokens=0)
    cached = estimate_usd("gpt-4o", 10_000, 0, cache_read_tokens=9_000)
    assert cached < full


def test_estimate_call_usd_reads_usage_extra() -> None:
    c = _call(
        model="gpt-4o",
        prompt=10_000,
        completion=0,
        usage_extra={"cache_read_tokens": 8_000},
    )
    with_cache = estimate_call_usd(c)
    without = estimate_usd("gpt-4o", 10_000, 0, cache_read_tokens=0)
    assert with_cache < without


def test_custom_pricing_table_override() -> None:
    table = {"my-local-model": ModelRate(1.0, 2.0)}
    usd = estimate_usd("my-local-model", 1_000_000, 1_000_000, pricing_table=table)
    assert abs(usd - 3.0) < 1e-9


def test_default_pricing_table_is_copy() -> None:
    a = default_pricing_table()
    b = default_pricing_table()
    a["zzz"] = ModelRate(1.0, 1.0)
    assert "zzz" not in b


def test_extract_openai_cache_read() -> None:
    usage = SimpleNamespace(
        prompt_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    assert extract_openai_cache_read(usage) == 40
    assert extract_openai_cache_read(None) == 0
    assert extract_openai_cache_read({"prompt_tokens_details": {"cached_tokens": 12}}) == 12


def test_extract_anthropic_cache_read() -> None:
    usage = SimpleNamespace(input_tokens=100, cache_read_input_tokens=55)
    assert extract_anthropic_cache_read(usage) == 55
    assert extract_anthropic_cache_read({"cache_read_input_tokens": 7}) == 7


def test_sentinel_burn_uses_model_rates() -> None:
    cheap = _call(model="claude-haiku-4-5", prompt=20_000, completion=2_000)
    expensive = _call(model="claude-opus-4", prompt=20_000, completion=2_000)
    assert _estimate_call_burn_usd(cheap) < _estimate_call_burn_usd(expensive)


def test_maybe_fill_missing_tokens_without_tiktoken() -> None:
    c = _call(model="gpt-4o", prompt=0, completion=0, messages=[{"role": "user", "content": "hi"}])
    # Soft no-op when estimator returns None (tiktoken missing / failure).
    with patch(
        "token_sentinel.pricing.try_estimate_tokens_from_messages",
        return_value=None,
    ):
        maybe_fill_missing_tokens(c)
    assert c.prompt_tokens == 0


def test_maybe_fill_missing_tokens_with_stub() -> None:
    c = _call(
        model="gpt-4o",
        prompt=0,
        completion=0,
        messages=[{"role": "user", "content": "hello world " * 50}],
    )
    with patch(
        "token_sentinel.pricing.try_estimate_tokens_from_messages",
        return_value=42,
    ):
        maybe_fill_missing_tokens(c)
    assert c.prompt_tokens == 42
    assert c.usage_extra.get("tokens_estimated") is True


def test_record_call_fills_tokens_when_zero() -> None:
    sentinel = Sentinel(project="p", mode="log")
    c = _call(
        model="gpt-4o",
        prompt=0,
        completion=0,
        messages=[{"role": "user", "content": "x" * 200}],
    )
    with patch(
        "token_sentinel.sentinel.maybe_fill_missing_tokens",
        side_effect=lambda call: (
            setattr(call, "prompt_tokens", 99)
            or call.usage_extra.update({"tokens_estimated": True})
        ),
    ):
        sentinel.record_call(c)
    assert c.prompt_tokens == 99
