"""Model-aware USD burn estimates for CallRecords and policy budgets.

Approximate **list-price heuristics**, not invoices. Rates are USD per
1,000,000 tokens (input / output / cache-read). Unknown models fall back
to the historical flat average (``FALLBACK_USD_PER_TOKEN``) so behaviour
stays continuous for obscure or self-hosted model strings.

Optional ``tiktoken`` (``pip install token-sentinel[tiktoken]``) can
estimate token counts when the provider response omitted usage
(e.g. OpenAI streaming without ``stream_options.include_usage``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from token_sentinel.events import CallRecord

# Historical flat average used when the model is unknown. Kept as a named
# constant so rule-side burn helpers and the policy plane stay aligned.
FALLBACK_USD_PER_TOKEN = 9e-6


@dataclass(frozen=True, slots=True)
class ModelRate:
    """USD per 1M tokens for a model family.

    ``cache_read_per_mtok`` defaults to 10% of input when omitted (common
    industry shape for prompt-cache hits). Set explicitly when known.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None

    def cache_read(self) -> float:
        if self.cache_read_per_mtok is not None:
            return self.cache_read_per_mtok
        return self.input_per_mtok * 0.1


# Prefix table — longest matching prefix wins. Ordered most-specific first
# is not required (we scan for longest match) but keeps the table readable.
# Rates are directionally correct mid-2026 list prices for FinOps *signals*;
# refresh opportunistically. Customers can override via Sentinel(pricing_table=).
_DEFAULT_RATES: dict[str, ModelRate] = {
    # Anthropic
    "claude-opus-4": ModelRate(15.0, 75.0, 1.5),
    "claude-opus": ModelRate(15.0, 75.0, 1.5),
    "claude-sonnet-4": ModelRate(3.0, 15.0, 0.3),
    "claude-sonnet": ModelRate(3.0, 15.0, 0.3),
    "claude-3-5-sonnet": ModelRate(3.0, 15.0, 0.3),
    "claude-3-7-sonnet": ModelRate(3.0, 15.0, 0.3),
    "claude-haiku-4": ModelRate(0.80, 4.0, 0.08),
    "claude-haiku": ModelRate(0.80, 4.0, 0.08),
    "claude-3-5-haiku": ModelRate(0.80, 4.0, 0.08),
    "claude-3-haiku": ModelRate(0.25, 1.25, 0.03),
    # OpenAI
    "gpt-5-mini": ModelRate(0.25, 2.0, 0.025),
    "gpt-5-nano": ModelRate(0.05, 0.40, 0.005),
    "gpt-5": ModelRate(1.25, 10.0, 0.125),
    "gpt-4.1-mini": ModelRate(0.40, 1.60, 0.04),
    "gpt-4.1-nano": ModelRate(0.10, 0.40, 0.01),
    "gpt-4.1": ModelRate(2.0, 8.0, 0.20),
    "gpt-4o-mini": ModelRate(0.15, 0.60, 0.075),
    "gpt-4o": ModelRate(2.50, 10.0, 1.25),
    "gpt-4-turbo": ModelRate(10.0, 30.0, 5.0),
    "gpt-4": ModelRate(30.0, 60.0, 15.0),
    "o3-mini": ModelRate(1.10, 4.40, 0.55),
    "o3": ModelRate(10.0, 40.0, 2.5),
    "o1-mini": ModelRate(1.10, 4.40, 0.55),
    "o1": ModelRate(15.0, 60.0, 7.5),
    # Google
    "gemini-2.5-pro": ModelRate(1.25, 10.0, 0.125),
    "gemini-2.5-flash": ModelRate(0.15, 0.60, 0.015),
    "gemini-2.0-flash": ModelRate(0.10, 0.40, 0.01),
    "gemini-2.0-pro": ModelRate(1.25, 5.0, 0.125),
    "gemini-1.5-pro": ModelRate(1.25, 5.0, 0.125),
    "gemini-1.5-flash": ModelRate(0.075, 0.30, 0.0075),
    # DeepSeek
    "deepseek-reasoner": ModelRate(0.55, 2.19, 0.14),
    "deepseek-chat": ModelRate(0.27, 1.10, 0.07),
    # Cohere
    "command-r-plus": ModelRate(2.50, 10.0, 0.25),
    "command-a": ModelRate(2.50, 10.0, 0.25),
    "command-r": ModelRate(0.15, 0.60, 0.015),
    # Mistral
    "mistral-large": ModelRate(2.0, 6.0, 0.20),
    "mistral-small": ModelRate(0.10, 0.30, 0.01),
    "mistral-medium": ModelRate(0.40, 2.0, 0.04),
}


def normalize_model_name(model: str | None) -> str:
    """Lowercase + strip common gateway vendor prefixes for table lookup."""
    if not model:
        return ""
    name = model.strip().lower()
    # Strip ``vendor/`` and Portkey ``@alias/`` prefixes (same spirit as
    # model_misroute) so ``anthropic/claude-sonnet-4`` matches ``claude-sonnet``.
    if name.startswith("@"):
        slash = name.find("/")
        if slash != -1:
            name = name[slash + 1 :]
    for prefix in (
        "anthropic/",
        "openai/",
        "google/",
        "google-vertex/",
        "meta-llama/",
        "mistral/",
        "cohere/",
        "groq/",
        "deepseek/",
        "xai/",
        "perplexity/",
        "bedrock/",
        "amazon/",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    # Bedrock-style ``anthropic.claude-sonnet-4-…``
    if "." in name and name.split(".", 1)[0] in {
        "anthropic",
        "amazon",
        "cohere",
        "meta",
        "mistral",
    }:
        name = name.split(".", 1)[1]
    return name


def lookup_rate(
    model: str | None,
    pricing_table: Mapping[str, ModelRate] | None = None,
) -> ModelRate | None:
    """Return the best matching :class:`ModelRate`, or ``None`` if unknown."""
    name = normalize_model_name(model)
    if not name:
        return None
    table = pricing_table if pricing_table is not None else _DEFAULT_RATES
    # Exact hit first.
    if name in table:
        return table[name]
    # Longest prefix among table keys that are prefixes of *name*.
    best_key = ""
    best_rate: ModelRate | None = None
    for key, rate in table.items():
        if name.startswith(key) and len(key) > len(best_key):
            best_key = key
            best_rate = rate
    return best_rate


def estimate_usd(
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_read_tokens: int = 0,
    pricing_table: Mapping[str, ModelRate] | None = None,
) -> float:
    """Estimate USD cost for a single call.

    Cache-read tokens (prompt-cache hits) are billed at the cache-read rate
    and **subtracted** from the billable input total so they are not double-
    counted at the full input rate.
    """
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    cache_read = max(0, min(int(cache_read_tokens), prompt))

    rate = lookup_rate(model, pricing_table)
    if rate is None:
        return (prompt + completion) * FALLBACK_USD_PER_TOKEN

    billable_input = max(0, prompt - cache_read)
    usd = (
        billable_input * rate.input_per_mtok / 1_000_000.0
        + cache_read * rate.cache_read() / 1_000_000.0
        + completion * rate.output_per_mtok / 1_000_000.0
    )
    return usd


def cache_read_tokens_from_extra(usage_extra: Mapping[str, Any] | None) -> int:
    """Pull ``cache_read_tokens`` from a CallRecord.usage_extra dict."""
    if not usage_extra:
        return 0
    raw = usage_extra.get("cache_read_tokens", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def estimate_call_usd(
    call: CallRecord,
    pricing_table: Mapping[str, ModelRate] | None = None,
) -> float:
    """Estimate USD for a :class:`~token_sentinel.events.CallRecord`."""
    return estimate_usd(
        call.model,
        call.prompt_tokens,
        call.completion_tokens,
        cache_read_tokens=cache_read_tokens_from_extra(call.usage_extra),
        pricing_table=pricing_table,
    )


def extract_openai_cache_read(usage: Any) -> int:
    """Read OpenAI ``prompt_tokens_details.cached_tokens`` from a usage object."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        raw = details.get("cached_tokens", 0)
    else:
        raw = getattr(details, "cached_tokens", 0)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def extract_anthropic_cache_read(usage: Any) -> int:
    """Read Anthropic ``cache_read_input_tokens`` from a usage object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        raw = usage.get("cache_read_input_tokens", 0)
    else:
        raw = getattr(usage, "cache_read_input_tokens", 0)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def try_estimate_tokens_from_text(
    text: str,
    *,
    model: str | None = None,
) -> int | None:
    """Best-effort token count via optional ``tiktoken``; ``None`` if unavailable."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None

    enc = None
    name = normalize_model_name(model)
    if name:
        try:
            enc = tiktoken.encoding_for_model(name)
        except Exception:
            enc = None
    if enc is None:
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None
    try:
        return len(enc.encode(text))
    except Exception:
        return None


def try_estimate_tokens_from_messages(
    messages: Any,
    *,
    model: str | None = None,
) -> int | None:
    """Estimate prompt tokens for an OpenAI/Anthropic-style messages list.

    Concatenates role + content strings. Ignores tool schemas (under-count
    vs true billing for tool-heavy agents). Returns ``None`` when tiktoken
    is not installed.
    """
    if not messages:
        return 0
    parts: list[str] = []
    try:
        for msg in messages:
            if isinstance(msg, dict):
                role = str(msg.get("role", ""))
                content = msg.get("content", "")
            else:
                role = str(getattr(msg, "role", "") or "")
                content = getattr(msg, "content", "")
            if isinstance(content, list):
                # Multimodal content blocks — keep text pieces only.
                texts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(str(block.get("text", "")))
                    elif isinstance(block, str):
                        texts.append(block)
                    else:
                        t = getattr(block, "text", None)
                        if t is not None:
                            texts.append(str(t))
                content_s = " ".join(texts)
            else:
                content_s = str(content or "")
            parts.append(f"{role}: {content_s}")
    except Exception:
        return None
    return try_estimate_tokens_from_text("\n".join(parts), model=model)


def maybe_fill_missing_tokens(call: CallRecord) -> None:
    """If both token counts are 0, try tiktoken against raw_request messages.

    Mutates ``call`` in place and stamps
    ``usage_extra["tokens_estimated"] = True`` when successful. No-op when
    tiktoken is missing or messages are empty.
    """
    if call.prompt_tokens or call.completion_tokens:
        return
    messages = None
    if isinstance(call.raw_request, dict):
        messages = call.raw_request.get("messages")
    if not messages:
        return
    estimated = try_estimate_tokens_from_messages(messages, model=call.model)
    if estimated is None or estimated <= 0:
        return
    call.prompt_tokens = estimated
    # Completions unknown without the response text; leave at 0.
    extra = dict(call.usage_extra) if call.usage_extra else {}
    extra["tokens_estimated"] = True
    extra["tokens_estimated_source"] = "tiktoken"
    call.usage_extra = extra


# Public table copy for customers who want to extend defaults.
def default_pricing_table() -> dict[str, ModelRate]:
    """Return a shallow copy of the built-in model rate table."""
    return dict(_DEFAULT_RATES)
