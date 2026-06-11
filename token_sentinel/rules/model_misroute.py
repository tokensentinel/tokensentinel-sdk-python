"""Rule: classification-shaped prompts sent to frontier models.

Fires when a small, classify-shaped prompt is routed to a frontier-class model
that is meaningfully more expensive than a small/fast alternative. Updated for
the May 2026 model lineup — covers the major closed and open frontiers:

- Anthropic: ``claude-opus-*``, ``claude-sonnet-*`` (incl. ``claude-3-5-sonnet``,
  ``claude-4-sonnet``)
- OpenAI: ``gpt-5-*`` (excluding ``gpt-5-mini`` / ``gpt-5-nano``), ``gpt-4-turbo``,
  ``gpt-4o`` (excluding ``gpt-4o-mini``)
- Google: ``gemini-2.5-pro``, ``gemini-2.0-pro``
- DeepSeek: ``deepseek-chat`` (V3.x), ``deepseek-reasoner`` (R1)
- Cohere: ``command-r-plus``, ``command-a``
- Mistral: ``mistral-large-*``

Models in ``EXCLUDE_PREFIXES`` (the ``*-mini`` / ``*-nano`` / ``*-flash`` tiers)
are treated as already-cheap and never fire — even if they share a frontier
prefix (e.g., ``gpt-5-mini`` shares the ``gpt-5`` prefix). The match logic is:
*any frontier prefix matches AND no exclude prefix matches*.

The ``CHEAP_ALTERNATIVES`` dict drives the ``suggested_action`` field on the
emitted ``LeakEvent`` so handlers can show the customer exactly which cheaper
model to route to.

Keyword matching:

Classification keywords are matched at **word boundaries** rather than as
substrings of the lowercased prompt. The substring form was too loose — e.g.,
``"is this"`` (formerly in the keyword list) fired on perfectly innocuous
prompts like ``"is this code correct"`` or ``"is this what you want"``. Word
boundaries plus a tighter keyword list (``"is this a "``/``"is this an "``
instead of bare ``"is this"``) eliminate the most common false-positive shape
while still catching the genuine classification cues. Regex is precompiled at
module load to keep the hot-path cost flat.

Gateway model-name normalisation:

Customers route through OpenAI-compatible gateways (OpenRouter, Portkey,
Helicone, Together) that ship model names with provider prefixes:

  - OpenRouter: ``anthropic/claude-3.7-sonnet``, ``openai/gpt-4o-mini``
  - Together: ``meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo``
  - Portkey virtual keys: ``@openai-prod/gpt-4o``

Previously, ``model.startswith("gpt-4o")`` returned False on
``"openai/gpt-4o"`` and the misroute rule silently never fired for gateway
traffic. :func:`_normalize_model_name` strips the known vendor prefixes
(plus the Portkey ``@env/`` virtual-key shape) so the existing allowlist /
exclude logic sees the canonical bare name.

**Preserve-bare-name policy.** The generic ``<vendor>/<model>`` fallback
only triggers when the call has a ``base_url`` set — i.e., we have positive
evidence the call went through a gateway. Bare ``model.startswith`` strings
that happen to contain ``/`` (a customer-named fine-tune, a model alias
with a slash) are left UNCHANGED in the direct-provider case. The known
vendor prefixes (``anthropic/``, ``openai/``, etc.) always strip — they
collide with no plausible direct-provider model name today — but the
catch-all only applies when we know we are talking to a gateway.
"""

from __future__ import annotations

import re
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

FRONTIER_PREFIXES = (
    # Anthropic
    "claude-opus",
    "claude-sonnet",
    "claude-3-5-sonnet",
    "claude-4-sonnet",
    # OpenAI
    "gpt-5",
    "gpt-4-turbo",
    "gpt-4o",
    # Google
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    # Cohere
    "command-r-plus",
    "command-a",
    # Mistral
    "mistral-large",
)

# Models that share a frontier prefix but are themselves the cheap tier and
# should NOT fire the rule. e.g., ``gpt-5-mini`` matches the ``gpt-5`` prefix
# but is the recommended cheap alternative — flagging it would be a loop.
EXCLUDE_PREFIXES = (
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o-mini",
)

# Maps each frontier model family to its recommended cheap alternative. The
# longest key wins for routing — keep the dict ordered most-specific first so
# a linear scan returns the right answer (claude-opus before claude-sonnet,
# deepseek-reasoner before deepseek-chat, etc.).
CHEAP_ALTERNATIVES: dict[str, str] = {
    # Anthropic — Haiku 4.5 is the small Claude in May 2026
    "claude-opus": "claude-haiku-4-5",
    "claude-sonnet": "claude-haiku-4-5",
    "claude-3-5-sonnet": "claude-haiku-4-5",
    "claude-4-sonnet": "claude-haiku-4-5",
    # OpenAI
    "gpt-5": "gpt-5-mini",
    "gpt-4-turbo": "gpt-4o-mini",
    "gpt-4o": "gpt-4o-mini",
    # Google — Flash is the small Gemini
    "gemini-2.5-pro": "gemini-2.5-flash",
    "gemini-2.0-pro": "gemini-2.5-flash",
    # DeepSeek — chat (V3) is materially cheaper than reasoner (R1)
    "deepseek-reasoner": "deepseek-chat",
    "deepseek-chat": "deepseek-chat",
    # Cohere — command-r is the smaller variant
    "command-r-plus": "command-r",
    "command-a": "command-r",
    # Mistral
    "mistral-large": "mistral-small-latest",
}

# Known vendor prefixes that gateways prepend to model names. These are
# ALWAYS stripped because no direct-provider model name today actually
# starts with ``anthropic/`` or ``openai/`` — they uniquely identify
# gateway traffic. Order matters only for documentation: the lookup is
# done with a single ``startswith`` check against the tuple.
_VENDOR_PREFIXES: tuple[str, ...] = (
    "anthropic/",
    "openai/",
    "google/",
    "meta-llama/",
    "mistral/",
    "cohere/",
    "groq/",
    "deepseek/",
    "xai/",
    "perplexity/",
)

# Portkey "virtual keys" are an environment alias the user creates on the
# Portkey dashboard, e.g. ``@openai-prod``. The model string then reads
# ``@openai-prod/gpt-4o``. The leading ``@`` distinguishes them from a
# vendor prefix, so we match with a dedicated regex.
_PORTKEY_VIRTUAL_KEY_RE = re.compile(r"^@[\w-]+/")

# Host-suffix hints used by :func:`_normalize_model_name` to recognise
# gateway traffic when only a ``base_url`` is available. The set is a hint
# (not authoritative) so future gateways added by the customer can use
# the ``<vendor>/`` shape we already strip. Suffix-matched so subdomains
# (``my-tenant.api.portkey.ai``) still hit.
_GATEWAY_HOST_HINTS: frozenset[str] = frozenset(
    {
        "openrouter.ai",
        "api.together.xyz",
        "api.portkey.ai",
        "api.deepinfra.com",
        "api.groq.com",
        "api.fireworks.ai",
        "api.anyscale.com",
        "api.helicone.ai",
        "openai.helicone.ai",
        "oai.helicone.ai",
    }
)


CLASSIFY_KEYWORDS = (
    "classify",
    "categorize",
    "categorise",
    "label this",
    "which category",
    "yes or no",
    "true or false",
    "is this a",
    "is this an",
    "rate from 1",
    "rate this on a scale",
)

# Word-boundary regex compiled once at module load. ``\b`` anchors each keyword
# at a non-word boundary on both sides so ``"classify"`` matches in
# ``"please classify this"`` but NOT in ``"declassified"``. Each keyword is
# ``re.escape``'d so any regex meta-characters in future additions don't break
# the alternation. The match is case-insensitive so we don't need to lowercase
# the prompt before scanning.
_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in CLASSIFY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class ModelMisrouteRule(Rule):
    name = "model_misroute"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None
        c = session[-1]

        max_prompt_tokens = self.get("max_prompt_tokens", 500)
        max_completion_tokens = self.get("max_completion_tokens", 50)

        if c.prompt_tokens > max_prompt_tokens:
            return None
        if c.completion_tokens > max_completion_tokens:
            return None
        # Normalise the model name through the gateway-prefix stripper
        # before the frontier check so gateway traffic ("openai/gpt-4o")
        # is recognised by the bare-name allowlist. ``base_url`` lives on
        # ``raw_request`` for the gateway-aware fallback path; the known
        # vendor prefixes strip unconditionally.
        base_url = _extract_base_url(c)
        normalized_model = _normalize_model_name(c.model, base_url=base_url)
        if not _is_frontier(normalized_model):
            return None

        # Word-boundary scan. We DON'T pre-lowercase the prompt because the
        # regex is case-insensitive and ``re.findall`` on the original string
        # preserves the customer's casing in the match list — but we then
        # normalise each match to lowercase for the evidence field so the
        # ``matched_keywords`` value is stable across "CLASSIFY" / "Classify"
        # / "classify" callers.
        prompt_text = _flatten_messages(c.raw_request.get("messages", []))
        raw_matches = _KEYWORD_PATTERN.findall(prompt_text)
        if not raw_matches:
            return None
        # Deduplicate while preserving first-seen order so the evidence list
        # reads the way a human would scan the prompt.
        seen: set[str] = set()
        matched: list[str] = []
        for m in raw_matches:
            normalised = m.lower()
            if normalised not in seen:
                seen.add(normalised)
                matched.append(normalised)

        recommended = _recommended_alternative(normalized_model)
        return LeakEvent(
            type="model_misroute",
            confidence=0.7,
            project=project,
            session_id=c.session_id,
            rule="v0.model_misroute",
            evidence={
                # Customer's literal model string is preserved so the
                # leak-handler ships the EXACT value the customer used —
                # useful for dashboard filters and "this is what we saw"
                # transparency. The normalised form is shipped alongside
                # iff it differs (i.e., a prefix was stripped) so the
                # consumer can render "openai/gpt-4o → gpt-4o".
                "model": c.model,
                **({"normalized_model": normalized_model} if normalized_model != c.model else {}),
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "matched_keywords": matched[:3],
                "recommended_alternative": recommended,
            },
            estimated_burn=round((c.prompt_tokens + c.completion_tokens) * 5e-5, 4),
            suggested_action=f"route_to_{recommended}",
        )


def _normalize_model_name(model: str, base_url: str | None = None) -> str:
    """Strip gateway-style provider prefixes from ``model`` for misroute matching.

    Used by :class:`ModelMisrouteRule` to fix the coverage gap where
    OpenRouter / Portkey / Together / Helicone customers' model strings
    didn't match the frontier-prefix allowlist:

    .. code-block::

        "anthropic/claude-3.7-sonnet"   → "claude-3.7-sonnet"
        "openai/gpt-4o"                 → "gpt-4o"
        "meta-llama/Meta-Llama-..."     → "Meta-Llama-..."
        "@openai-prod/gpt-4o"           → "gpt-4o"  (Portkey virtual key)
        "groq/llama-3.1-70b-versatile"  → "llama-3.1-70b-versatile"

    Order of checks:

      1. Portkey virtual-key shape (``@<env>/...``) — unambiguous via the
         leading ``@``.
      2. Known vendor prefixes (``anthropic/``, ``openai/``, ...) — these
         strip unconditionally; no direct-provider model name today shares
         this shape.
      3. Generic ``<vendor>/<model>`` fallback — only fires when
         ``base_url`` is set AND points to a known gateway host. This is
         the **preserve-bare-name policy**: a customer using
         ``client = OpenAI()`` with no base_url and passing
         ``model="my-fine-tune/v1"`` keeps the literal string. We only
         strip the catch-all when we have positive evidence (the
         ``base_url``) that the call went through a gateway.

    Returns the input unchanged when no rule fires.
    """
    if not isinstance(model, str) or not model:
        return model
    # Portkey virtual key: ``@<env>/<model>``. Strip the entire ``@<env>/``
    # prefix and recurse — the trailing portion may itself carry a vendor
    # prefix (e.g., ``@openai-prod/openai/gpt-4o`` in unusual setups).
    pk_match = _PORTKEY_VIRTUAL_KEY_RE.match(model)
    if pk_match is not None:
        return _normalize_model_name(model[pk_match.end() :], base_url=base_url)
    # Known vendor prefixes — strip unconditionally.
    for prefix in _VENDOR_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    # Generic ``<vendor>/<model>`` fallback — only applies when we know
    # the call went through a gateway (preserve-bare-name policy).
    if base_url and _looks_like_gateway(base_url) and "/" in model:
        # Strip only the FIRST segment so multi-slash fine-tune aliases
        # (e.g., ``vendor/family/variant``) still preserve the rest.
        head, _, tail = model.partition("/")
        if head and tail:
            return tail
    return model


def _looks_like_gateway(base_url: str) -> bool:
    """Return True if ``base_url``'s host suffix-matches a known gateway.

    The match is suffix-based so subdomains
    (``mytenant.api.portkey.ai``) and path-bearing URLs
    (``https://openrouter.ai/api/v1``) both hit. We don't parse the URL
    strictly (urllib.parse is overkill in the hot path); a substring
    check against the hint set is good enough for the rule-firing
    gate. The set is documented as a hint, not authoritative.
    """
    if not base_url:
        return False
    lower = base_url.lower()
    return any(host in lower for host in _GATEWAY_HOST_HINTS)


def _extract_base_url(call: CallRecord) -> str | None:
    """Pull a ``base_url`` off ``call.raw_request`` if the wrapper recorded it.

    The OpenAI wrapper records ``messages`` / ``tools`` / ``max_tokens``
    today; ``base_url`` lives on the underlying client object, not on
    ``kwargs``, so it isn't currently surfaced. We return ``None`` when
    absent, which keeps the rule's behaviour on direct-provider calls
    unchanged. The generic ``<vendor>/`` strip simply doesn't fire — the
    known vendor prefixes still do. When wrapper changes start
    capturing the client's ``base_url`` on ``raw_request['base_url']``
    (or equivalent), this helper finds it without further rule changes.
    """
    raw = getattr(call, "raw_request", None)
    if not isinstance(raw, dict):
        return None
    value = raw.get("base_url")
    if isinstance(value, str) and value:
        return value
    return None


def _is_frontier(model: str) -> bool:
    """True iff ``model`` matches a frontier prefix and no exclude prefix.

    The exclude check runs first so ``gpt-5-mini`` returns ``False`` even
    though ``gpt-5`` is a frontier prefix.
    """
    if any(model.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    return any(model.startswith(p) for p in FRONTIER_PREFIXES)


def _recommended_alternative(model: str) -> str:
    """Pick the cheap-tier alternative for ``model``.

    Uses the longest matching prefix so ``deepseek-reasoner`` resolves before
    ``deepseek-chat`` and ``claude-3-5-sonnet`` before ``claude-sonnet``.
    """
    matches = [p for p in CHEAP_ALTERNATIVES if model.startswith(p)]
    if not matches:
        return "haiku_or_gpt-4o-mini"  # generic fallback
    longest = max(matches, key=len)
    return CHEAP_ALTERNATIVES[longest]


def _flatten_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for m in messages:
        # Skip non-dict messages (Pydantic BaseModel, str, None, etc.) to
        # avoid silent rule failures from AttributeError on `m.get`.
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
    return " ".join(parts)
