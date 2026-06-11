"""Tests for ``ModelMisrouteRule``.

Defaults: max_prompt_tokens=500, max_completion_tokens=50.
Frontier prefixes: claude-opus, claude-sonnet, gpt-5, gpt-4-turbo, gpt-4o,
                  gemini-2.5-pro, gemini-2.0-pro, deepseek-chat, deepseek-reasoner,
                  command-r-plus, command-a, mistral-large.
Classify keywords (word-boundary matched, case-insensitive in v0.3.2):
                  classify, categorize, categorise, yes or no, true or false,
                  rate from 1, rate this on a scale, is this a, is this an,
                  label this, which category.
"""

from __future__ import annotations

import pytest

from token_sentinel.rules.model_misroute import (
    CLASSIFY_KEYWORDS,
    FRONTIER_PREFIXES,
    ModelMisrouteRule,
    _flatten_messages,
    _normalize_model_name,
)

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert ModelMisrouteRule({}).evaluate([], project="p") is None


def test_non_frontier_model_no_fire(make_call):
    """Haiku is fine — model_misroute only flags frontier-bound classify prompts."""
    session = [
        make_call(
            model="claude-haiku-3-5",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Classify this as positive."}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_no_classify_keyword_no_fire(make_call):
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Write a poem about hamsters."}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_long_prompt_no_fire(make_call):
    """501 prompt tokens > max_prompt_tokens=500 → not classification-shaped."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=501,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Classify this sentence."}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_long_completion_no_fire(make_call):
    """51 completion tokens > max_completion_tokens=50 → not classification-shaped."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=51,
            raw_request={"messages": [{"role": "user", "content": "Classify this sentence."}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_no_messages_in_request(make_call):
    """Missing ``messages`` key — no text to scan, no fire."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=10,
            completion_tokens=5,
            raw_request={},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "gpt-5-medium",
        "gpt-4-turbo-2024",
        "gpt-4o",
        "gemini-2.5-pro",
        "deepseek-reasoner",
        "command-r-plus-08-2024",
        "mistral-large-latest",
    ],
)
def test_each_frontier_prefix_fires(make_call, model):
    session = [
        make_call(
            model=model,
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "Classify this sentence as positive or negative",
                    }
                ]
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["model"] == model
    assert ev.confidence == 0.7


@pytest.mark.parametrize(
    "keyword,prompt_text",
    [
        ("classify", "Classify the sentiment please"),
        ("yes or no", "Was the meeting useful — yes or no"),
        ("true or false", "true or false: Paris is in Spain"),
        ("rate from 1", "rate from 1 to 10 the quality"),
        ("rate this on a scale", "rate this on a scale of 1-5"),
        ("is this a", "is this a happy story?"),
        ("categorize", "categorize the following document"),
        ("label this", "label this image with one word"),
        ("which category", "which category does this fit best?"),
    ],
)
def test_each_classify_keyword_fires(make_call, keyword, prompt_text):
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": prompt_text}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert keyword in ev.evidence["matched_keywords"]


def test_event_fields(make_call):
    session = [
        make_call(
            model="claude-opus-4-1",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": "Classify this as positive or negative"}]
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "model_misroute"
    assert ev.rule == "v0.model_misroute"
    # claude-opus-4-1 → CHEAP_ALTERNATIVES['claude-opus'] = 'claude-haiku-4-5'
    assert ev.suggested_action.startswith("route_to_")
    assert "haiku" in ev.suggested_action.lower()
    assert ev.estimated_burn > 0
    assert "model" in ev.evidence
    assert "matched_keywords" in ev.evidence


def test_uses_only_last_call(make_call):
    """Rule operates on session[-1] only."""
    early = make_call(
        model="claude-sonnet-4-6",
        prompt_tokens=10,
        completion_tokens=5,
        raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
    )
    later = make_call(
        model="claude-haiku-3-5",
        prompt_tokens=10,
        completion_tokens=5,
        raw_request={"messages": [{"role": "user", "content": "Classify y"}]},
    )
    # Only the latest call (haiku) is checked → no fire.
    assert ModelMisrouteRule({}).evaluate([early, later], project="p") is None


def test_content_as_block_list(make_call):
    """Anthropic sends ``content`` as a list of blocks. Rule must flatten them."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Classify this as positive."}],
                    }
                ]
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None


def test_keyword_case_insensitive(make_call):
    """Keywords are matched against ``.lower()`` of the flattened text."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=50,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "CLASSIFY THIS AS POSITIVE"}]},
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None


# ---------------------------------------------------------------------------
# Threshold boundaries
# ---------------------------------------------------------------------------


def test_at_max_prompt_tokens_exactly(make_call):
    """500 prompt tokens exactly → still fires (rule uses > not >=)."""
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=500,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is not None


def test_at_max_completion_tokens_exactly(make_call):
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=50,
            raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is not None


def test_one_above_max_prompt(make_call):
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=501,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_one_above_max_completion(make_call):
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=51,
            raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_custom_thresholds(make_call):
    rule = ModelMisrouteRule(
        {
            "model_misroute.max_prompt_tokens": 100,
            "model_misroute.max_completion_tokens": 10,
        }
    )
    session = [
        make_call(
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={"messages": [{"role": "user", "content": "Classify x"}]},
        )
    ]
    assert rule.evaluate(session, project="p") is not None


# ---------------------------------------------------------------------------
# Helper: _flatten_messages
# ---------------------------------------------------------------------------


def test_flatten_messages_string_content():
    msgs = [{"role": "user", "content": "hello"}]
    assert _flatten_messages(msgs) == "hello"


def test_flatten_messages_block_list():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    assert _flatten_messages(msgs) == "hello world"


def test_flatten_messages_mixed():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
    ]
    assert _flatten_messages(msgs) == "first second"


def test_flatten_messages_empty():
    assert _flatten_messages([]) == ""


def test_flatten_messages_block_without_text_field():
    """Image / tool_use blocks lack a ``text`` field — must be skipped."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": "..."}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    assert _flatten_messages(msgs) == "describe"


def test_constants_align_with_spec():
    """Sanity: keywords and prefixes match the leak-taxonomy spec."""
    assert "claude-opus" in FRONTIER_PREFIXES
    assert "gpt-4o" in FRONTIER_PREFIXES
    assert "classify" in CLASSIFY_KEYWORDS


# ---------------------------------------------------------------------------
# False-positive hazards
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="V1 mitigation: customer allow-list per route name",
    strict=False,
)
def test_legitimate_nuanced_classification_should_not_fire(make_call):
    """Classify legal/medical text needs Opus — no per-route allow-list in V0."""
    session = [
        make_call(
            model="claude-opus-4-1",
            prompt_tokens=200,
            completion_tokens=10,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "Classify the legal liability exposure of this clause.",
                    }
                ]
            },
        )
    ]
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# gateway model-name normalisation
# ---------------------------------------------------------------------------


def test_normalize_strips_anthropic_prefix():
    """OpenRouter ships ``anthropic/<name>``; the normaliser must strip it."""
    assert _normalize_model_name("anthropic/claude-3.7-sonnet") == "claude-3.7-sonnet"
    # The stripped form must match the frontier allowlist via ``startswith``.
    assert _normalize_model_name("anthropic/claude-opus-4-7") == "claude-opus-4-7"


def test_normalize_strips_meta_llama_prefix():
    """Together-style ``meta-llama/<name>`` strips to the bare model."""
    raw = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    assert _normalize_model_name(raw) == "Meta-Llama-3.1-70B-Instruct-Turbo"


def test_normalize_strips_openai_prefix():
    """OpenRouter ``openai/gpt-4o`` strips to ``gpt-4o`` for misroute matching."""
    assert _normalize_model_name("openai/gpt-4o") == "gpt-4o"
    assert _normalize_model_name("openai/gpt-4o-mini") == "gpt-4o-mini"


def test_normalize_strips_portkey_virtual_key():
    """Portkey ``@<env>/<model>`` virtual-key shape strips the ``@env/`` head."""
    assert _normalize_model_name("@openai-prod/gpt-4o") == "gpt-4o"
    # And a multi-segment env name with dashes/digits matches the regex.
    assert _normalize_model_name("@my-env-2026/claude-opus") == "claude-opus"


def test_normalize_strips_portkey_then_vendor_prefix():
    """Stacked Portkey + vendor: ``@env/anthropic/<model>`` → bare name."""
    raw = "@openai-prod/anthropic/claude-3.7-sonnet"
    assert _normalize_model_name(raw) == "claude-3.7-sonnet"


def test_normalize_preserves_bare_name():
    """A model with no prefix and no slash is left unchanged."""
    assert _normalize_model_name("claude-opus-4-7") == "claude-opus-4-7"
    assert _normalize_model_name("gpt-4o") == "gpt-4o"


def test_normalize_does_not_strip_without_base_url():
    """A bare name that happens to contain ``/`` (a fine-tune alias on a
    direct-provider call) is left unchanged. The catch-all ``<vendor>/``
    strip only fires when ``base_url`` is set to a known gateway."""
    # Not a known vendor prefix + no base_url → preserved.
    assert _normalize_model_name("my-team/fine-tune-v1") == "my-team/fine-tune-v1"
    # Same string with a non-gateway base_url → still preserved.
    assert (
        _normalize_model_name(
            "my-team/fine-tune-v1",
            base_url="https://api.openai.com/v1",
        )
        == "my-team/fine-tune-v1"
    )


def test_normalize_strips_unknown_vendor_when_gateway_hinted():
    """An unknown ``<vendor>/<model>`` string strips iff base_url hints a gateway."""
    # No base_url → preserved.
    assert _normalize_model_name("custom-vendor/some-model") == "custom-vendor/some-model"
    # OpenRouter base_url → catch-all strips the first segment.
    assert (
        _normalize_model_name(
            "custom-vendor/some-model",
            base_url="https://openrouter.ai/api/v1",
        )
        == "some-model"
    )


def test_normalize_handles_empty_and_non_string():
    """Defensive: empty string and non-string inputs are returned as-is."""
    assert _normalize_model_name("") == ""
    # Bypass the type hint to confirm the runtime guard works.
    assert _normalize_model_name(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# full-rule integration through gateway model strings
# ---------------------------------------------------------------------------


def test_misroute_fires_on_openrouter_prefixed_frontier_model(make_call):
    """An OpenRouter call with ``anthropic/<name>`` on a classify prompt
    must fire — pre- it was silently invisible to the allowlist."""
    session = [
        make_call(
            model="anthropic/claude-opus-4-7",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "Classify this as positive or negative",
                    }
                ],
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    # Customer's literal model is preserved in evidence; the normalised
    # form is shipped alongside so handlers can render the strip.
    assert ev.evidence["model"] == "anthropic/claude-opus-4-7"
    assert ev.evidence["normalized_model"] == "claude-opus-4-7"
    # ``claude-opus`` resolves to haiku via the existing CHEAP_ALTERNATIVES dict.
    assert "haiku" in ev.suggested_action.lower()


def test_misroute_fires_on_portkey_virtual_key(make_call):
    """Portkey ``@<env>/<model>`` resolves to a frontier match."""
    session = [
        make_call(
            model="@openai-prod/gpt-4o",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": "Classify x as A or B"}],
                "base_url": "https://api.portkey.ai/v1",
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["normalized_model"] == "gpt-4o"
    # gpt-4o → gpt-4o-mini per CHEAP_ALTERNATIVES.
    assert "gpt-4o-mini" in ev.suggested_action


def test_misroute_doesnt_fire_on_classify_routed_to_haiku(make_call):
    """Even with the ``anthropic/`` prefix, a haiku call is the desired
    state — the normalised form must NOT trip the frontier check."""
    session = [
        make_call(
            model="anthropic/claude-haiku-4-5",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": "Classify this"}],
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
    ]
    # Haiku is not a frontier prefix → no fire even after normalisation.
    assert ModelMisrouteRule({}).evaluate(session, project="p") is None


def test_misroute_event_omits_normalized_when_unchanged(make_call):
    """When ``c.model`` already is the canonical bare name, the
    ``normalized_model`` field must NOT be added to evidence (so existing
    dashboards reading ``evidence["model"]`` keep working unchanged)."""
    session = [
        make_call(
            model="claude-opus-4-1",
            prompt_tokens=80,
            completion_tokens=5,
            raw_request={
                "messages": [{"role": "user", "content": "Classify x"}],
            },
        )
    ]
    ev = ModelMisrouteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["model"] == "claude-opus-4-1"
    assert "normalized_model" not in ev.evidence
