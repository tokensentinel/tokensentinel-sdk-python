"""Tests for ``Sentinel.wrap`` accessor-failure error messages (MED-1).

Closes the v0.2.0 code-review finding "Sentinel.wrap raises if the client's
accessor properties throw" — a partially-initialised client whose attribute
access raises (e.g., a property that throws until config is set) used to
crash with an opaque ``AttributeError`` deep in the wrapper module. ``wrap``
now validates the expected accessor for each provider *before* delegating
to the provider-specific wrapper, and raises a clear ``TypeError`` (chaining
the original exception via ``__cause__``) when the access fails.

We do NOT swallow the error — failure isolation is for the hot path, not
for setup-time misuse. The customer needs to know their client isn't usable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel

# ---------------------------------------------------------------------------
# Test helpers — properties that raise on access
# ---------------------------------------------------------------------------


def _make_anthropic_class_with_raising_messages(exc: BaseException) -> type:
    """Build an Anthropic-shaped class whose ``messages`` property raises."""

    class _Messages:
        @property
        def create(self) -> Any:  # noqa: D401 — property
            raise exc

    class FakeAnthropic:
        __module__ = "anthropic"

        @property
        def messages(self) -> _Messages:  # type: ignore[override]
            raise exc

    return FakeAnthropic


def _make_anthropic_class_with_raising_create(exc: BaseException) -> type:
    """Build an Anthropic-shaped class where ``messages.create`` raises."""

    class _Messages:
        @property
        def create(self) -> Any:
            raise exc

    class FakeAnthropic:
        __module__ = "anthropic"

        def __init__(self) -> None:
            self.messages = _Messages()

    return FakeAnthropic


def _make_openai_class_with_raising_chat(exc: BaseException) -> type:
    class FakeOpenAI:
        __module__ = "openai"

        @property
        def chat(self) -> Any:
            raise exc

        # Provide a real embeddings.create so we know the failure was on chat.
        def __init__(self) -> None:
            self.embeddings = SimpleNamespace(create=lambda *a, **kw: None)

    return FakeOpenAI


def _make_openai_class_with_raising_embeddings(exc: BaseException) -> type:
    class FakeOpenAI:
        __module__ = "openai"

        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **kw: None))

        @property
        def embeddings(self) -> Any:
            raise exc

    return FakeOpenAI


def _make_gemini_class_with_raising_models(exc: BaseException) -> type:
    class FakeGenAI:
        __module__ = "google.genai"

        @property
        def models(self) -> Any:
            raise exc

    return FakeGenAI


def _make_bedrock_class_with_raising_converse(exc: BaseException) -> type:
    class FakeBedrock:
        # Bedrock detection is name- or service_model-based; "bedrock" in the
        # class name is the cheapest fingerprint.
        __module__ = "botocore.client"

        @property
        def converse(self) -> Any:
            raise exc

    FakeBedrock.__name__ = "BedrockRuntime"
    return FakeBedrock


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_messages_attributeerror_yields_clear_typeerror():
    """``client.messages`` raising AttributeError must surface as a TypeError."""
    cls = _make_anthropic_class_with_raising_messages(AttributeError("boom"))
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "Sentinel could not instrument" in msg
    assert "messages" in msg


def test_anthropic_messages_attributeerror_chains_original_via_cause():
    """The original AttributeError must be preserved on ``__cause__``."""
    original = AttributeError("client not initialised")
    cls = _make_anthropic_class_with_raising_messages(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    assert exc_info.value.__cause__ is original


def test_anthropic_messages_create_attributeerror_yields_clear_typeerror():
    """``client.messages.create`` raising AttributeError surfaces as TypeError."""
    original = AttributeError("create unavailable")
    cls = _make_anthropic_class_with_raising_create(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    assert "messages.create" in str(exc_info.value)
    assert exc_info.value.__cause__ is original


def test_anthropic_messages_non_attribute_error_yields_clear_typeerror():
    """A non-AttributeError (e.g. RuntimeError) on the accessor still maps to TypeError."""
    original = RuntimeError("client not configured")
    cls = _make_anthropic_class_with_raising_messages(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "RuntimeError" in msg
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# OpenAI — both chat.completions.create and embeddings.create paths
# ---------------------------------------------------------------------------


def test_openai_chat_completions_create_attributeerror_yields_clear_typeerror():
    original = AttributeError("chat not ready")
    cls = _make_openai_class_with_raising_chat(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "chat" in msg
    assert exc_info.value.__cause__ is original


def test_openai_embeddings_create_attributeerror_yields_clear_typeerror():
    original = AttributeError("embeddings not ready")
    cls = _make_openai_class_with_raising_embeddings(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "embeddings" in msg
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_models_attributeerror_yields_clear_typeerror():
    original = AttributeError("models not configured")
    cls = _make_gemini_class_with_raising_models(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "models" in msg
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


def test_bedrock_converse_attributeerror_yields_clear_typeerror():
    original = AttributeError("converse op missing")
    cls = _make_bedrock_class_with_raising_converse(original)
    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(cls())
    msg = str(exc_info.value)
    assert "converse" in msg
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Non-LLM clients still get the unmodified "Unsupported client type" TypeError
# ---------------------------------------------------------------------------


def test_unsupported_client_type_keeps_original_error_message():
    """A random non-LLM object must hit the original ``Unsupported client
    type`` branch — NOT the new accessor-validation TypeError."""

    class RandomThing:
        __module__ = "totally.unrelated"

    s = Sentinel(project="proj")
    with pytest.raises(TypeError) as exc_info:
        s.wrap(RandomThing())
    msg = str(exc_info.value)
    assert "Unsupported client type" in msg
    # Must NOT be the accessor-validation message — that one starts with
    # "Sentinel could not instrument".
    assert "could not instrument" not in msg
    # No __cause__ chain on the unsupported-type path.
    assert exc_info.value.__cause__ is None


# ---------------------------------------------------------------------------
# Happy-path wrap is unaffected by the new validation step
# ---------------------------------------------------------------------------


def test_happy_path_wrap_unaffected_anthropic(mock_anthropic_client):
    """A well-formed Anthropic mock still wraps cleanly with no exception."""
    s = Sentinel(project="proj")
    wrapped = s.wrap(mock_anthropic_client)
    assert wrapped is mock_anthropic_client
    # The instrumented method must be installed (i.e. wrap_anthropic ran).
    assert callable(wrapped.messages.create)


def test_happy_path_wrap_unaffected_openai(mock_openai_client):
    """A well-formed OpenAI mock still wraps cleanly with no exception."""
    s = Sentinel(project="proj")
    wrapped = s.wrap(mock_openai_client)
    assert wrapped is mock_openai_client
    assert callable(wrapped.chat.completions.create)
    assert callable(wrapped.embeddings.create)
