"""Tests: OpenAI-compatible endpoints flow through ``wrap_openai`` identically.

The dispatch in ``Sentinel.wrap`` is purely module-string based:
``type(client).__module__.startswith("openai")``. When you instantiate
``openai.OpenAI(base_url="https://api.deepseek.com")`` (or Together / Fireworks
/ Groq / OpenRouter / vLLM / Ollama / ...) the type still belongs to the
``openai`` SDK and the wrapper instrumentation is identical. The ``base_url``
is just an HTTP transport detail the wrapper doesn't care about.

These tests verify that property end-to-end with mock clients. **No real API
calls are made** — we attach mock chat/embeddings methods to a class whose
``__module__`` is ``openai`` and exercise the wrapper.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord

# ---------------------------------------------------------------------------
# Provider matrix: (label, base_url, model commonly served by that provider)
# ---------------------------------------------------------------------------

OPENAI_COMPAT_PROVIDERS = [
    pytest.param(
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-chat",
        id="deepseek",
    ),
    pytest.param(
        "together",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        id="together",
    ),
    pytest.param(
        "fireworks",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        id="fireworks",
    ),
    pytest.param(
        "groq",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        id="groq",
    ),
    pytest.param(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "anthropic/claude-3.5-sonnet",
        id="openrouter",
    ),
]


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _chat_response(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 30,
    text: str = "ok",
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build an OpenAI-shaped chat completion response (works for compat too)."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _embedding_response(*, prompt_tokens: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
    )


class _RecordingCreate:
    """Same shape as conftest._RecordingCreate but local — keeps the test file
    self-contained and avoids leaking conftest internals."""

    __name__ = "create"
    __qualname__ = "Compat.create"
    __module__ = "openai.resources.chat.completions"
    __annotations__: dict = {}
    __doc__ = "mock create"

    def __init__(self, return_value: Any = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value = return_value
        self.base_url: str | None = None  # purely informational

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        return self.return_value


def _make_compat_client(*, base_url: str) -> Any:
    """Build a mock OpenAI client with a custom ``base_url`` set on it.

    The crucial property is ``type(client).__module__ == 'openai'`` — that's
    what ``Sentinel.wrap`` routes on. The ``base_url`` is stored on the
    instance (mirroring what the real ``openai.OpenAI(base_url=...)``
    constructor does) but is otherwise irrelevant to instrumentation.
    """
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.base_url = base_url
    chat_create = _RecordingCreate(return_value=_chat_response())
    embed_create = _RecordingCreate(return_value=_embedding_response())
    embed_create.__module__ = "openai.resources.embeddings"
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    client.embeddings = SimpleNamespace(create=embed_create)
    return client


# ---------------------------------------------------------------------------
# Parameterised tests across the OpenAI-compat provider matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,base_url,model", OPENAI_COMPAT_PROVIDERS)
def test_compat_chat_completion_records_call(label: str, base_url: str, model: str):
    """For every OpenAI-compat provider, ``wrap_openai`` records a CallRecord
    with the right model, token counts, and provider tag."""
    client = _make_compat_client(base_url=base_url)
    client.chat.completions.create.return_value = _chat_response(
        prompt_tokens=77, completion_tokens=11, text=f"reply from {label}"
    )

    sentinel = Sentinel(project=f"compat-{label}")
    sentinel.wrap(client)  # routes to wrap_openai because module="openai"

    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"hi {label}"}],
        max_tokens=20,
        _sentinel_session_id=f"s-{label}",
    )

    records = sentinel.tracer.session(f"s-{label}")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    # Provider is always "openai" — the wrapper can't distinguish DeepSeek
    # from Together from the SDK type alone. The model name is the
    # discriminator the customer sees in evidence.
    assert rec.provider == "openai"
    assert rec.method == "chat.completions.create"
    assert rec.model == model
    assert rec.prompt_tokens == 77
    assert rec.completion_tokens == 11
    # base_url is purely an HTTP transport concern; not stored in the record.
    assert "base_url" not in rec.raw_request


@pytest.mark.parametrize("label,base_url,model", OPENAI_COMPAT_PROVIDERS)
def test_compat_dispatch_routes_to_wrap_openai(label: str, base_url: str, model: str):
    """``Sentinel.wrap`` routes a compat client through ``wrap_openai``.

    Verified by checking the ``create`` method got swapped (i.e., wrapper ran
    successfully and didn't raise ``TypeError: Unsupported client type``).
    """
    client = _make_compat_client(base_url=base_url)
    original_chat = client.chat.completions.create
    original_embed = client.embeddings.create

    sentinel = Sentinel(project="dispatch-test")
    out = sentinel.wrap(client)

    assert out is client  # wrap returns the same instance, mutated
    assert client.chat.completions.create is not original_chat
    assert client.embeddings.create is not original_embed


# ---------------------------------------------------------------------------
# Identical instrumentation: hash + record shape match across providers
# ---------------------------------------------------------------------------


def test_request_hash_is_independent_of_base_url():
    """Two identical calls to different OpenAI-compat providers (same model
    name + same messages) produce the same ``request_hash``. The hash covers
    the request payload only — base_url is not part of it."""
    client_a = _make_compat_client(base_url="https://api.deepseek.com")
    client_b = _make_compat_client(base_url="https://api.together.xyz/v1")

    s_a = Sentinel(project="hash-a")
    s_b = Sentinel(project="hash-b")
    s_a.wrap(client_a)
    s_b.wrap(client_b)

    payload = {
        "model": "shared-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    client_a.chat.completions.create(_sentinel_session_id="x", **payload)
    client_b.chat.completions.create(_sentinel_session_id="y", **payload)

    rec_a = s_a.tracer.session("x")[0]
    rec_b = s_b.tracer.session("y")[0]
    assert rec_a.request_hash == rec_b.request_hash


def test_compat_embeddings_records_call():
    """OpenAI-compat embeddings (e.g., Together's BGE) flow through
    ``_patch_embeddings`` identically — same request hash logic, same record."""
    client = _make_compat_client(base_url="https://api.together.xyz/v1")
    client.embeddings.create.return_value = _embedding_response(prompt_tokens=42)

    sentinel = Sentinel(project="emb")
    sentinel.wrap(client)

    client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input="lookup query",
        _sentinel_session_id="emb-1",
    )

    rec = sentinel.tracer.session("emb-1")[0]
    assert rec.method == "embeddings.create"
    assert rec.model == "BAAI/bge-large-en-v1.5"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 0
    # Hash should match the documented (model, input) shape — same scheme as
    # native OpenAI, which is what makes embedding_waste work transparently.
    expected = hashlib.sha256(
        json.dumps(
            {"model": "BAAI/bge-large-en-v1.5", "input": "lookup query"},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    assert rec.request_hash == expected


def test_compat_session_id_kwarg_stripped_before_hitting_provider():
    """``_sentinel_session_id`` must never reach the underlying provider —
    DeepSeek / Together / etc. would 400 on an unknown kwarg."""
    client = _make_compat_client(base_url="https://api.deepseek.com")
    sentinel = Sentinel(project="strip")
    sentinel.wrap(client)

    client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        _sentinel_session_id="leaked",
    )

    # Inspect what the underlying create method received
    received_kwargs = client.chat.completions.create.__wrapped__  # type: ignore[attr-defined]
    # The original method (now closure-captured) recorded the call. The mock
    # _RecordingCreate is the original; reach it via the wrapper's closure.
    # functools.wraps copies attributes, so __wrapped__ points to the original.
    # The original's `calls` list shows what arguments it actually saw.
    original = received_kwargs
    assert hasattr(original, "calls")
    assert len(original.calls) == 1
    _, kwargs = original.calls[0]
    assert "_sentinel_session_id" not in kwargs
    assert kwargs["model"] == "deepseek-chat"


def test_self_hosted_endpoints_also_dispatch_through_wrap_openai():
    """Local vLLM / Ollama URLs are just another base_url. Dispatch is
    identical — the customer can hit ``http://localhost:11434/v1`` and get
    the same instrumentation as a public API."""
    cases = [
        ("http://localhost:8000/v1", "meta-llama/Llama-3.1-8B-Instruct"),  # vLLM
        ("http://localhost:11434/v1", "llama3.2"),  # Ollama
        ("http://localhost:3000/v1", "mistral-7b"),  # LM Studio
    ]
    for base_url, model in cases:
        client = _make_compat_client(base_url=base_url)
        sentinel = Sentinel(project="self-hosted")
        sentinel.wrap(client)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id=f"s-{model}",
        )
        records = sentinel.tracer.session(f"s-{model}")
        assert len(records) == 1
        assert records[0].provider == "openai"
        assert records[0].model == model


def test_compat_streaming_instrumented_for_all_providers():
    """``stream=True`` is fully instrumented across the compat matrix as of
    stable release -- behavior must match native OpenAI exactly so customers see
    streaming records on DeepSeek/Together/etc identically.
    """
    for label, base_url, model in [
        ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("fireworks", "https://api.fireworks.ai/inference/v1", "fw-model"),
    ]:
        client = _make_compat_client(base_url=base_url)
        # Provide a real iterable so the proxy can wrap it.
        client.chat.completions.create.return_value = []

        sentinel = Sentinel(project=f"stream-{label}")
        sentinel.wrap(client)
        proxy = client.chat.completions.create(
            model=model,
            messages=[],
            stream=True,
            _sentinel_session_id=f"s-{label}",
        )
        # Iterate the proxy to flush the record.
        observed = list(proxy)
        assert observed == []
        # Streaming is now instrumented -- a record exists per stable release.
        assert len(sentinel.tracer.session(f"s-{label}")) == 1
        rec = sentinel.tracer.session(f"s-{label}")[0]
        assert rec.provider == "openai"
        assert rec.model == model
        assert rec.raw_response_meta.get("streamed") is True


def test_compat_unsupported_provider_module_still_rejected():
    """Sanity check: a client whose module is *not* ``openai`` must NOT pick
    up ``wrap_openai`` even if it has an ``openai``-shaped surface. This
    confirms the dispatch is module-name-based, not duck-typed, so we don't
    accidentally instrument unrelated SDKs."""
    cls = type("ImpostorClient", (), {"__module__": "some_other_sdk"})
    client = cls()
    client.base_url = "https://api.deepseek.com"
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_RecordingCreate()))
    client.embeddings = SimpleNamespace(create=_RecordingCreate())

    sentinel = Sentinel(project="impostor")
    with pytest.raises(TypeError, match="Unsupported client type"):
        sentinel.wrap(client)
