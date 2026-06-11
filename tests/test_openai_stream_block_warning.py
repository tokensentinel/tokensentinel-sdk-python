"""Tests for the defensive-fallback block-mode-on-streaming warning in the
OpenAI wrapper.

As of stable release, OpenAI streaming is fully instrumented via
``_OpenAIStreamProxy`` / ``_AsyncOpenAIStreamProxy``. The block-mode warning
machinery (``_warn_block_mode_stream_once``) is retained as a defensive
fallback: if proxy construction itself fails for any reason (a non-iterable
mock return, a future SDK shape change, etc.), the wrapper falls back to
passthrough and emits a loud, suppressible ``RuntimeWarning`` under
``mode='block'`` so customers know their leak detection is bypassed.

Under normal streaming usage (the proxy CAN be constructed), no warning
fires -- streams are instrumented exactly like Anthropic / Gemini / Bedrock.

Tests verify:
  - Warning is NOT emitted on normal streaming (proxy constructs successfully)
  - Warning IS emitted when proxy construction fails under mode='block'
  - Warning does NOT fire under mode='log' or mode='alert' even on
    proxy-construction failure
  - Warning does NOT fire on non-stream calls
  - Warning fires once per (Sentinel instance, sync/async path)
  - Warning is suppressible via warnings.filterwarnings
  - Two distinct Sentinel instances each get their own warning
  - Sync and async paths each get a separate warning even on the same Sentinel
"""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

import token_sentinel.wrappers.openai as openai_module
from token_sentinel import Sentinel
from token_sentinel.wrappers.openai import (
    _BLOCK_MODE_STREAM_MESSAGE,
    wrap_openai,
)

# ---------------------------------------------------------------------------
# Fixtures: a working, iterable streaming client (proxy construction succeeds)
# ---------------------------------------------------------------------------


def _make_sync_client(stream_chunks: list[Any] | None = None) -> Any:
    """Sync OpenAI client whose stream returns an iterable list of chunks."""
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def _create(**kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            return iter(stream_chunks or [])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))
    return client


def _make_async_client(stream_chunks: list[Any] | None = None) -> Any:
    """Async OpenAI client; ``stream=True`` returns an async iterator."""
    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()

    class _AsyncIter:
        def __init__(self, chunks: list[Any]) -> None:
            self._chunks = chunks

        def __aiter__(self):
            return self._aiter()

        async def _aiter(self):
            for c in self._chunks:
                yield c

    async def _acreate(**kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            return _AsyncIter(stream_chunks or [])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    async def _aembed(**kwargs: Any) -> Any:
        return SimpleNamespace(usage=None)

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_acreate))
    client.embeddings = SimpleNamespace(create=_aembed)
    return client


def _stream_warnings(
    records: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Filter to only the OpenAI-streaming-bypass warnings."""
    return [w for w in records if str(w.message) == _BLOCK_MODE_STREAM_MESSAGE]


# ---------------------------------------------------------------------------
# Helpers to force proxy-construction failure
# ---------------------------------------------------------------------------


class _BoomProxy:
    """A stand-in for ``_OpenAIStreamProxy`` whose constructor always raises.

    Used to simulate the defensive-fallback path -- whatever the cause (a
    future SDK shape, a mocked client returning a wholly unwrappable object,
    etc.), the wrapper must hand back the raw stream and warn under block
    mode rather than crash the user's call.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated proxy construction failure")


# ===========================================================================
# Normal streaming flow: warning is NOT emitted
# ===========================================================================


def test_sync_warning_not_emitted_on_normal_streaming():
    """Streaming under mode='block' must NOT warn under normal flow because
    the proxy IS instrumented now."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        list(out)  # iterate to flush

    assert _stream_warnings(caught) == []


def test_async_warning_not_emitted_on_normal_streaming():
    """Async streaming under mode='block' must NOT warn under normal flow."""
    client = _make_async_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    async def run():
        proxy = await client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        async for _ in proxy:
            pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(run())

    assert _stream_warnings(caught) == []


# ===========================================================================
# Defensive fallback: warning fires when proxy construction fails
# ===========================================================================


def test_sync_warning_fires_on_proxy_construction_failure_under_block_mode(monkeypatch):
    """If proxy construction fails under mode='block', the wrapper falls
    back to passthrough and emits a single RuntimeWarning."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    relevant = _stream_warnings(caught)
    assert len(relevant) == 1
    assert relevant[0].category is RuntimeWarning
    assert str(relevant[0].message) == _BLOCK_MODE_STREAM_MESSAGE


def test_async_warning_fires_on_proxy_construction_failure_under_block_mode(
    monkeypatch,
):
    """Async counterpart of the proxy-construction-failure warning."""
    client = _make_async_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_AsyncOpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    relevant = _stream_warnings(caught)
    assert len(relevant) == 1
    assert relevant[0].category is RuntimeWarning


def test_sync_warning_message_mentions_roadmap(monkeypatch):
    """The canonical message text mentions the roadmap and the alternate providers
    so customers know what to use today."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    relevant = _stream_warnings(caught)
    assert len(relevant) == 1
    msg = str(relevant[0].message)
    assert "roadmap" in msg
    assert "Anthropic" in msg
    assert "Gemini" in msg
    assert "Bedrock" in msg


# ===========================================================================
# Warning does NOT fire under non-block modes (even on construction failure)
# ===========================================================================


@pytest.mark.parametrize("mode", ["log", "alert"])
def test_sync_warning_does_not_fire_under_non_block_modes(monkeypatch, mode):
    """Even on proxy-construction failure, mode='log'/'alert' must be silent
    -- the bypass only matters when block mode would have halted user code."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode=mode)
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert _stream_warnings(caught) == []


@pytest.mark.parametrize("mode", ["log", "alert"])
def test_async_warning_does_not_fire_under_non_block_modes(monkeypatch, mode):
    """Async counterpart of the non-block-mode silence check."""
    client = _make_async_client(stream_chunks=[])
    s = Sentinel(project="proj", mode=mode)
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_AsyncOpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert _stream_warnings(caught) == []


# ===========================================================================
# Warning does NOT fire on non-stream calls
# ===========================================================================


def test_sync_warning_does_not_fire_on_non_stream_calls_under_block_mode():
    client = _make_sync_client()
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="gpt-4o", messages=[])
        client.chat.completions.create(model="gpt-4o", messages=[], stream=False)

    assert _stream_warnings(caught) == []


def test_async_warning_does_not_fire_on_non_stream_calls_under_block_mode():
    client = _make_async_client()
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[]))
        asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[], stream=False))

    assert _stream_warnings(caught) == []


# ===========================================================================
# Warning fires only ONCE per (Sentinel instance, path)
# ===========================================================================


def test_sync_warning_fires_once_per_sentinel_instance(monkeypatch):
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(20):
            client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert len(_stream_warnings(caught)) == 1


def test_async_warning_fires_once_per_sentinel_instance(monkeypatch):
    client = _make_async_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_AsyncOpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(20):
            asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert len(_stream_warnings(caught)) == 1


def test_distinct_sentinels_each_get_their_own_warning(monkeypatch):
    """Two separate Sentinel instances each produce a warning -- bookkeeping
    is per-instance."""
    client_a = _make_sync_client(stream_chunks=[])
    client_b = _make_sync_client(stream_chunks=[])
    s_a = Sentinel(project="proj-a", mode="block")
    s_b = Sentinel(project="proj-b", mode="block")
    wrap_openai(client_a, s_a)
    wrap_openai(client_b, s_b)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client_a.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        client_b.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        # Repeats -- each instance still warns only once.
        client_a.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        client_b.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert len(_stream_warnings(caught)) == 2


def test_sync_and_async_paths_warn_separately_on_same_sentinel(monkeypatch):
    """The same Sentinel + sync + async clients => two warnings (one per path)."""
    sync_client = _make_sync_client(stream_chunks=[])
    async_client = _make_async_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(sync_client, s)
    wrap_openai(async_client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)
    monkeypatch.setattr(openai_module, "_AsyncOpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sync_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        sync_client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        asyncio.run(async_client.chat.completions.create(model="gpt-4o", messages=[], stream=True))
        asyncio.run(async_client.chat.completions.create(model="gpt-4o", messages=[], stream=True))

    assert len(_stream_warnings(caught)) == 2


# ===========================================================================
# Suppressibility via warnings.filterwarnings
# ===========================================================================


def test_warning_is_suppressible_via_filterwarnings(monkeypatch):
    """The warning is suppressible via the standard ``warnings.filterwarnings``
    mechanism -- the contract for a well-behaved RuntimeWarning."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("ignore", message=".*OpenAI streaming bypass.*")
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert _stream_warnings(caught) == []


def test_warning_is_suppressible_by_category(monkeypatch):
    """Suppressing all RuntimeWarnings also silences ours -- confirms a
    stdlib-compatible RuntimeWarning."""
    client = _make_sync_client(stream_chunks=[])
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", RuntimeWarning)
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    assert _stream_warnings(caught) == []


# ===========================================================================
# Defensive-fallback behavior: passthrough still works
# ===========================================================================


def test_warning_does_not_break_streaming_passthrough(monkeypatch):
    """On proxy-construction failure, the wrapper hands back the raw stream
    so user code doesn't break (the fallback contract)."""
    sentinel_marker = iter([])  # an iterable distinct from the proxy
    client = _make_sync_client(stream_chunks=[])
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)
    # Override the create to return the marker so we can identity-check it.

    def _create(**kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            return sentinel_marker
        return SimpleNamespace(choices=[], usage=None)

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = client.chat.completions.create(model="gpt-4o", messages=[], stream=True)

    # Got back the raw iterator -- proxy construction failed so we fell back.
    assert out is sentinel_marker


def test_warning_does_not_break_session_id_kwarg_strip(monkeypatch):
    """The defensive-fallback path still strips ``_sentinel_session_id``
    before the SDK sees it (regression guard)."""
    received: dict[str, Any] = {}

    def real_create(**kwargs: Any) -> Any:
        received.update(kwargs)
        return iter([])

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=real_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)
    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="abc",
        )

    assert "_sentinel_session_id" not in received
    assert received.get("stream") is True
