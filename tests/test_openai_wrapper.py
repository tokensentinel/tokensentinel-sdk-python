"""Tests for ``token_sentinel.wrappers.openai.wrap_openai``.

NO real API calls. Mirrors ``test_anthropic_wrapper`` for the OpenAI surface:
sync + async clients, chat completions, and embeddings.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.wrappers.openai import (
    _build_chat_record,
    _build_embedding_record,
    wrap_openai,
)

# ---------------------------------------------------------------------------
# Mock OpenAI response factories
# ---------------------------------------------------------------------------


def _chat_response(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 30,
    finish_reason: str = "stop",
    text: str | None = "hello",
    tool_calls: list[dict] | None = None,
) -> SimpleNamespace:
    """Build an OpenAI-shaped chat completion response."""
    raw_tool_calls = []
    if tool_calls:
        for tc in tool_calls:
            raw_tool_calls.append(
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=tc["name"],
                        arguments=tc.get("arguments", "{}"),
                    )
                )
            )
    message = SimpleNamespace(
        content=text,
        tool_calls=raw_tool_calls or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _embedding_response(*, prompt_tokens: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
    )


# ---------------------------------------------------------------------------
# wrap_openai: instrumentation
# ---------------------------------------------------------------------------


def test_wrap_replaces_chat_and_embeddings(mock_openai_client):
    s = Sentinel(project="proj")
    chat_orig = mock_openai_client.chat.completions.create
    embed_orig = mock_openai_client.embeddings.create
    wrap_openai(mock_openai_client, s)
    assert mock_openai_client.chat.completions.create is not chat_orig
    assert mock_openai_client.embeddings.create is not embed_orig


def test_wrap_returns_same_client(mock_openai_client):
    s = Sentinel(project="proj")
    out = wrap_openai(mock_openai_client, s)
    assert out is mock_openai_client


def test_chat_completions_records(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = _chat_response(
        prompt_tokens=42, completion_tokens=8, text="hi"
    )
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    mock_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        _sentinel_session_id="s-1",
    )
    records = s.tracer.session("s-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "openai"
    assert rec.method == "chat.completions.create"
    assert rec.model == "gpt-4o"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 8
    assert rec.user_facing_output is True
    assert rec.raw_response_meta == {"finish_reason": "stop"}


def test_chat_completions_with_tool_calls_not_user_facing(mock_openai_client):
    """Response with tool_calls is intermediate, not user-facing."""
    mock_openai_client.chat.completions.create.return_value = _chat_response(
        text="I will call a tool",
        tool_calls=[{"name": "search", "arguments": '{"q": "kittens"}'}],
        finish_reason="tool_calls",
    )
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    mock_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        _sentinel_session_id="s-1",
    )
    rec = s.tracer.session("s-1")[0]
    assert rec.user_facing_output is False
    # OpenAI tool args are JSON-encoded strings; wrapper must parse them.
    assert rec.tool_calls == [{"name": "search", "arguments": {"q": "kittens"}}]


def test_chat_completions_tool_call_unparseable_arguments(mock_openai_client):
    """If arguments aren't valid JSON, the raw string is preserved."""
    mock_openai_client.chat.completions.create.return_value = _chat_response(
        text=None,
        tool_calls=[{"name": "search", "arguments": "not valid json {"}],
    )
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    mock_openai_client.chat.completions.create(
        model="gpt-4o", messages=[], _sentinel_session_id="s-1"
    )
    rec = s.tracer.session("s-1")[0]
    assert rec.tool_calls[0]["arguments"] == "not valid json {"


def test_embeddings_records(mock_openai_client):
    mock_openai_client.embeddings.create.return_value = _embedding_response(prompt_tokens=12)
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    mock_openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="user query",
        _sentinel_session_id="s-1",
    )
    records = s.tracer.session("s-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "embeddings.create"
    assert rec.prompt_tokens == 12
    assert rec.completion_tokens == 0
    assert rec.raw_request["input"] == "user query"


def test_chat_streaming_returns_proxy(mock_openai_client):
    """``stream=True`` returns an iterable proxy as of stable release.

    The proxy delegates to the underlying SDK ``Stream`` so user code that
    iterates the result keeps working. This test confirms the proxy is NOT
    the raw passthrough object (the underlying ``return_value``).
    """
    # Use an iterable (list) as the stand-in stream so the proxy can wrap it.
    sentinel_response: list[Any] = []  # empty stream — finalizes immediately
    mock_openai_client.chat.completions.create.return_value = sentinel_response
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    out = mock_openai_client.chat.completions.create(
        model="gpt-4o", messages=[], stream=True, _sentinel_session_id="s-1"
    )
    # Proxy wraps the underlying iterable -- it is NOT identity-equal to the
    # raw return_value (that's the whole point of streaming instrumentation).
    assert out is not sentinel_response
    # Iterating an empty stream finalizes the record immediately.
    assert list(out) == []
    # An (empty-content) record exists for the streaming call.
    assert len(s.tracer.session("s-1")) == 1


def test_user_call_continues_when_tracer_throws(mock_openai_client, monkeypatch):
    response = _chat_response(text="ok")
    mock_openai_client.chat.completions.create.return_value = response
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    monkeypatch.setattr(
        s, "record_call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = mock_openai_client.chat.completions.create(model="gpt-4o", messages=[])
    assert out is response


def test_underlying_chat_exception_propagates(mock_openai_client):
    s = Sentinel(project="proj")
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
    wrap_openai(mock_openai_client, s)
    with pytest.raises(RuntimeError, match="API down"):
        mock_openai_client.chat.completions.create(model="gpt-4o", messages=[])


def test_session_id_kwarg_stripped_from_chat():
    received: dict[str, Any] = {}

    def real_create(**kwargs):
        received.update(kwargs)
        return _chat_response(text="ok")

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=real_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())
    s = Sentinel(project="proj")
    wrap_openai(client, s)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        _sentinel_session_id="abc",
    )
    assert "_sentinel_session_id" not in received


# ---------------------------------------------------------------------------
# Async OpenAI
# ---------------------------------------------------------------------------


def test_async_chat_records_call():
    captured: list[dict[str, Any]] = []

    async def achat_create(**kwargs):
        captured.append(kwargs)
        return _chat_response(text="async ok", prompt_tokens=50, completion_tokens=20)

    async def aembed_create(**kwargs):
        return _embedding_response()

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=achat_create))
    client.embeddings = SimpleNamespace(create=aembed_create)

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    response = asyncio.run(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id="async-1",
        )
    )
    assert response is not None
    assert len(s.tracer.session("async-1")) == 1
    assert "_sentinel_session_id" not in captured[-1]


def test_async_embeddings_record_call():
    async def aembed_create(**kwargs):
        return _embedding_response(prompt_tokens=20)

    async def achat_create(**kwargs):
        return _chat_response()

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=achat_create))
    client.embeddings = SimpleNamespace(create=aembed_create)

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    asyncio.run(
        client.embeddings.create(
            model="text-embedding-3-small",
            input="hi",
            _sentinel_session_id="emb-1",
        )
    )
    rec = s.tracer.session("emb-1")[0]
    assert rec.method == "embeddings.create"
    assert rec.prompt_tokens == 20


def test_async_chat_streaming_returns_proxy():
    """Async ``stream=True`` returns an async-iterable proxy as of stable release."""

    class _AsyncIter:
        def __aiter__(self):
            return self._aiter()

        async def _aiter(self):
            if False:  # empty async iterator
                yield None

    async def achat_create(**kwargs):
        return _AsyncIter()

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=achat_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: None)

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    async def run():
        proxy = await client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="s-1",
        )
        out = []
        async for chunk in proxy:
            out.append(chunk)
        return proxy, out

    proxy, out = asyncio.run(run())
    # Proxy is not the raw async iterator and iteration finalizes the record.
    assert out == []
    assert len(s.tracer.session("s-1")) == 1


# ---------------------------------------------------------------------------
# _build_chat_record / _build_embedding_record unit tests
# ---------------------------------------------------------------------------


def test_build_chat_record_no_choices():
    """Response missing choices yields zero tool_calls and not user-facing."""
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=0),
        choices=[],
    )
    rec = _build_chat_record(
        session_id="s",
        kwargs={"model": "gpt-4o", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert isinstance(rec, CallRecord)
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    assert rec.raw_response_meta == {"finish_reason": None}


def test_build_chat_record_text_only_user_facing():
    response = _chat_response(text="final answer", tool_calls=None)
    rec = _build_chat_record(
        session_id="s",
        kwargs={"model": "gpt-4o", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.user_facing_output is True


def test_build_chat_record_text_and_tool_calls_not_user_facing():
    response = _chat_response(
        text="thinking",
        tool_calls=[{"name": "x", "arguments": "{}"}],
    )
    rec = _build_chat_record(
        session_id="s",
        kwargs={"model": "gpt-4o", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.user_facing_output is False


def test_build_chat_record_missing_usage():
    response = SimpleNamespace(choices=[], usage=None)
    rec = _build_chat_record(
        session_id="s",
        kwargs={"model": "gpt-4o", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


def test_build_chat_record_unknown_model_default():
    response = SimpleNamespace(choices=[], usage=None)
    rec = _build_chat_record(
        session_id="s",
        kwargs={"messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.model == "unknown"


def test_build_chat_record_request_hash_stable():
    response = SimpleNamespace(choices=[], usage=None)
    a = _build_chat_record(
        session_id="s1",
        kwargs={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        latency_ms=1.0,
    )
    b = _build_chat_record(
        session_id="s2",  # different session — irrelevant to the hash
        kwargs={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        latency_ms=99.0,
    )
    assert a.request_hash == b.request_hash


def test_build_embedding_record_basic():
    response = _embedding_response(prompt_tokens=42)
    rec = _build_embedding_record(
        session_id="s",
        kwargs={"model": "text-embedding-3-small", "input": "hello"},
        response=response,
        latency_ms=10.0,
    )
    assert rec.method == "embeddings.create"
    assert rec.provider == "openai"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 0
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    assert rec.raw_request == {"input": "hello", "model": "text-embedding-3-small"}


def test_build_embedding_record_missing_usage():
    response = SimpleNamespace(usage=None)
    rec = _build_embedding_record(
        session_id="s",
        kwargs={"model": "text-embedding-3-small", "input": "hi"},
        response=response,
        latency_ms=10.0,
    )
    assert rec.prompt_tokens == 0


def test_build_embedding_record_default_model():
    response = SimpleNamespace(usage=None)
    rec = _build_embedding_record(
        session_id="s",
        kwargs={"input": "hi"},
        response=response,
        latency_ms=10.0,
    )
    assert rec.model == "unknown"


def test_build_embedding_record_hash_distinguishes_inputs():
    response = SimpleNamespace(usage=None)
    a = _build_embedding_record(
        session_id="s",
        kwargs={"model": "m", "input": "x"},
        response=response,
        latency_ms=1.0,
    )
    b = _build_embedding_record(
        session_id="s",
        kwargs={"model": "m", "input": "y"},
        response=response,
        latency_ms=1.0,
    )
    assert a.request_hash != b.request_hash


# ---------------------------------------------------------------------------
# Whisper (audio.transcriptions.create + audio.translations.create)
# ---------------------------------------------------------------------------
#
# These six tests cover the additive Whisper paths. We confirm:
#   - sync transcriptions / translations record a CallRecord with
#     ``method="audio.transcriptions.create"`` / ``method="audio.translations.create"``
#   - ``usage_extra.dimension_kind == "per_second"`` (Whisper bills per-second
#     of audio, same as Deepgram)
#   - the audio file bytes are stripped from ``raw_request``
#   - async equivalents work (``AsyncOpenAI.audio.transcriptions.create``)
#   - a client without ``audio`` is wrapped without error (defensive)


def _whisper_response(
    *, text: str = "hello world", duration: float | None = None
) -> SimpleNamespace:
    """Build a Whisper-shaped response.

    response_format="json" / "verbose_json" returns a pydantic-shaped object
    with a ``.text`` field. ``verbose_json`` additionally has ``.duration``.
    """
    if duration is None:
        return SimpleNamespace(text=text)
    return SimpleNamespace(text=text, duration=duration)


def _build_openai_client_with_audio(*, async_: bool = False) -> Any:
    """Build a fake OpenAI client that exposes ``client.audio.transcriptions``
    + ``client.audio.translations`` for the Whisper paths.

    Mirrors the ``mock_openai_client`` fixture's class shape (so
    ``Sentinel.wrap`` routes through the OpenAI branch) but adds the
    audio accessor chain. We use plain functions instead of
    ``_RecordingCreate`` so async detection via
    ``inspect.iscoroutinefunction`` flips on for the async case.
    """
    cls_name = "AsyncOpenAI" if async_ else "OpenAI"
    cls = type(cls_name, (), {"__module__": "openai"})
    client = cls()

    if async_:

        async def transcribe(**kwargs):
            return _whisper_response(text="async transcription", duration=12.5)

        async def translate(**kwargs):
            return _whisper_response(text="translated", duration=8.0)

        async def chat_create(**kwargs):
            return _chat_response()

        async def embed_create(**kwargs):
            return _embedding_response()
    else:

        def transcribe(**kwargs):
            return _whisper_response(text="sync transcription", duration=15.5)

        def translate(**kwargs):
            return _whisper_response(text="translated text", duration=10.0)

        def chat_create(**kwargs):
            return _chat_response()

        def embed_create(**kwargs):
            return _embedding_response()

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    client.embeddings = SimpleNamespace(create=embed_create)
    client.audio = SimpleNamespace(
        transcriptions=SimpleNamespace(create=transcribe),
        translations=SimpleNamespace(create=translate),
    )
    return client


def test_audio_transcriptions_records_callrecord():
    """``audio.transcriptions.create`` records a CallRecord."""
    client = _build_openai_client_with_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    response = client.audio.transcriptions.create(
        file=b"\x00\x01\x02",
        model="whisper-1",
        response_format="verbose_json",
        language="en",
        _sentinel_session_id="whisp-1",
    )
    assert response.text == "sync transcription"

    records = s.tracer.session("whisp-1")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    assert rec.provider == "openai"
    assert rec.method == "audio.transcriptions.create"
    assert rec.model == "whisper-1"
    # prompt_tokens=0 (audio in, no token concept); completion_tokens is
    # the char count of the transcribed text.
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == len("sync transcription")
    assert rec.user_facing_output is True


def test_audio_translations_records_callrecord():
    """``audio.translations.create`` records a CallRecord with the right method."""
    client = _build_openai_client_with_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.translations.create(
        file=b"\x00\x01",
        model="whisper-1",
        response_format="verbose_json",
        _sentinel_session_id="tr-1",
    )

    rec = s.tracer.session("tr-1")[0]
    assert rec.method == "audio.translations.create"
    assert rec.model == "whisper-1"
    assert rec.completion_tokens == len("translated text")


def test_audio_usage_extra_per_second_populated():
    """Whisper bills per-second of audio; ``usage_extra.dimension_kind``
    must be ``"per_second"`` and ``dimension_value`` the response duration.

    The duration is read from ``response.duration`` (present only when
    ``response_format="verbose_json"``). ``model_specific_meta`` carries
    ``response_format`` and ``language`` so a future per-language /
    per-response_format rule can dispatch on them.
    """
    client = _build_openai_client_with_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\x00",
        model="whisper-1",
        response_format="verbose_json",
        language="es",
        _sentinel_session_id="dim-1",
    )
    rec = s.tracer.session("dim-1")[0]
    assert rec.usage_extra["dimension_kind"] == "per_second"
    assert rec.usage_extra["dimension_value"] == 15.5
    msm = rec.usage_extra["model_specific_meta"]
    assert msm["response_format"] == "verbose_json"
    assert msm["language"] == "es"


def test_audio_async_transcriptions_records_callrecord():
    """``AsyncOpenAI.audio.transcriptions.create`` (async) records a record."""
    client = _build_openai_client_with_audio(async_=True)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    asyncio.run(
        client.audio.transcriptions.create(
            file=b"\x00\x01",
            model="whisper-1",
            response_format="verbose_json",
            _sentinel_session_id="async-w-1",
        )
    )

    rec = s.tracer.session("async-w-1")[0]
    assert rec.provider == "openai"
    assert rec.method == "audio.transcriptions.create"
    assert rec.model == "whisper-1"
    assert rec.completion_tokens == len("async transcription")
    assert rec.usage_extra["dimension_kind"] == "per_second"
    assert rec.usage_extra["dimension_value"] == 12.5


def test_audio_attribute_missing_does_not_error():
    """A client without ``audio`` is wrapped without raising any error.

    Older openai SDK versions and trimmed mocks may not expose
    ``client.audio`` at all. The wrapper must silently skip the
    Whisper patching in that case so the existing chat/embeddings paths
    continue to work.
    """
    # The standard fixture-style mock has no .audio attribute.
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _chat_response()))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())
    # No client.audio set.

    s = Sentinel(project="proj")
    # No exception expected; the existing chat/embeddings paths are still
    # patched correctly.
    out = wrap_openai(client, s)
    assert out is client
    # Confirm chat still works.
    client.chat.completions.create(model="gpt-4o", messages=[])


def test_audio_raw_request_strips_audio_bytes():
    """The audio ``file`` bytes are stripped from ``raw_request``.

    Customer audio can be multi-MB; shipping it on every CallRecord would
    be a memory + bandwidth disaster. The wrapper replaces the bytes with
    a redaction marker that preserves the byte length for triage.
    """
    audio_bytes = b"\xff" * 2048  # 2 KB of fake audio bytes
    client = _build_openai_client_with_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=audio_bytes,
        model="whisper-1",
        response_format="verbose_json",
        _sentinel_session_id="rd-1",
    )

    rec = s.tracer.session("rd-1")[0]
    file_value = rec.raw_request.get("file", "")
    # Marker form: "<redacted:2048_bytes>"
    assert isinstance(file_value, str)
    assert "redacted" in file_value
    assert "2048" in file_value
    # The actual bytes are NOT in raw_request.
    assert audio_bytes not in file_value.encode()
    # response_format + model are kept (not sensitive).
    assert rec.raw_request["model"] == "whisper-1"
    assert rec.raw_request["response_format"] == "verbose_json"


# ---------------------------------------------------------------------------
# Whisper duration fallback via mutagen-based audio-file probing
# ---------------------------------------------------------------------------
#
# These five tests cover the fallback in ``_probe_audio_duration``.
# The  path reads ``response.duration`` when present
# (``response_format="verbose_json"``). For ``response_format`` in
# ``{"text", "srt", "vtt", "json"}`` the response does NOT carry a
# duration field; the wrapper now falls back to probing the audio file
# itself via mutagen. Tests inject a fake ``mutagen`` module into
# ``sys.modules`` so the wrapper's lazy ``import mutagen`` resolves to
# the fake instead of forcing the real dep into the test matrix.


def _fake_mutagen_module(length: float | None) -> Any:
    """Build a fake ``mutagen`` module whose ``File`` factory returns an
    object with ``.info.length == length`` (or None when ``length`` is
    None).
    """
    module = SimpleNamespace()

    def _file(_target: Any) -> Any:
        if length is None:
            return None
        return SimpleNamespace(info=SimpleNamespace(length=length))

    module.File = _file
    return module


def _install_fake_mutagen(monkeypatch: Any, fake: Any) -> None:
    """Install ``fake`` as the ``mutagen`` module in ``sys.modules``.

    Uses ``monkeypatch.setitem`` so the change is reverted after the
    test, even if it raises.
    """
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "mutagen", fake)


def _uninstall_mutagen(monkeypatch: Any) -> None:
    """Force ``import mutagen`` to raise ImportError inside the wrapper.

    We install a finder that blocks the import so the wrapper's
    ``except ImportError`` branch fires.
    """
    import sys as _sys

    # Drop any real or fake module that was previously installed.
    monkeypatch.delitem(_sys.modules, "mutagen", raising=False)

    class _BlockingFinder:
        @staticmethod
        def find_spec(name: str, _path: Any = None, _target: Any = None) -> Any:
            if name == "mutagen":
                # Raising here surfaces as ImportError at the import site.
                raise ImportError("mutagen not installed")
            return None

    monkeypatch.setattr(_sys, "meta_path", [_BlockingFinder()] + list(_sys.meta_path))


def test_whisper_duration_from_response_duration_field():
    """behavior, untouched: ``response.duration`` populates
    ``usage_extra.dimension_value`` and ``dimension_unavailable`` is NOT
    set when the response field is present.
    """
    client = _build_openai_client_with_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\x00\x01\x02",
        model="whisper-1",
        response_format="verbose_json",
        _sentinel_session_id="v18-ok",
    )

    rec = s.tracer.session("v18-ok")[0]
    assert rec.usage_extra["dimension_value"] == 15.5  # from _whisper_response
    # No dimension_unavailable flag — the response carried duration.
    assert "dimension_unavailable" not in rec.raw_response_meta


def test_whisper_duration_falls_back_to_mutagen_when_response_lacks_it(monkeypatch):
    """When ``response.duration`` is missing (response_format=text), the
    wrapper falls back to ``_probe_audio_duration`` which probes via
    mutagen. Verify the probed duration lands in ``dimension_value`` and
    ``dimension_unavailable`` is NOT set.
    """
    # Install a fake mutagen that returns length=42.0 seconds.
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=42.0))

    # Build a client whose transcribe returns a response WITHOUT
    # duration (response_format="text" doesn't carry it).
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kwargs):
        # Plain string — the response_format="text" shape per the SDK.
        return "hello transcribed text"

    client.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe))
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _chat_response()))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 1024,
        model="whisper-1",
        response_format="text",
        _sentinel_session_id="mutagen-ok",
    )

    rec = s.tracer.session("mutagen-ok")[0]
    # Duration came from mutagen.
    assert rec.usage_extra["dimension_value"] == 42.0
    # dimension_unavailable was cleared because mutagen succeeded.
    assert "dimension_unavailable" not in rec.raw_response_meta


def test_whisper_duration_zero_when_mutagen_unavailable(monkeypatch):
    """If ``mutagen`` raises ImportError, the wrapper degrades silently
    to ``dimension_value=0.0`` and sets ``dimension_unavailable=True``.
    No warning, no exception — the audio-metadata extra is optional.
    """
    _uninstall_mutagen(monkeypatch)

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kwargs):
        return "hello"

    client.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe))
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _chat_response()))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 128,
        model="whisper-1",
        response_format="text",
        _sentinel_session_id="no-mutagen",
    )

    rec = s.tracer.session("no-mutagen")[0]
    assert rec.usage_extra["dimension_value"] == 0.0
    # fallback flag stays set so dashboards can render the gap.
    assert rec.raw_response_meta.get("dimension_unavailable") is True


def test_whisper_duration_zero_when_mutagen_cant_parse(monkeypatch):
    """When mutagen IS installed but ``mutagen.File`` returns None
    (unsupported format / corrupt file), the wrapper degrades to the
     fallback path.
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=None))

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kwargs):
        return "hello"

    client.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe))
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _chat_response()))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 64,
        model="whisper-1",
        response_format="text",
        _sentinel_session_id="bad-mutagen",
    )

    rec = s.tracer.session("bad-mutagen")[0]
    assert rec.usage_extra["dimension_value"] == 0.0
    assert rec.raw_response_meta.get("dimension_unavailable") is True


def test_whisper_dimension_unavailable_flag_still_set_when_zero(monkeypatch):
    """The ``dimension_unavailable`` flag is the  customer-facing
    signal —  customers who don't install the audio-metadata extra
    must still see it on the record for backwards-compatible dashboards.
    """
    _uninstall_mutagen(monkeypatch)

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kwargs):
        # response_format="srt" returns a raw SRT string — no duration.
        return "1\n00:00:00,000 --> 00:00:02,000\nhello\n"

    client.audio = SimpleNamespace(translations=SimpleNamespace(create=transcribe))
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _chat_response()))
    client.embeddings = SimpleNamespace(create=lambda **kw: _embedding_response())

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.translations.create(
        file=b"\xff" * 64,
        model="whisper-1",
        response_format="srt",
        _sentinel_session_id="srt-1",
    )

    rec = s.tracer.session("srt-1")[0]
    # No duration on the response, no mutagen — flag is set, dim is 0.
    assert rec.raw_response_meta.get("dimension_unavailable") is True
    assert rec.usage_extra["dimension_value"] == 0.0
    # The transcribed text IS the SRT string, so completion_tokens is
    # its char count — unchanged from .
    srt_text = "1\n00:00:00,000 --> 00:00:02,000\nhello\n"
    assert rec.completion_tokens == len(srt_text)
