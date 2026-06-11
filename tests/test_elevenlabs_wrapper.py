"""Tests for ``token_sentinel.wrappers.elevenlabs.wrap_elevenlabs``.

NO real ElevenLabs API calls. We construct mocks shaped like the modern
``elevenlabs.client.ElevenLabs`` / ``AsyncElevenLabs`` clients
(``SimpleNamespace`` + recording callables) and verify the wrapper:

  - swaps in instrumented ``convert`` / ``convert_as_stream`` / ``stream``
    on the ``text_to_speech`` accessor
  - delegates to the originals unchanged (return value, kwargs pass-through)
  - builds a ``CallRecord`` matching the  ``usage_extra`` schema with
    ``dimension_kind="per_character"``
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - is dispatched correctly by ``Sentinel.wrap`` (``elevenlabs.client.*``
    detected by module prefix)
  - feeds the future  ``voice_switching_loop`` rule the
    ``voice_id`` + ``text_hash`` shape it needs (captured under
    ``usage_extra.model_specific_meta``)

Per the task spec: ``pytest.importorskip("elevenlabs")`` is used so the
suite quietly skips when the optional dep isn't installed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

# Skip the entire module if the elevenlabs optional dep isn't installed.
pytest.importorskip("elevenlabs")

from token_sentinel import Sentinel  # noqa: E402
from token_sentinel.events import CallRecord  # noqa: E402
from token_sentinel.wrappers.elevenlabs import (  # noqa: E402
    METHOD_TTS_CONVERT,
    METHOD_TTS_CONVERT_AS_STREAM,
    METHOD_TTS_STREAM,
    _build_record,
    _extract_text,
    _extract_voice_id,
    wrap_elevenlabs,
)

# ---------------------------------------------------------------------------
# Mock SDK shapes
# ---------------------------------------------------------------------------


class _RecordingCallable:
    """Real callable that records calls and returns a configurable response.

    Mirrors the test helpers in ``test_deepgram_wrapper.py``. We use a real
    class instead of MagicMock because ``functools.wraps`` (used inside
    ``wrap_elevenlabs``) needs ``__name__`` / ``__qualname__`` to be strings.
    """

    __name__ = "convert"
    __qualname__ = "TextToSpeech.convert"
    __module__ = "elevenlabs.text_to_speech.client"
    __annotations__: dict = {}
    __doc__ = "mock convert"

    def __init__(self, name: str = "convert"):
        self.__name__ = name
        self.__qualname__ = f"TextToSpeech.{name}"
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value: Any = None
        self.side_effect: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value


class _AsyncRecordingCallable:
    """Async counterpart of ``_RecordingCallable``.

    ``inspect.iscoroutinefunction`` is the wrapper's async detection hook.
    Defining ``__call__`` as ``async def`` is what flips that bit.
    """

    __name__ = "convert"
    __qualname__ = "AsyncTextToSpeech.convert"
    __module__ = "elevenlabs.text_to_speech.client"
    __annotations__: dict = {}
    __doc__ = "mock async convert"

    def __init__(self, name: str = "convert"):
        self.__name__ = name
        self.__qualname__ = f"AsyncTextToSpeech.{name}"
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value: Any = None
        self.side_effect: Any = None

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value


def _bytes_iter(chunks: list[bytes] | None = None) -> Any:
    """Build an ``Iterator[bytes]`` matching ElevenLabs' return shape.

    Implemented as a generator function so the result satisfies the
    ``__iter__`` + ``__next__`` protocol — same as the real SDK's
    response (the SDK lazily yields decoded audio chunks).
    """
    chunks = chunks or [b"\x00\x01", b"\x02\x03", b"\x04"]

    def _gen():
        yield from chunks

    return _gen()


async def _async_bytes_iter(chunks: list[bytes] | None = None) -> Any:
    """Async iterator of bytes, matching newer SDK async streaming."""
    chunks = chunks or [b"\x00\x01", b"\x02\x03"]
    for c in chunks:
        yield c


def _make_elevenlabs_client(
    *,
    is_async: bool = False,
    convert_return: Any = None,
    convert_as_stream_return: Any = None,
    stream_return: Any = None,
    include_stream: bool = True,
) -> Any:
    """Build a mock shaped like ``elevenlabs.client.ElevenLabs``.

    The class name reports ``ElevenLabs`` (or ``AsyncElevenLabs``) and its
    ``__module__`` starts with ``elevenlabs.client`` so ``Sentinel.wrap``'s
    dispatch logic routes us to ``wrap_elevenlabs``.
    """
    cls_name = "AsyncElevenLabs" if is_async else "ElevenLabs"
    cls = type(cls_name, (), {"__module__": "elevenlabs.client"})
    client = cls()

    if is_async:
        convert = _AsyncRecordingCallable("convert")
        convert_as_stream = _AsyncRecordingCallable("convert_as_stream")
        stream = _AsyncRecordingCallable("stream") if include_stream else None
    else:
        convert = _RecordingCallable("convert")
        convert_as_stream = _RecordingCallable("convert_as_stream")
        stream = _RecordingCallable("stream") if include_stream else None

    convert.return_value = convert_return if convert_return is not None else _bytes_iter()
    convert_as_stream.return_value = (
        convert_as_stream_return if convert_as_stream_return is not None else _bytes_iter()
    )
    if stream is not None:
        stream.return_value = stream_return if stream_return is not None else _bytes_iter()

    tts = SimpleNamespace(convert=convert, convert_as_stream=convert_as_stream)
    if stream is not None:
        tts.stream = stream
    client.text_to_speech = tts
    return client


# ---------------------------------------------------------------------------
# 0. Smoke check: Sentinel.wrap(real ElevenLabs client) returns the wrapped
# ---------------------------------------------------------------------------


def test_sentinel_wrap_real_elevenlabs_returns_wrapped_client():
    """``Sentinel.wrap(elevenlabs.client.ElevenLabs(...))`` returns the wrapped
    client (in-place mutation; returned object is identical).

    The real ``ElevenLabs(api_key=...)`` constructor doesn't make any HTTP
    calls; it just stores the key. We can construct one without secrets and
    inspect that ``text_to_speech.convert`` was replaced.
    """
    import elevenlabs.client

    s = Sentinel(project="proj")
    real_client = elevenlabs.client.ElevenLabs(api_key="fake-key-not-used")
    original_convert = real_client.text_to_speech.convert
    out = s.wrap(real_client)
    assert out is real_client
    # The convert method was mutated in place.
    assert real_client.text_to_speech.convert is not original_convert


# ---------------------------------------------------------------------------
# 1. Sentinel.wrap dispatch — ElevenLabs detection via module prefix
# ---------------------------------------------------------------------------


def test_wrap_elevenlabs_client_detection():
    """``Sentinel.wrap`` routes an ElevenLabs mock to ``wrap_elevenlabs``."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    out = s.wrap(client)
    assert out is client
    # convert was replaced with our instrumented version.
    assert client.text_to_speech.convert.__name__ in {"convert", "instrumented"}


# ---------------------------------------------------------------------------
# 2. Sync convert records CallRecord on iterator exhaustion
# ---------------------------------------------------------------------------


def test_convert_records_callrecord_on_iterator_exhaustion():
    """A sync convert call returns an iterator + records on consumption.

    The CallRecord is NOT emitted until the iterator is exhausted — the
    API call is logically not complete until all bytes have streamed.
    """
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert(
        voice_id="voice-abc",
        text="Hello world, this is a test.",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        _sentinel_session_id="el-1",
    )

    # Before iteration: NO record yet.
    assert len(s.tracer.session("el-1")) == 0

    # Iterate to exhaustion.
    chunks = list(result)
    assert chunks  # got some bytes

    # Record is emitted.
    records = s.tracer.session("el-1")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    assert rec.provider == "elevenlabs"
    assert rec.method == METHOD_TTS_CONVERT
    assert rec.model == "voice-abc"
    # Char-count proxy for prompt_tokens.
    assert rec.prompt_tokens == len("Hello world, this is a test.")
    assert rec.completion_tokens == 0
    assert rec.user_facing_output is True


# ---------------------------------------------------------------------------
# 3. convert_as_stream records CallRecord
# ---------------------------------------------------------------------------


def test_convert_as_stream_records_callrecord():
    """``convert_as_stream`` records a CallRecord with the right method."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert_as_stream(
        voice_id="voice-stream",
        text="streamed text",
        _sentinel_session_id="el-stream-1",
    )
    list(result)

    records = s.tracer.session("el-stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == METHOD_TTS_CONVERT_AS_STREAM
    assert rec.model == "voice-stream"


# ---------------------------------------------------------------------------
# 4. text_to_speech.stream method (newer SDK shape) is also instrumented
# ---------------------------------------------------------------------------


def test_stream_method_records_callrecord():
    """``text_to_speech.stream`` (V2.x+ method name) is also instrumented.

    The wrapper defensively checks each method name in ``_TTS_METHODS`` —
    SDK versions that lack ``stream`` skip it silently; SDK versions
    that have it get it patched alongside ``convert`` /
    ``convert_as_stream``.
    """
    client = _make_elevenlabs_client(include_stream=True)
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.stream(
        voice_id="voice-v2",
        text="newer SDK",
        _sentinel_session_id="el-v2-1",
    )
    list(result)

    records = s.tracer.session("el-v2-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == METHOD_TTS_STREAM
    assert rec.model == "voice-v2"


# ---------------------------------------------------------------------------
# 5. Async convert records CallRecord
# ---------------------------------------------------------------------------


def test_async_convert_records_callrecord():
    """``AsyncElevenLabs.text_to_speech.convert`` (async) records a record.

    Modern (v2.x+) async SDKs return an awaited iterator. Our v1.0-style
    mock returns ``Iterator[bytes]`` directly from an async call — the
    wrapper accepts both shapes because ``_is_byte_iterator`` is the
    same on the awaited result.
    """
    client = _make_elevenlabs_client(is_async=True)
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    async def run() -> Any:
        result = await client.text_to_speech.convert(
            voice_id="voice-async",
            text="async TTS",
            _sentinel_session_id="el-async-1",
        )
        # ``result`` is an iterator of bytes; iterating triggers finalize.
        return list(result)

    chunks = asyncio.run(run())
    assert chunks

    records = s.tracer.session("el-async-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "elevenlabs"
    assert rec.method == METHOD_TTS_CONVERT
    assert rec.model == "voice-async"
    assert rec.prompt_tokens == len("async TTS")


# ---------------------------------------------------------------------------
# 6. Async convert returning an async iterator of bytes records on aiter
# ---------------------------------------------------------------------------


def test_async_convert_with_async_iterator_records_on_aiter_exhaustion():
    """When the async SDK returns an async iterator, the proxy aiter triggers
    finalize on stream exhaustion."""
    client = _make_elevenlabs_client(is_async=True)
    # Override return value to be an async iterator (newer SDK shape).
    client.text_to_speech.stream.return_value = _async_bytes_iter()

    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    async def run() -> Any:
        proxy = await client.text_to_speech.stream(
            voice_id="voice-aiter",
            text="aiter path",
            _sentinel_session_id="el-aiter-1",
        )
        out = []
        async for chunk in proxy:
            out.append(chunk)
        return out

    chunks = asyncio.run(run())
    assert chunks

    records = s.tracer.session("el-aiter-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == METHOD_TTS_STREAM
    assert rec.model == "voice-aiter"


# ---------------------------------------------------------------------------
# 7. Char-count is prompt_tokens
# ---------------------------------------------------------------------------


def test_char_count_is_prompt_tokens():
    """``prompt_tokens`` on TTS records equals the input text's char count.

    ElevenLabs bills per-character of input text, so char count is the
    natural prompt_tokens proxy — same convention as Voyage's embed inputs.
    """
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    expected = len(text)
    result = client.text_to_speech.convert(voice_id="vx", text=text, _sentinel_session_id="cc-1")
    list(result)

    rec = s.tracer.session("cc-1")[0]
    assert rec.prompt_tokens == expected


# ---------------------------------------------------------------------------
# 8. usage_extra populated with voice_id + model_id meta + per_character
# ---------------------------------------------------------------------------


def test_usage_extra_per_character_dimension_kind():
    """``CallRecord.usage_extra`` carries ElevenLabs' per-character dimension.

    The  ``usage_extra`` schema's ``dimension_kind`` is the cloud-side
    cost estimator's dispatch key (``NON_TOKEN_PRICES``). For ElevenLabs
    that key is ``"per_character"``.
    """
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    text = "exactly 26 chars in this str"
    result = client.text_to_speech.convert(
        voice_id="v-1",
        text=text,
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128",
        _sentinel_session_id="dim-1",
    )
    list(result)

    rec = s.tracer.session("dim-1")[0]
    assert rec.usage_extra["dimension_kind"] == "per_character"
    assert rec.usage_extra["dimension_value"] == len(text)
    msm = rec.usage_extra["model_specific_meta"]
    assert msm["voice_id"] == "v-1"
    assert msm["model_id"] == "eleven_turbo_v2"
    assert msm["output_format"] == "mp3_44100_128"
    # text_hash is present (voice_switching_loop telemetry).
    assert isinstance(msm["text_hash"], str)
    assert len(msm["text_hash"]) > 0


# ---------------------------------------------------------------------------
# 9. Failure isolation
# ---------------------------------------------------------------------------


def test_failure_in_instrumentation_does_not_break_user_call(monkeypatch):
    """A bug inside ``Sentinel.record_call`` must not crash the user's call.

    The wrapper's two-level safety boundary wraps record_call in
    try/except LeakDetected: raise / except Exception: pass.
    """
    sentinel_response = _bytes_iter([b"audio"])
    client = _make_elevenlabs_client(convert_return=sentinel_response)
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    def boom(*_args, **_kwargs):
        raise RuntimeError("instrumentation bug")

    monkeypatch.setattr(s, "record_call", boom)

    # The user's call returns the wrapped iterator unchanged behavior —
    # iterate it and confirm we get the expected bytes.
    result = client.text_to_speech.convert(voice_id="vx", text="x", _sentinel_session_id="brk-1")
    chunks = list(result)
    assert chunks == [b"audio"]


# ---------------------------------------------------------------------------
# 10. voice_id surfaces as the .model field on the CallRecord
# ---------------------------------------------------------------------------


def test_voice_id_surfaces_as_model_field():
    """The voice_id becomes ``CallRecord.model`` (per-voice cost dashboards).

    ElevenLabs doesn't have a single "model" concept the way OpenAI does —
    ``model_id`` selects the synthesis engine, ``voice_id`` selects the
    voice. The voice is the more meaningful billing target for customer
    dashboards ("which voice am I burning the most characters on?"), so
    we surface voice_id as the .model field. The model_id stays available
    on ``usage_extra.model_specific_meta`` for rules that need it.
    """
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    for voice_id in ("rachel", "adam", "bella", "antoni"):
        result = client.text_to_speech.convert(
            voice_id=voice_id,
            text="hello",
            _sentinel_session_id=f"v-{voice_id}",
        )
        list(result)
        rec = s.tracer.session(f"v-{voice_id}")[0]
        assert rec.model == voice_id


# ---------------------------------------------------------------------------
# 11. raw_request strips text content
# ---------------------------------------------------------------------------


def test_raw_request_strips_text_content():
    """``raw_request`` MUST NOT carry the literal text — strip to a marker.

    Customer text may be sensitive (emails, financial summaries, medical
    content). The wrapper replaces the literal text with a redaction
    marker that preserves char count for triage.
    """
    sensitive = "SSN 123-45-6789 wire transfer to..."
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert(
        voice_id="vx",
        text=sensitive,
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128",
        _sentinel_session_id="rd-1",
    )
    list(result)

    rec = s.tracer.session("rd-1")[0]
    raw_req_text = rec.raw_request.get("text", "")
    # Marker form preserves char count for triage.
    assert isinstance(raw_req_text, str)
    assert "redacted" in raw_req_text
    assert str(len(sensitive)) in raw_req_text
    # The actual text is NOT in raw_request.
    assert sensitive not in raw_req_text
    # voice_id + model_id + output_format ARE kept (no PII risk on those).
    assert rec.raw_request["voice_id"] == "vx"
    assert rec.raw_request["model_id"] == "eleven_turbo_v2"
    assert rec.raw_request["output_format"] == "mp3_44100_128"


# ---------------------------------------------------------------------------
# 12. Provider field set to "elevenlabs"
# ---------------------------------------------------------------------------


def test_provider_field_set_to_elevenlabs():
    """Every record emitted by the ElevenLabs wrapper carries provider="elevenlabs"."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert(voice_id="vx", text="x", _sentinel_session_id="prov-1")
    list(result)

    rec = s.tracer.session("prov-1")[0]
    assert rec.provider == "elevenlabs"


# ---------------------------------------------------------------------------
# 13. Method field distinguishes the three TTS surfaces
# ---------------------------------------------------------------------------


def test_method_field_distinguishes_tts_paths():
    """The three TTS entry points produce three distinct ``method`` strings.

    The constants ``METHOD_TTS_CONVERT`` / ``METHOD_TTS_CONVERT_AS_STREAM`` /
    ``METHOD_TTS_STREAM`` are imported here so a typo in the wrapper would
    surface as a clear failure. The future ``voice_switching_loop`` rule
    keys on the ``provider == "elevenlabs"`` check rather than the method
    label, so all three are equivalent from a rule standpoint.
    """
    client = _make_elevenlabs_client(include_stream=True)
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    for method_name, expected in [
        ("convert", METHOD_TTS_CONVERT),
        ("convert_as_stream", METHOD_TTS_CONVERT_AS_STREAM),
        ("stream", METHOD_TTS_STREAM),
    ]:
        bound = getattr(client.text_to_speech, method_name)
        result = bound(
            voice_id="vx",
            text="text",
            _sentinel_session_id=f"m-{method_name}",
        )
        list(result)
        rec = s.tracer.session(f"m-{method_name}")[0]
        assert rec.method == expected

    # Sanity: the three constants are all distinct.
    assert len({METHOD_TTS_CONVERT, METHOD_TTS_CONVERT_AS_STREAM, METHOD_TTS_STREAM}) == 3


# ---------------------------------------------------------------------------
# 14. Defensive: SDK missing ``text_to_speech.stream`` doesn't crash wrap
# ---------------------------------------------------------------------------


def test_missing_stream_method_does_not_crash_wrap():
    """SDK versions without ``text_to_speech.stream`` (v1.x) still wrap cleanly.

    The wrapper's ``_TTS_METHODS`` iteration is defensive — missing
    attributes are skipped silently, so the same wrapper code works for
    v1.x SDKs (no ``stream``) and v2.x+ SDKs (with ``stream``).
    """
    client = _make_elevenlabs_client(include_stream=False)
    # Sanity check: no .stream attribute on this mock.
    assert not hasattr(client.text_to_speech, "stream")
    s = Sentinel(project="proj")
    out = wrap_elevenlabs(client, s)
    assert out is client
    # ``convert`` is still patched.
    result = client.text_to_speech.convert(
        voice_id="vx", text="t", _sentinel_session_id="no-stream-1"
    )
    list(result)
    assert len(s.tracer.session("no-stream-1")) == 1


# ---------------------------------------------------------------------------
# 15. text_hash is stable for the same input ( rule telemetry)
# ---------------------------------------------------------------------------


def test_text_hash_stable_for_same_input():
    """The  ``voice_switching_loop`` rule keys on ``text_hash`` matching
    across calls with different ``voice_id`` values. The hash must be
    deterministic given the same input text — different voice_ids must NOT
    affect the text_hash."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    text = "Hello, world."
    for i, voice_id in enumerate(["voice-a", "voice-b", "voice-c"]):
        result = client.text_to_speech.convert(
            voice_id=voice_id, text=text, _sentinel_session_id=f"vsl-{i}"
        )
        list(result)

    hashes = []
    voices = []
    for i in range(3):
        rec = s.tracer.session(f"vsl-{i}")[0]
        hashes.append(rec.usage_extra["model_specific_meta"]["text_hash"])
        voices.append(rec.usage_extra["model_specific_meta"]["voice_id"])

    assert len(set(hashes)) == 1, "text_hash must be stable for the same input"
    # voice_ids must NOT be stable — they're what changes in a switching loop.
    assert len(set(voices)) == 3


# ---------------------------------------------------------------------------
# 16. Latency is captured
# ---------------------------------------------------------------------------


def test_latency_captured():
    """CallRecord.latency_ms is a positive float covering full iter time."""
    import time as _time

    def slow_chunks():
        yield b"x"
        _time.sleep(0.015)  # 15ms
        yield b"y"

    client = _make_elevenlabs_client(convert_return=slow_chunks())
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert(voice_id="vx", text="hi", _sentinel_session_id="lat")
    list(result)
    rec = s.tracer.session("lat")[0]
    assert isinstance(rec.latency_ms, float)
    # Allow scheduler slack but confirm > 0.
    assert rec.latency_ms >= 10.0
    assert rec.latency_ms < 5000.0


# ---------------------------------------------------------------------------
# 17. Empty text input handled (edge case)
# ---------------------------------------------------------------------------


def test_empty_text_input_handled():
    """An empty-text call produces a CallRecord with prompt_tokens=0
    rather than crashing the wrapper."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    result = client.text_to_speech.convert(voice_id="vx", text="", _sentinel_session_id="empty")
    list(result)

    rec = s.tracer.session("empty")[0]
    assert rec.prompt_tokens == 0
    assert rec.usage_extra["dimension_value"] == 0


# ---------------------------------------------------------------------------
# 18. Positional voice_id is handled
# ---------------------------------------------------------------------------


def test_positional_voice_id_handled():
    """``convert("voice-id", text=...)`` — voice_id as positional argument."""
    client = _make_elevenlabs_client()
    s = Sentinel(project="proj")
    wrap_elevenlabs(client, s)

    # voice_id positional, everything else kwargs.
    result = client.text_to_speech.convert(
        "positional-voice",
        text="hi",
        _sentinel_session_id="pos-1",
    )
    list(result)

    rec = s.tracer.session("pos-1")[0]
    assert rec.model == "positional-voice"
    assert rec.usage_extra["model_specific_meta"]["voice_id"] == "positional-voice"


# ---------------------------------------------------------------------------
# Small unit tests on the extractor helpers — not counted in the 18 but
# cheap insurance on the positional/kwarg argument code path.
# ---------------------------------------------------------------------------


def test_extract_voice_id_positional():
    """voice_id as args[0]; kwargs absence falls through to positional."""
    assert _extract_voice_id(("vp",), {}) == "vp"
    assert _extract_voice_id(("ignored",), {"voice_id": "kv"}) == "kv"
    assert _extract_voice_id((), {}) == ""


def test_extract_text_handles_missing():
    """Missing text returns empty string, not None."""
    assert _extract_text((), {}) == ""
    assert _extract_text((), {"text": "hello"}) == "hello"


def test_build_record_unknown_voice():
    """Missing voice_id yields model='unknown' rather than crashing."""
    rec = _build_record(
        session_id="s",
        args=(),
        kwargs={"text": "x"},
        method_label=METHOD_TTS_CONVERT,
        latency_ms=1.0,
    )
    assert rec.model == "unknown"
    assert rec.provider == "elevenlabs"
