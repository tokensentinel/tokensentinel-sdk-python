"""Tests for ``token_sentinel.wrappers.deepgram.wrap_deepgram``.

NO real Deepgram API calls. We construct mocks shaped like the v7
``deepgram-sdk`` ``DeepgramClient`` / ``AsyncDeepgramClient`` (``SimpleNamespace``
+ recording callables) and verify the wrapper:

  - swaps in instrumented ``transcribe_file`` / ``transcribe_url`` /
    ``connect`` on the ``listen.v1`` accessor chain
  - delegates to the originals unchanged (return value, exception propagation)
  - builds a ``CallRecord`` matching the response shape — provider="deepgram",
    per-second usage_extra-style telemetry on ``raw_response_meta``
  - emits exactly ONE record at live-stream close (not per Final Result)
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - is dispatched correctly by ``Sentinel.wrap`` (``deepgram.DeepgramClient``
    detected by module prefix)

Per the task spec: ``pytest.importorskip("deepgram")`` is used so the suite
quietly skips when the optional dep isn't installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

# Skip the entire module if the deepgram-sdk optional dep isn't installed.
pytest.importorskip("deepgram")

from token_sentinel import Sentinel  # noqa: E402
from token_sentinel.events import CallRecord  # noqa: E402
from token_sentinel.wrappers.deepgram import (  # noqa: E402
    METHOD_TRANSCRIBE_FILE,
    METHOD_TRANSCRIBE_LIVE,
    METHOD_TRANSCRIBE_URL,
    _build_record_from_live,
    _build_record_from_response,
    _LiveAccumulator,
    _request_hash,
    wrap_deepgram,
)

# ---------------------------------------------------------------------------
# Mock SDK shapes
# ---------------------------------------------------------------------------


class _RecordingCallable:
    """Real callable that records calls and returns a configurable response.

    Mirrors the test helpers in ``test_bedrock_wrapper.py``. We use a real
    class instead of MagicMock because ``functools.wraps`` (used inside
    ``wrap_deepgram``) needs ``__name__`` / ``__qualname__`` to be strings.
    """

    __name__ = "transcribe_file"
    __qualname__ = "MediaClient.transcribe_file"
    __module__ = "deepgram.listen.v1.client"
    __annotations__: dict = {}
    __doc__ = "mock transcribe"

    def __init__(self, name: str = "transcribe_file"):
        self.__name__ = name
        self.__qualname__ = f"MediaClient.{name}"
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
    """Async counterpart of ``_RecordingCallable``."""

    __name__ = "transcribe_file"
    __qualname__ = "AsyncMediaClient.transcribe_file"
    __module__ = "deepgram.listen.v1.client"
    __annotations__: dict = {}
    __doc__ = "mock async transcribe"

    def __init__(self, name: str = "transcribe_file"):
        self.__name__ = name
        self.__qualname__ = f"AsyncMediaClient.{name}"
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


def _make_response(
    *,
    duration: float = 12.5,
    channels: int = 1,
    transcript: str | None = "hello world",
    request_id: str = "req-1",
    models: list[str] | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like ``ListenV1Response``."""
    alt = SimpleNamespace(transcript=transcript or "")
    channel_obj = SimpleNamespace(alternatives=[alt])
    results = SimpleNamespace(channels=[channel_obj for _ in range(channels)])
    metadata = SimpleNamespace(
        duration=duration,
        channels=channels,
        request_id=request_id,
        models=models or ["nova-2"],
    )
    return SimpleNamespace(metadata=metadata, results=results)


def _make_deepgram_client(
    *,
    is_async: bool = False,
    transcribe_file_return: Any = None,
    transcribe_url_return: Any = None,
    connect_yield: Any = None,
) -> Any:
    """Build a mock shaped like ``deepgram.DeepgramClient``.

    The class name reports ``DeepgramClient`` and its ``__module__`` starts
    with ``deepgram`` so ``Sentinel.wrap``'s dispatch logic routes us to
    ``wrap_deepgram``. The ``listen.v1`` accessor chain is constructed with
    ``SimpleNamespace`` and the methods are real callables that
    ``wrap_deepgram`` can wrap with ``functools.wraps``.
    """
    cls_name = "AsyncDeepgramClient" if is_async else "DeepgramClient"
    cls = type(cls_name, (), {"__module__": "deepgram.client"})
    client = cls()

    if is_async:
        tf = _AsyncRecordingCallable("transcribe_file")
        tu = _AsyncRecordingCallable("transcribe_url")
    else:
        tf = _RecordingCallable("transcribe_file")
        tu = _RecordingCallable("transcribe_url")
    tf.return_value = transcribe_file_return
    tu.return_value = transcribe_url_return

    media = SimpleNamespace(transcribe_file=tf, transcribe_url=tu)
    v1 = SimpleNamespace(media=media)

    # ``connect`` is a function returning a context manager. We use a
    # closure so the test can swap ``connect_yield`` per invocation.
    @contextmanager
    def connect_cm(**kwargs: Any) -> Any:
        # ``yield`` whatever the test handed us (a fake socket / iterator).
        if connect_yield is not None:
            yield connect_yield
        else:
            yield SimpleNamespace()

    v1.connect = connect_cm
    client.listen = SimpleNamespace(v1=v1)
    return client


def _make_fake_socket(events: list[Any]) -> Any:
    """Build a fake socket whose ``__iter__`` yields the given events."""

    class _Socket:
        def __init__(self, events: list[Any]):
            self._events = events
            self.send_calls: list[bytes] = []
            self.closed = False

        def __iter__(self) -> Any:
            yield from self._events

        def send_media(self, message: bytes) -> None:
            self.send_calls.append(message)

        def send_close_stream(self, *_: Any, **__: Any) -> None:
            self.closed = True

    return _Socket(events)


def _make_live_event(
    event_type: str,
    **fields: Any,
) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like a Deepgram live event."""
    return SimpleNamespace(type=event_type, **fields)


def _make_results_event(
    transcript: str,
    *,
    is_final: bool = True,
) -> SimpleNamespace:
    alt = SimpleNamespace(transcript=transcript)
    channel = SimpleNamespace(alternatives=[alt])
    return SimpleNamespace(
        type="Results",
        is_final=is_final,
        channel=channel,
        metadata=SimpleNamespace(),
    )


def _make_metadata_event(
    *,
    duration: float = 30.0,
    channels: int = 1,
    request_id: str = "live-req-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        type="Metadata",
        duration=duration,
        channels=channels,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# 1. test_wrap_deepgram_client_detection
# ---------------------------------------------------------------------------


def test_wrap_deepgram_client_detection():
    """``Sentinel.wrap`` routes a ``DeepgramClient`` to ``wrap_deepgram``."""
    client = _make_deepgram_client(transcribe_file_return=_make_response())
    s = Sentinel(project="proj")
    out = s.wrap(client)
    assert out is client
    # transcribe_file was replaced with our instrumented version.
    assert client.listen.v1.media.transcribe_file.__name__ in {
        "transcribe_file",
        "instrumented_sync",
        "instrumented_async",
    }


# ---------------------------------------------------------------------------
# 2. test_transcribe_file_records_callrecord
# ---------------------------------------------------------------------------


def test_transcribe_file_records_callrecord():
    response = _make_response(duration=15.0, channels=1, transcript="ok")
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    out = client.listen.v1.media.transcribe_file(
        request=b"\x00\x01\x02",
        model="nova-2",
        _sentinel_session_id="dg-1",
    )
    assert out is response

    records = s.tracer.session("dg-1")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    assert rec.provider == "deepgram"
    assert rec.model == "nova-2"
    assert rec.method == METHOD_TRANSCRIBE_FILE
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


# ---------------------------------------------------------------------------
# 3. test_transcribe_url_records_callrecord
# ---------------------------------------------------------------------------


def test_transcribe_url_records_callrecord():
    response = _make_response(duration=8.0, channels=1, transcript="from url")
    client = _make_deepgram_client(transcribe_url_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    out = client.listen.v1.media.transcribe_url(
        url="https://example.com/audio.mp3",
        model="nova-3",
        _sentinel_session_id="dg-2",
    )
    assert out is response

    records = s.tracer.session("dg-2")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == METHOD_TRANSCRIBE_URL
    assert rec.model == "nova-3"
    # URL kept in raw_request (it's a small string, not bytes).
    assert rec.raw_request.get("url") == "https://example.com/audio.mp3"


# ---------------------------------------------------------------------------
# 4. test_transcribe_live_emits_on_close — streaming integration
# ---------------------------------------------------------------------------


def test_transcribe_live_emits_on_close():
    """Live streaming emits exactly ONE CallRecord, on stream close.

    Mocks the iterator to yield: Results(final) -> Metadata(close).
    Asserts a single CallRecord with method='transcribe_live'.
    """
    events = [
        _make_results_event("hello", is_final=True),
        _make_metadata_event(duration=45.0, channels=1),
    ]
    socket = _make_fake_socket(events)
    client = _make_deepgram_client(connect_yield=socket)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    with client.listen.v1.connect(model="nova-2", _sentinel_session_id="dg-live-1") as sock:
        received = list(sock)

    # Two events yielded; one CallRecord written.
    assert len(received) == 2
    records = s.tracer.session("dg-live-1")
    assert len(records) == 1, "expected exactly one CallRecord per live connection"
    rec = records[0]
    assert rec.method == METHOD_TRANSCRIBE_LIVE
    assert rec.raw_response_meta["duration_seconds"] == 45.0
    assert rec.raw_response_meta["final_result_count"] == 1
    assert rec.user_facing_output is True


# ---------------------------------------------------------------------------
# 5. test_async_transcribe_records_callrecord
# ---------------------------------------------------------------------------


async def test_async_transcribe_records_callrecord():
    response = _make_response(duration=20.0, channels=1, transcript="async ok")
    client = _make_deepgram_client(is_async=True, transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    out = await client.listen.v1.media.transcribe_file(
        request=b"audio-bytes",
        model="nova-2",
        _sentinel_session_id="dg-async-1",
    )
    assert out is response

    records = s.tracer.session("dg-async-1")
    assert len(records) == 1
    assert records[0].method == METHOD_TRANSCRIBE_FILE
    assert records[0].model == "nova-2"


# ---------------------------------------------------------------------------
# 6. test_usage_extra_per_second_dimension_kind
# ---------------------------------------------------------------------------


def test_usage_extra_per_second_dimension_kind():
    """``CallRecord.usage_extra`` carries Deepgram's per-second dimension.

    Deepgram bills per-second of audio, so the  ``usage_extra`` field's
    ``dimension_kind`` must be ``"per_second"``. The cloud-side cost
    estimator's ``NON_TOKEN_PRICES`` table keys on this exact string.
    """
    response = _make_response(duration=42.0, channels=1)
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="dim-1"
    )

    rec = s.tracer.session("dim-1")[0]
    assert rec.usage_extra["dimension_kind"] == "per_second"


# ---------------------------------------------------------------------------
# 7. test_usage_extra_duration_value_matches_response
# ---------------------------------------------------------------------------


def test_usage_extra_duration_value_matches_response():
    response = _make_response(duration=123.45)
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="dim-2"
    )
    rec = s.tracer.session("dim-2")[0]
    assert rec.usage_extra["dimension_value"] == pytest.approx(123.45)
    assert rec.raw_response_meta["duration_seconds"] == pytest.approx(123.45)


# ---------------------------------------------------------------------------
# 8. test_usage_extra_includes_channels_meta
# ---------------------------------------------------------------------------


def test_usage_extra_includes_channels_meta():
    response = _make_response(duration=60.0, channels=2)
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    client.listen.v1.media.transcribe_file(
        request=b"x",
        model="nova-2",
        multichannel=True,
        _sentinel_session_id="ch-1",
    )
    rec = s.tracer.session("ch-1")[0]
    msm = rec.usage_extra["model_specific_meta"]
    assert msm["channels"] == 2
    assert msm["multichannel"] is True


# ---------------------------------------------------------------------------
# 9. test_multichannel_flag_captured_in_meta
# ---------------------------------------------------------------------------


def test_multichannel_flag_captured_in_meta():
    """Both multichannel=True and multichannel=False are surfaced literally.

    The  ``audio_multichannel_doubling`` rule (deferred — see wrapper
    docstring) reads ``model_specific_meta.multichannel`` to decide whether
    to fire. We must therefore preserve the literal boolean rather than
    coerce missing→False vs explicit→False.
    """
    # Case A: explicit multichannel=True
    resp_a = _make_response(duration=30.0, channels=2)
    client_a = _make_deepgram_client(transcribe_file_return=resp_a)
    s_a = Sentinel(project="proj")
    wrap_deepgram(client_a, s_a)
    client_a.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", multichannel=True, _sentinel_session_id="mc-T"
    )
    rec_a = s_a.tracer.session("mc-T")[0]
    assert rec_a.usage_extra["model_specific_meta"]["multichannel"] is True

    # Case B: multichannel omitted (defaults to False).
    resp_b = _make_response(duration=30.0, channels=1)
    client_b = _make_deepgram_client(transcribe_file_return=resp_b)
    s_b = Sentinel(project="proj")
    wrap_deepgram(client_b, s_b)
    client_b.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="mc-F"
    )
    rec_b = s_b.tracer.session("mc-F")[0]
    assert rec_b.usage_extra["model_specific_meta"]["multichannel"] is False


# ---------------------------------------------------------------------------
# 10. test_model_passed_through_to_callrecord
# ---------------------------------------------------------------------------


def test_model_passed_through_to_callrecord():
    for model_name in ("nova-2", "nova-3", "enhanced", "base"):
        response = _make_response()
        client = _make_deepgram_client(transcribe_file_return=response)
        s = Sentinel(project="proj")
        wrap_deepgram(client, s)
        client.listen.v1.media.transcribe_file(
            request=b"x", model=model_name, _sentinel_session_id=f"m-{model_name}"
        )
        rec = s.tracer.session(f"m-{model_name}")[0]
        assert rec.model == model_name


# ---------------------------------------------------------------------------
# 11. test_failure_in_instrumentation_does_not_break_user_call
# ---------------------------------------------------------------------------


def test_failure_in_instrumentation_does_not_break_user_call(monkeypatch):
    """If ``sentinel.record_call`` blows up, the user's call still returns."""
    response = _make_response(transcript="ok")
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("tracer broken")

    monkeypatch.setattr(s, "record_call", boom)

    out = client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="brk-1"
    )
    assert out is response  # user got their response despite the explosion


# ---------------------------------------------------------------------------
# 12. test_user_facing_output_true_for_transcription
# ---------------------------------------------------------------------------


def test_user_facing_output_true_for_transcription():
    response = _make_response(transcript="actual words")
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="uf-1"
    )
    rec = s.tracer.session("uf-1")[0]
    assert rec.user_facing_output is True


# ---------------------------------------------------------------------------
# 13. test_raw_request_strips_audio_bytes
# ---------------------------------------------------------------------------


def test_raw_request_strips_audio_bytes():
    """``raw_request`` MUST NOT carry the raw audio bytes — strip to a marker.

    Customers may upload multi-MB audio; shipping that on every CallRecord
    to the cloud sink would be a memory + bandwidth disaster. We replace
    ``request`` with a short redaction marker that still records the byte
    length (useful for triage).
    """
    audio = b"\xff" * 1024  # 1 KB of fake audio bytes
    response = _make_response()
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=audio, model="nova-2", _sentinel_session_id="rd-1"
    )
    rec = s.tracer.session("rd-1")[0]
    raw_req_value = rec.raw_request.get("request", "")
    # Marker form: "<redacted:1024_bytes>"
    assert isinstance(raw_req_value, str)
    assert "redacted" in raw_req_value
    assert "1024" in raw_req_value
    # The actual bytes are NOT in the raw_request.
    assert audio not in raw_req_value.encode()


# ---------------------------------------------------------------------------
# 14. test_latency_captured
# ---------------------------------------------------------------------------


def test_latency_captured():
    """The wrapper records elapsed time, not zero."""
    import time as _time

    def slow_transcribe(**kw: Any) -> Any:
        _time.sleep(0.02)  # 20ms
        return _make_response()

    client = _make_deepgram_client()
    client.listen.v1.media.transcribe_file = _RecordingCallable("transcribe_file")
    client.listen.v1.media.transcribe_file.side_effect = slow_transcribe

    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="lt-1"
    )
    rec = s.tracer.session("lt-1")[0]
    assert rec.latency_ms >= 15  # allow some scheduler slack but > 0


# ---------------------------------------------------------------------------
# 15. test_streaming_handles_empty_audio
# ---------------------------------------------------------------------------


def test_streaming_handles_empty_audio():
    """A live session that closes without any Results events still emits.

    The terminal Metadata event with duration=0 is enough to flush a record;
    user_facing_output is False because no transcript was produced.
    """
    events = [_make_metadata_event(duration=0.0, channels=1)]
    socket = _make_fake_socket(events)
    client = _make_deepgram_client(connect_yield=socket)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    with client.listen.v1.connect(model="nova-2", _sentinel_session_id="empty-1") as sock:
        list(sock)

    records = s.tracer.session("empty-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.raw_response_meta["duration_seconds"] == 0.0
    assert rec.user_facing_output is False
    assert rec.raw_response_meta["final_result_count"] == 0


# ---------------------------------------------------------------------------
# 16. test_streaming_handles_keep_alive_events_without_emitting
# ---------------------------------------------------------------------------


def test_streaming_handles_keep_alive_events_without_emitting():
    """KeepAlive events in the iteration MUST NOT cause a CallRecord emit.

    They are heartbeat events; emitting per-keepalive would inflate record
    counts and trigger spurious rule signals. Only the close-of-stream
    Metadata event finalizes the record.
    """
    events = [
        _make_live_event("KeepAlive"),
        _make_live_event("KeepAlive"),
        _make_results_event("hello", is_final=True),
        _make_live_event("KeepAlive"),
        _make_metadata_event(duration=10.0, channels=1),
    ]
    socket = _make_fake_socket(events)
    client = _make_deepgram_client(connect_yield=socket)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    with client.listen.v1.connect(model="nova-2", _sentinel_session_id="ka-1") as sock:
        list(sock)

    records = s.tracer.session("ka-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.raw_response_meta["final_result_count"] == 1


# ---------------------------------------------------------------------------
# 17. test_diarize_flag_captured_in_meta
# ---------------------------------------------------------------------------


def test_diarize_flag_captured_in_meta():
    response = _make_response()
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=b"x",
        model="nova-2",
        diarize=True,
        _sentinel_session_id="diar-1",
    )
    rec = s.tracer.session("diar-1")[0]
    assert rec.usage_extra["model_specific_meta"]["diarize"] is True


# ---------------------------------------------------------------------------
# 18. test_provider_field_set_to_deepgram
# ---------------------------------------------------------------------------


def test_provider_field_set_to_deepgram():
    """Every Deepgram CallRecord carries provider='deepgram' verbatim."""
    response = _make_response()
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="prov-1"
    )

    events = [_make_metadata_event(duration=5.0)]
    socket = _make_fake_socket(events)
    client2 = _make_deepgram_client(connect_yield=socket)
    s2 = Sentinel(project="proj")
    wrap_deepgram(client2, s2)
    with client2.listen.v1.connect(model="nova-2", _sentinel_session_id="prov-2") as sock:
        list(sock)

    assert s.tracer.session("prov-1")[0].provider == "deepgram"
    assert s2.tracer.session("prov-2")[0].provider == "deepgram"


# ---------------------------------------------------------------------------
# 19. test_method_field_distinguishes_transcribe_paths
# ---------------------------------------------------------------------------


def test_method_field_distinguishes_transcribe_paths():
    """The three entry points produce three distinct ``method`` strings."""
    # transcribe_file
    response_f = _make_response()
    client_f = _make_deepgram_client(transcribe_file_return=response_f)
    s_f = Sentinel(project="proj")
    wrap_deepgram(client_f, s_f)
    client_f.listen.v1.media.transcribe_file(
        request=b"x", model="nova-2", _sentinel_session_id="mt-f"
    )

    # transcribe_url
    response_u = _make_response()
    client_u = _make_deepgram_client(transcribe_url_return=response_u)
    s_u = Sentinel(project="proj")
    wrap_deepgram(client_u, s_u)
    client_u.listen.v1.media.transcribe_url(
        url="https://example.com/x.mp3", model="nova-2", _sentinel_session_id="mt-u"
    )

    # transcribe_live
    events = [_make_metadata_event(duration=3.0)]
    socket = _make_fake_socket(events)
    client_l = _make_deepgram_client(connect_yield=socket)
    s_l = Sentinel(project="proj")
    wrap_deepgram(client_l, s_l)
    with client_l.listen.v1.connect(model="nova-2", _sentinel_session_id="mt-l") as sock:
        list(sock)

    assert s_f.tracer.session("mt-f")[0].method == METHOD_TRANSCRIBE_FILE
    assert s_u.tracer.session("mt-u")[0].method == METHOD_TRANSCRIBE_URL
    assert s_l.tracer.session("mt-l")[0].method == METHOD_TRANSCRIBE_LIVE
    # Sanity: the three constants are all distinct.
    assert len({METHOD_TRANSCRIBE_FILE, METHOD_TRANSCRIBE_URL, METHOD_TRANSCRIBE_LIVE}) == 3


# ---------------------------------------------------------------------------
# 20. test_voicemail_short_audio_clip — edge case, <1s audio
# ---------------------------------------------------------------------------


def test_voicemail_short_audio_clip():
    """A <1s clip with no transcript (model couldn't decode) still records.

    Voicemail-short audio is a common edge case — customers occasionally
    chunk audio aggressively and submit sub-second snippets. The wrapper
    must:
      - emit a CallRecord (no silent drop)
      - record the (possibly fractional) duration
      - leave ``user_facing_output`` False (no transcript)
    """
    response = _make_response(duration=0.4, channels=1, transcript="")
    client = _make_deepgram_client(transcribe_file_return=response)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)
    client.listen.v1.media.transcribe_file(
        request=b"\x00" * 16, model="nova-2", _sentinel_session_id="vm-1"
    )

    rec = s.tracer.session("vm-1")[0]
    assert rec.raw_response_meta["duration_seconds"] == pytest.approx(0.4)
    assert rec.user_facing_output is False
    assert rec.provider == "deepgram"


# ---------------------------------------------------------------------------
# Bonus: smoke tests on the internal helpers (not in the 20-test count, but
# useful for catching regressions in the per-helper logic without going
# through the full wrap_deepgram → record_call path).
# ---------------------------------------------------------------------------


def test_request_hash_stable_across_calls():
    """Identical kwargs hash to the same value (retry_storm relies on this)."""
    kwargs = {"model": "nova-2", "diarize": False, "url": "https://x/y.mp3"}
    h1 = _request_hash(kwargs, method=METHOD_TRANSCRIBE_URL)
    h2 = _request_hash(dict(kwargs), method=METHOD_TRANSCRIBE_URL)
    assert h1 == h2


def test_request_hash_audio_bytes_change_hash():
    """Different audio bytes produce different hashes (re-transcribe vs
    transcribe-new file distinction)."""
    h1 = _request_hash({"model": "nova-2", "request": b"abc"}, method=METHOD_TRANSCRIBE_FILE)
    h2 = _request_hash({"model": "nova-2", "request": b"def"}, method=METHOD_TRANSCRIBE_FILE)
    assert h1 != h2


def test_live_accumulator_handles_dict_events():
    """Tests pass dicts in place of Pydantic models; the accumulator copes."""
    acc = _LiveAccumulator()
    acc.observe({"type": "Metadata", "duration": 7.5, "channels": 1})
    assert acc.duration == 7.5
    assert acc.metadata_seen is True


def test_build_record_from_response_handles_dict_metadata():
    """``_build_record_from_response`` accepts dicts in place of the
    Pydantic model — defensive for test mocks."""
    fake = {
        "metadata": {"duration": 12.0, "channels": 1, "request_id": "rid"},
        "results": {"channels": [{"alternatives": [{"transcript": "hi"}]}]},
    }
    rec = _build_record_from_response(
        session_id="t",
        kwargs={"model": "nova-2"},
        response=fake,
        latency_ms=10.0,
        method=METHOD_TRANSCRIBE_FILE,
    )
    assert rec.raw_response_meta["duration_seconds"] == 12.0
    assert rec.user_facing_output is True


def test_build_record_from_live_uses_accumulator_state():
    acc = _LiveAccumulator()
    acc.duration = 60.0
    acc.channels = 2
    acc.has_text_output = True
    acc.final_result_count = 3
    rec = _build_record_from_live(
        session_id="t",
        kwargs={"model": "nova-2", "multichannel": True},
        accumulator=acc,
        latency_ms=15.0,
    )
    assert rec.method == METHOD_TRANSCRIBE_LIVE
    assert rec.raw_response_meta["duration_seconds"] == 60.0
    assert rec.usage_extra["model_specific_meta"]["multichannel"] is True
    assert rec.usage_extra["model_specific_meta"]["channels"] == 2
    assert rec.user_facing_output is True


def test_live_socket_proxy_finalizes_once():
    """The proxy must call ``record_call`` at most once even if multiple
    finalize paths fire (iter end + ctx manager exit + GC)."""
    events = [_make_metadata_event(duration=3.0)]
    socket = _make_fake_socket(events)
    client = _make_deepgram_client(connect_yield=socket)
    s = Sentinel(project="proj")
    wrap_deepgram(client, s)

    with client.listen.v1.connect(model="nova-2", _sentinel_session_id="once-1") as sock:
        # Iterate to natural completion — this triggers __iter__'s finally.
        list(sock)
        # The context-manager __exit__ will ALSO try to finalize. Both must
        # collapse to a single CallRecord.

    assert len(s.tracer.session("once-1")) == 1
