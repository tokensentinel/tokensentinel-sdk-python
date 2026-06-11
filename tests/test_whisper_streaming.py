"""Whisper streaming-duration tests.

 added ``_probe_audio_duration(file_arg)`` that uses mutagen to read
audio metadata for the ``verbose_json`` / ``text`` / ``srt`` / ``vtt``
/ ``json`` response formats. Streaming (``stream=True``) was deferred
because:

  - The stream returns audio segments incrementally
  - No single response object has a duration
  - The mutagen probe needs raw bytes, but the audio file is consumed
    by the time the response stream completes

 ships the buffer-and-probe approach via
:func:`_probe_streaming_audio_duration`: when the SDK's ``file`` kwarg
is bytes / bytes-in-a-tuple / a seekable file-like object, the wrapper
extracts the captured bytes and probes them through mutagen. The
real-time-microphone case (a non-seekable / synthesised stream) is
flagged via ``streaming_realtime=True`` so dashboards distinguish
"we tried and failed" from "this was real-time and probing isn't
possible".

The mutagen import is monkey-patched in each test to avoid forcing the
optional ``token-sentinel[audio-metadata]`` extra into the test matrix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.wrappers.openai import (
    _WHISPER_MAX_FILE_SIZE_BYTES,
    _probe_streaming_audio_duration,
    wrap_openai,
)

# ---------------------------------------------------------------------------
# Mock-mutagen plumbing (mirrors test_openai_wrapper.py's  fixtures)
# ---------------------------------------------------------------------------


def _fake_mutagen_module(length: float | None) -> Any:
    """A fake ``mutagen`` module whose ``File`` factory returns an object
    with ``.info.length == length`` (or ``None`` when ``length is None``).
    """
    module = SimpleNamespace()

    def _file(_target: Any) -> Any:
        if length is None:
            return None
        return SimpleNamespace(info=SimpleNamespace(length=length))

    module.File = _file
    return module


def _install_fake_mutagen(monkeypatch: Any, fake: Any) -> None:
    """Install ``fake`` as the ``mutagen`` module in ``sys.modules``."""
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "mutagen", fake)


def _uninstall_mutagen(monkeypatch: Any) -> None:
    """Make ``import mutagen`` fail with ImportError inside the wrapper.

    Drops any existing module entry and installs a meta_path finder that
    blocks the import. Mirrors the helper in ``test_openai_wrapper.py``.
    """
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "mutagen", raising=False)

    class _BlockingFinder:
        @staticmethod
        def find_spec(name: str, _path: Any = None, _target: Any = None) -> Any:
            if name == "mutagen":
                raise ImportError("mutagen not installed")
            return None

    monkeypatch.setattr(_sys, "meta_path", [_BlockingFinder()] + list(_sys.meta_path))


# ---------------------------------------------------------------------------
# Whisper streaming-response factory
# ---------------------------------------------------------------------------


def _streamed_response_iter(text: str = "streamed text") -> Any:
    """Build a "streamed" response: an iterable of segment-shaped objects.

    The wrapper doesn't iterate the stream itself for  — it just
    needs the response object to NOT carry a top-level ``.duration``
    attribute. We return a list with the final accumulated text on the
    first object (so ``_extract_audio_text`` can return something) and
    no duration field anywhere.
    """
    return [SimpleNamespace(text=text)]


def _build_openai_with_streaming_audio(*, response_factory: Any = None) -> Any:
    """Build a fake OpenAI client whose ``audio.transcriptions.create``
    returns a streaming-shape response (iterable of segments, no
    ``.duration``).
    """
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kwargs: Any) -> Any:
        if response_factory is not None:
            return response_factory()
        return _streamed_response_iter()

    client.audio = SimpleNamespace(
        transcriptions=SimpleNamespace(create=transcribe),
    )
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **_kw: SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
            )
        )
    )
    client.embeddings = SimpleNamespace(
        create=lambda **_kw: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=0),
            data=[SimpleNamespace(embedding=[])],
        )
    )
    return client


# ===========================================================================
# 1. Streaming with captured bytes → mutagen probes duration
# ===========================================================================


def test_streaming_with_captured_bytes_probes_duration(monkeypatch: Any) -> None:
    """When ``stream=True`` and the customer passes raw bytes for ``file``,
    the wrapper captures the bytes BEFORE the streamed iterator returns
    and probes duration via mutagen. The probed value lands in
    ``usage_extra.dimension_value`` and ``dimension_unavailable`` is
    cleared.
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=27.5))

    client = _build_openai_with_streaming_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 4096,  # 4 KB of fake audio bytes
        model="whisper-1",
        response_format="text",
        stream=True,
        _sentinel_session_id="stream-ok",
    )

    rec = s.tracer.session("stream-ok")[0]
    assert isinstance(rec, CallRecord)
    assert rec.method == "audio.transcriptions.create"
    # Duration came from the streaming-bytes mutagen probe.
    assert rec.usage_extra["dimension_value"] == 27.5
    # Streaming flag set, but dimension_unavailable cleared because
    # the probe succeeded.
    assert rec.raw_response_meta.get("streamed") is True
    assert "dimension_unavailable" not in rec.raw_response_meta
    # Real-time flag NOT set — we DID have bytes to probe.
    assert "streaming_realtime" not in rec.raw_response_meta


# ===========================================================================
# 2. Real-time streaming (no captured bytes) → dimension_unavailable +
#    streaming_realtime flag
# ===========================================================================


def test_streaming_realtime_microphone_sets_flag(monkeypatch: Any) -> None:
    """When ``file`` is a non-seekable / non-readable source (real-time
    microphone case), bytes don't exist before the call returns. The
    wrapper sets ``streaming_realtime=True`` and ``dimension_unavailable=True``
    so dashboards distinguish "we tried" from "this is real-time".

    Even if mutagen IS available, the probe never runs on a non-readable
    source — the helper short-circuits to ``(None, None)`` from
    ``_captured_audio_bytes``.
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=42.0))

    # An object that has neither read() nor seek() — the real-time
    # microphone source shape. We use a bare SimpleNamespace so the
    # wrapper's defensive helper falls into the ``(None, None)`` branch.
    realtime_source = SimpleNamespace()

    client = _build_openai_with_streaming_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=realtime_source,
        model="whisper-1",
        response_format="text",
        stream=True,
        _sentinel_session_id="stream-realtime",
    )

    rec = s.tracer.session("stream-realtime")[0]
    assert rec.raw_response_meta.get("streamed") is True
    assert rec.raw_response_meta.get("streaming_realtime") is True
    assert rec.raw_response_meta.get("dimension_unavailable") is True
    assert rec.usage_extra["dimension_value"] == 0.0


# ===========================================================================
# 3. Mutagen failure on streaming bytes → graceful fallback
# ===========================================================================


def test_streaming_mutagen_parse_failure_graceful_fallback(monkeypatch: Any) -> None:
    """If mutagen IS installed but ``mutagen.File`` returns None
    (corrupt / unsupported format), the wrapper degrades to
    ``dimension_unavailable=True`` and records the failure evidence.
    The user's call still returns successfully.
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=None))

    client = _build_openai_with_streaming_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\x00\x01\x02\x03" * 64,  # garbage bytes
        model="whisper-1",
        response_format="text",
        stream=True,
        _sentinel_session_id="stream-corrupt",
    )

    rec = s.tracer.session("stream-corrupt")[0]
    assert rec.usage_extra["dimension_value"] == 0.0
    assert rec.raw_response_meta.get("dimension_unavailable") is True
    assert rec.raw_response_meta.get("streamed") is True
    # Failure evidence captured so dashboards can spot patterns.
    assert rec.raw_response_meta.get("streaming_probe_failed") is True
    assert rec.raw_response_meta.get("streaming_byte_count") == 256
    # streaming_realtime NOT set — we DID have bytes, the probe just
    # couldn't parse them.
    assert "streaming_realtime" not in rec.raw_response_meta


# ===========================================================================
# 4. Bytes exceeding the 25 MB Whisper file-size cap → no probe
# ===========================================================================


def test_streaming_oversize_bytes_respects_25mb_cap(monkeypatch: Any) -> None:
    """Whisper's documented file-size cap is 25 MB. The streaming probe
    skips any byte buffer larger than that — the API would have rejected
    the call upstream so probing locally wastes memory.

    We assert via ``_probe_streaming_audio_duration`` directly (no need
    to spin up a fake client; the probe is exposed as a module helper).
    """
    # Even if mutagen is "installed", oversize bytes short-circuit
    # before mutagen.File is called.
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=999.0))

    too_big = b"\xff" * (_WHISPER_MAX_FILE_SIZE_BYTES + 1)
    result = _probe_streaming_audio_duration(too_big)
    assert result is None

    # Confirm the boundary: exactly at the cap is still allowed (mutagen
    # would parse; we return its length).
    at_cap = b"\xff" * _WHISPER_MAX_FILE_SIZE_BYTES
    result_at_cap = _probe_streaming_audio_duration(at_cap)
    assert result_at_cap == 999.0


# ===========================================================================
# 5. Non-streaming path unchanged (regression check)
# ===========================================================================


def test_nonstreaming_path_unchanged(monkeypatch: Any) -> None:
    """The  non-streaming behaviour is preserved verbatim: when
    ``response.duration`` is present (response_format=verbose_json), the
    wrapper uses it; ``stream`` is not in kwargs and the streaming
    branches are skipped entirely. No ``streamed=True`` evidence appears.
    """
    # Install a fake mutagen that would return 99.0 — if we wrongly fall
    # into the streaming branch this would overwrite duration. The
    # regression check is that we DO NOT.
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=99.0))

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def transcribe(**_kw: Any) -> Any:
        return SimpleNamespace(text="non-streamed", duration=14.0)

    client.audio = SimpleNamespace(
        transcriptions=SimpleNamespace(create=transcribe),
    )
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **_kw: SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
            )
        )
    )
    client.embeddings = SimpleNamespace(
        create=lambda **_kw: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=0),
            data=[SimpleNamespace(embedding=[])],
        )
    )

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 128,
        model="whisper-1",
        response_format="verbose_json",
        _sentinel_session_id="nonstream-reg",
    )

    rec = s.tracer.session("nonstream-reg")[0]
    # Duration from response.duration — NOT from the fake mutagen.
    assert rec.usage_extra["dimension_value"] == 14.0
    # No streaming evidence keys.
    assert "streamed" not in rec.raw_response_meta
    assert "streaming_realtime" not in rec.raw_response_meta
    assert "streaming_probe_failed" not in rec.raw_response_meta
    assert "dimension_unavailable" not in rec.raw_response_meta


# ===========================================================================
# 6. mutagen not installed on streaming path → graceful
# ===========================================================================


def test_streaming_mutagen_import_failure_graceful(monkeypatch: Any) -> None:
    """If mutagen is unavailable, the streaming probe returns None and
    the wrapper records ``dimension_unavailable=True`` — same
    behaviour as the non-streaming path. No exception leaks to the
    customer.
    """
    _uninstall_mutagen(monkeypatch)

    client = _build_openai_with_streaming_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=b"\xff" * 256,
        model="whisper-1",
        response_format="text",
        stream=True,
        _sentinel_session_id="stream-no-mutagen",
    )

    rec = s.tracer.session("stream-no-mutagen")[0]
    assert rec.usage_extra["dimension_value"] == 0.0
    assert rec.raw_response_meta.get("dimension_unavailable") is True
    assert rec.raw_response_meta.get("streamed") is True
    # The probe failure path records evidence regardless of whether
    # the failure was "mutagen absent" vs "mutagen couldn't parse".
    assert rec.raw_response_meta.get("streaming_probe_failed") is True


# ===========================================================================
# 7. Bonus: streaming with tuple (filename, bytes, mime) form → probes
# ===========================================================================


def test_streaming_tuple_form_probes_duration(monkeypatch: Any) -> None:
    """The OpenAI SDK tuple shape ``(filename, content, mime)`` is the
    most common form for HTTP uploads. With ``stream=True`` the wrapper
    extracts the ``content`` bytes from the tuple and probes via
    mutagen, surfacing the mime hint to the probe (advisory only).
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=33.3))

    client = _build_openai_with_streaming_audio()
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    client.audio.transcriptions.create(
        file=("speech.mp3", b"\xff" * 2048, "audio/mpeg"),
        model="whisper-1",
        response_format="text",
        stream=True,
        _sentinel_session_id="stream-tuple",
    )

    rec = s.tracer.session("stream-tuple")[0]
    assert rec.usage_extra["dimension_value"] == 33.3
    assert "dimension_unavailable" not in rec.raw_response_meta
    assert rec.raw_response_meta.get("streamed") is True


# ===========================================================================
# 8. _probe_streaming_audio_duration: defensive arg shapes
# ===========================================================================


def test_probe_streaming_audio_duration_defensive_args(monkeypatch: Any) -> None:
    """The probe accepts ``bytes`` and ``bytearray`` but None / non-bytes
    / empty bytes collapse to ``None``. Defensive contract — the caller
    shouldn't have to type-check the input before the call.
    """
    _install_fake_mutagen(monkeypatch, _fake_mutagen_module(length=10.0))

    assert _probe_streaming_audio_duration(None) is None  # type: ignore[arg-type]
    assert _probe_streaming_audio_duration("not-bytes") is None  # type: ignore[arg-type]
    assert _probe_streaming_audio_duration(b"") is None
    # bytearray works (same shape as bytes for our purposes).
    assert _probe_streaming_audio_duration(bytearray(b"\xff" * 64)) == 10.0
