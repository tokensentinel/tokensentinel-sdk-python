"""Wrap an ElevenLabs ``elevenlabs.client.ElevenLabs`` client to capture call
records for text-to-speech requests.

Pattern: in-place mutation of ``client.text_to_speech.convert``,
``client.text_to_speech.convert_as_stream``, and ``client.text_to_speech.stream``
(when present). The original is captured in a closure so the instrumented
version can delegate. This preserves all type hints in IDEs because we mutate
the live instance, not a subclass.

Reference pattern: see ``wrappers/deepgram.py`` for the per-second-audio
billing pattern. ElevenLabs is the symmetric counterpart — Deepgram is STT
(audio in, text out), ElevenLabs is TTS (text in, audio out). Both bill on
a non-token dimension and both have a streaming iterator on the audio side.
The Deepgram wrapper's ``_LiveSocketProxy`` (observe-then-yield iterator
pattern) is the template here.

SDK shape across versions
-------------------------
The ElevenLabs SDK is Fern-generated and its method surface has shifted across
v1.x / v2.x / v3.x:

  - v1.0+: ``client.text_to_speech.convert(voice_id, text=..., model_id=...,
    output_format=...)`` returning an ``Iterator[bytes]``.
  - v1.0+: ``client.text_to_speech.convert_as_stream(...)`` — same signature,
    same iterator-of-bytes return.
  - v2.x+ added: ``client.text_to_speech.stream(...)`` as the canonical
    streaming entrypoint (newer Fern templates renamed the method).
  - ``AsyncElevenLabs`` client: in 1.0.x the methods are NOT coroutines (the
    Fern template hadn't grown async support yet); in 2.x+ they are. We
    detect via ``inspect.iscoroutinefunction`` and only patch the async
    branch when the method actually IS a coroutine.

The wrapper attribute-checks each method (``getattr(..., None)``) and
silently skips ones that don't exist, so it works across the v1/v2/v3
matrix without a version sniff. The customer's ``pip install`` resolves
whichever methods exist; our patcher mutates what's reachable.

Token accounting — char-count proxy
-----------------------------------
ElevenLabs bills per-character of input text. The CallRecord:

  - ``prompt_tokens = len(text)`` — char count as a proxy. Same convention
    as ``wrappers/voyage.py`` (which uses char count for embedding inputs).
  - ``completion_tokens = 0`` — the output is audio bytes, not tokens. The
    audio output is the BILLABLE unit on the customer's side, but we
    surface it on ``usage_extra.dimension_value`` (per-character of
    INPUT, which is the actual billing unit), not on ``completion_tokens``.
  - ``user_facing_output = True`` — TTS output IS the agent's user-facing
    delivery. Unlike embedding/rerank, this isn't an intermediate
    retrieval step; the bytes the iterator yields are what the customer
    plays back to their end-user.

The ``embedding_waste`` rule and the future ``voice_switching_loop`` rule
(deferred — see "Novel leak pattern" below) both rely on a stable per-call
char count, so we use the char count even though we have no per-call token
figure to compare against.

Novel leak pattern: voice-switching loops
--------------------------------------------------------------
When an agent re-synthesizes the SAME text against DIFFERENT ``voice_id``
values in quick succession (within a single session, say 5–10 seconds), it
is typically a UI experimentation pattern: a developer comparing voices,
or an agent re-rolling voices to find a "better" one. This is virtually
always dev-loop behavior that should not reach production — and at a
per-character billing rate, even short experimentation loops can run up
real dollar amounts.

The  ``voice_switching_loop`` rule would fire when:

  1. ``CallRecord.usage_extra.model_specific_meta.text_hash`` matches across
     N >= 3 consecutive calls in the same session, AND
  2. ``model_specific_meta.voice_id`` differs across those calls, AND
  3. The wall-clock gap between calls is < 10 seconds.

This wrapper  captures the underlying telemetry on
``CallRecord.usage_extra.model_specific_meta`` — specifically ``voice_id``
and ``text_hash`` — so the  rule has zero wrapper-side work. The
``text_hash`` is a SHA-256 of the input text, NOT the text itself: shipping
the raw text on every CallRecord would bloat the cloud sink and increase
the data-residency surface area for customers in regulated industries.

Failure isolation
-----------------
Standard two-level safety boundary mirrors ``wrappers/voyage.py`` and
``wrappers/openai.py``:

  - Record-building errors are swallowed (instrumentation must never break
    the user's call).
  - ``record_call`` exceptions are caught EXCEPT ``LeakDetected``, which
    must propagate so ``mode='block'`` works.

For iterator returns, the finalize call runs once at iterator exhaustion or
generator close. A user who breaks out of iteration early still gets their
record at GC time (with ``LeakDetected`` suppression that mirrors the
OpenAI streaming proxy's ``__del__`` discipline).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from collections.abc import Iterator

    from token_sentinel.sentinel import Sentinel


# Method label written to ``CallRecord.method`` per ElevenLabs entry point.
# Single source of truth so tests can import these constants instead of
# string-literal-ing them; future rule code does the same.
METHOD_TTS_CONVERT = "text_to_speech.convert"
METHOD_TTS_CONVERT_AS_STREAM = "text_to_speech.convert_as_stream"
METHOD_TTS_STREAM = "text_to_speech.stream"


# The set of method names we'll attempt to patch on ``client.text_to_speech``.
# Each entry: (attribute_name, method_label_constant). The wrapper iterates
# this list defensively: missing attributes are skipped silently, so the
# same wrapper code works for v1.x (which has only ``convert`` /
# ``convert_as_stream``) and v2.x+ (which adds ``stream``).
_TTS_METHODS: tuple[tuple[str, str], ...] = (
    ("convert", METHOD_TTS_CONVERT),
    ("convert_as_stream", METHOD_TTS_CONVERT_AS_STREAM),
    ("stream", METHOD_TTS_STREAM),
)


def wrap_elevenlabs(client: Any, sentinel: Sentinel) -> Any:
    """Wrap an ElevenLabs client. Mutates ``text_to_speech.{convert,
    convert_as_stream, stream}`` in place.

    Supports both ``elevenlabs.client.ElevenLabs`` (sync) and
    ``elevenlabs.client.AsyncElevenLabs`` (async). Detects async per-method
    via ``inspect.iscoroutinefunction`` — ElevenLabs' Fern-generated SDK
    has had inconsistent async support across versions, so we cannot
    dispatch on class name alone.

    Returns the same client object with instrumented methods. Missing
    methods (e.g. an older SDK without ``text_to_speech.stream``) are
    tolerated silently — we patch what's reachable.
    """
    tts = getattr(client, "text_to_speech", None)
    if tts is None:
        # Some trimmed mocks / fork SDKs may not expose ``text_to_speech``;
        # silently return rather than crash the customer's setup.
        return client

    for attr_name, method_label in _TTS_METHODS:
        _patch_tts_method(tts, sentinel, attr_name, method_label)

    return client


def _patch_tts_method(
    tts: Any,
    sentinel: Sentinel,
    attr_name: str,
    method_label: str,
) -> None:
    """Replace ``tts.<attr_name>`` with an instrumented variant."""
    original = getattr(tts, attr_name, None)
    if original is None or not callable(original):
        return

    is_async = inspect.iscoroutinefunction(original)

    if is_async:
        instrumented: Any = _make_async_tts(original, sentinel, method_label)
    else:
        instrumented = _make_sync_tts(original, sentinel, method_label)

    try:
        setattr(tts, attr_name, instrumented)
    except Exception:
        # If the SDK forbids attribute assignment on the TTS subclient
        # (frozen / slotted), silently skip. The customer's call still
        # works through the original method; only our instrumentation
        # is missing.
        return


# ---------------------------------------------------------------------------
# sync + async wrappers
# ---------------------------------------------------------------------------


def _make_sync_tts(original: Any, sentinel: Sentinel, method_label: str) -> Any:
    @functools.wraps(original)
    def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        captured_kwargs = dict(kwargs)
        captured_args = args

        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except Exception:
            # SDK rejected the call; nothing to record. Propagate.
            raise

        # If the return value is an iterator (the common case for ElevenLabs
        # TTS — bytes stream in as audio is synthesized), wrap it so we can
        # finalize a CallRecord on stream exhaustion. Otherwise (e.g. a mock
        # returning bytes directly), finalize immediately.
        if _is_byte_iterator(result):
            return _SyncBytesProxy(
                stream=result,
                sentinel=sentinel,
                session_id=session_id,
                args=captured_args,
                kwargs=captured_kwargs,
                method_label=method_label,
                start=start,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        _finalize(
            sentinel=sentinel,
            session_id=session_id,
            args=captured_args,
            kwargs=captured_kwargs,
            method_label=method_label,
            latency_ms=elapsed_ms,
        )
        return result

    return instrumented


def _make_async_tts(original: Any, sentinel: Sentinel, method_label: str) -> Any:
    @functools.wraps(original)
    async def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        captured_kwargs = dict(kwargs)
        captured_args = args

        start = time.perf_counter()
        result = await original(*args, **kwargs)

        # The async TTS path may return either an async iterator (when the
        # SDK exposes streaming bytes) or a plain awaited value (when the
        # SDK collects bytes server-side first). Handle both.
        if _is_async_byte_iterator(result):
            return _AsyncBytesProxy(
                stream=result,
                sentinel=sentinel,
                session_id=session_id,
                args=captured_args,
                kwargs=captured_kwargs,
                method_label=method_label,
                start=start,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        _finalize(
            sentinel=sentinel,
            session_id=session_id,
            args=captured_args,
            kwargs=captured_kwargs,
            method_label=method_label,
            latency_ms=elapsed_ms,
        )
        return result

    return instrumented


# ---------------------------------------------------------------------------
# Iterator detection
# ---------------------------------------------------------------------------


def _is_byte_iterator(obj: Any) -> bool:
    """Return True if ``obj`` is an iterator (and not bytes / str directly).

    ElevenLabs' ``convert`` returns an ``Iterator[bytes]``; bytes itself
    (which IS iterable) should NOT be treated as a stream — we want to
    finalize immediately for the simple bytes case.
    """
    if obj is None:
        return False
    if isinstance(obj, (bytes, bytearray, str)):
        return False
    # An object with ``__iter__`` and ``__next__`` is a proper iterator.
    # Generators satisfy both; lists satisfy __iter__ only (still
    # acceptable — wrapping them is harmless).
    return hasattr(obj, "__iter__") and not isinstance(obj, dict)


def _is_async_byte_iterator(obj: Any) -> bool:
    """Return True if ``obj`` is an async iterator."""
    if obj is None:
        return False
    if isinstance(obj, (bytes, bytearray, str)):
        return False
    return hasattr(obj, "__aiter__")


# ---------------------------------------------------------------------------
# Sync / async bytes proxies
# ---------------------------------------------------------------------------


class _SyncBytesProxy:
    """Proxy around an ``Iterator[bytes]`` returned by ElevenLabs TTS.

    Forwards attribute access to the wrapped iterator so anything else the
    SDK exposes on the return value (rare, but possible — newer Fern
    templates wrap the iterator in a thin object with metadata) keeps
    working. Re-implements ``__iter__`` so we can observe consumption and
    finalize a CallRecord on iterator exhaustion / close / GC.

    Mirrors ``wrappers.openai._OpenAIStreamProxy`` and
    ``wrappers.deepgram._LiveSocketProxy``: dunders defined on the class
    (Python's special-method lookup bypasses instance attributes), single
    finalize gate guarded by ``_finalized``, GeneratorExit-safe
    ``LeakDetected`` suppression in the GC path.
    """

    __slots__ = (
        "_stream",
        "_sentinel",
        "_session_id",
        "_args",
        "_kwargs",
        "_method_label",
        "_start",
        "_finalized",
        "_bytes_yielded",
    )

    def __init__(
        self,
        *,
        stream: Any,
        sentinel: Sentinel,
        session_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        method_label: str,
        start: float,
    ) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_sentinel", sentinel)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_method_label", method_label)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_finalized", False)
        object.__setattr__(self, "_bytes_yielded", 0)

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._stream:
                if isinstance(chunk, (bytes, bytearray)):
                    object.__setattr__(self, "_bytes_yielded", self._bytes_yielded + len(chunk))
                yield chunk
        finally:
            self._safe_finalize()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ fires only on lookup miss, so this transparently
        # exposes anything the SDK's wrapped iterator offers without us
        # enumerating it.
        return getattr(self._stream, name)

    def __del__(self) -> None:
        # Last-resort flush. ``__del__`` must never raise — Python silently
        # swallows ``__del__`` exceptions anyway, which means LeakDetected
        # from block mode would vanish without the user's app halting.
        # Suppress LeakDetected explicitly and emit a warning so the user
        # knows block mode was best-effort on this path.
        try:
            self._safe_finalize(suppress_block=True)
        except Exception:
            pass

    def _safe_finalize(self, *, suppress_block: bool = False) -> None:
        if object.__getattribute__(self, "_finalized"):
            return
        object.__setattr__(self, "_finalized", True)

        is_close = sys.exc_info()[0] is GeneratorExit
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            _finalize(
                sentinel=self._sentinel,
                session_id=self._session_id,
                args=self._args,
                kwargs=self._kwargs,
                method_label=self._method_label,
                latency_ms=elapsed_ms,
                bytes_yielded=self._bytes_yielded,
            )
        except LeakDetected:
            if is_close or suppress_block:
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in ElevenLabs "
                    "stream close path (block mode is best-effort on "
                    "abandoned audio iterators; iterate to completion for "
                    "block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass


class _AsyncBytesProxy:
    """Async counterpart of ``_SyncBytesProxy``.

    Wraps an async iterator of bytes (newer SDK versions of
    ``AsyncElevenLabs.text_to_speech.stream``). Re-implements ``__aiter__``;
    everything else is forwarded.
    """

    __slots__ = (
        "_stream",
        "_sentinel",
        "_session_id",
        "_args",
        "_kwargs",
        "_method_label",
        "_start",
        "_finalized",
        "_bytes_yielded",
    )

    def __init__(
        self,
        *,
        stream: Any,
        sentinel: Sentinel,
        session_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        method_label: str,
        start: float,
    ) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_sentinel", sentinel)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_method_label", method_label)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_finalized", False)
        object.__setattr__(self, "_bytes_yielded", 0)

    def __aiter__(self) -> Any:
        return self._aiter_impl()

    async def _aiter_impl(self) -> Any:
        try:
            async for chunk in self._stream:
                if isinstance(chunk, (bytes, bytearray)):
                    object.__setattr__(self, "_bytes_yielded", self._bytes_yielded + len(chunk))
                yield chunk
        finally:
            self._safe_finalize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        try:
            self._safe_finalize(suppress_block=True)
        except Exception:
            pass

    def _safe_finalize(self, *, suppress_block: bool = False) -> None:
        if object.__getattribute__(self, "_finalized"):
            return
        object.__setattr__(self, "_finalized", True)

        is_close = sys.exc_info()[0] is GeneratorExit
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            _finalize(
                sentinel=self._sentinel,
                session_id=self._session_id,
                args=self._args,
                kwargs=self._kwargs,
                method_label=self._method_label,
                latency_ms=elapsed_ms,
                bytes_yielded=self._bytes_yielded,
            )
        except LeakDetected:
            if is_close or suppress_block:
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in ElevenLabs "
                    "async stream close path (block mode is best-effort on "
                    "abandoned audio iterators; iterate to completion for "
                    "block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Finalize: build CallRecord + record_call
# ---------------------------------------------------------------------------


def _finalize(
    *,
    sentinel: Sentinel,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_label: str,
    latency_ms: float,
    bytes_yielded: int = 0,
) -> None:
    """Build a CallRecord and ship it through ``sentinel.record_call``.

    Two-level safety boundary:
      - Record-build errors are swallowed (instrumentation must never break
        the user's call).
      - ``record_call`` exceptions are caught EXCEPT ``LeakDetected``, which
        must propagate so ``mode='block'`` works.
    """
    try:
        record = _build_record(
            session_id=session_id,
            args=args,
            kwargs=kwargs,
            method_label=method_label,
            latency_ms=latency_ms,
            bytes_yielded=bytes_yielded,
        )
    except Exception:
        return
    try:
        sentinel.record_call(record)
    except LeakDetected:
        raise
    except Exception:
        pass


def _build_record(
    *,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_label: str,
    latency_ms: float,
    bytes_yielded: int = 0,
) -> CallRecord:
    """Build a CallRecord from an ElevenLabs TTS call.

    Argument extraction is defensive — ElevenLabs' Fern-generated SDK takes
    ``voice_id`` as the first positional argument and the rest as kwargs,
    but the customer may have called positionally or via kwargs only. We
    handle both.

    Char-count proxy for ``prompt_tokens`` (see module docstring). The
    ``raw_request`` strips the literal text content (some customers route
    sensitive copy through TTS — emails, financial summaries — and we
    should not be the path that ships it to the cloud sink) but keeps
    the configuration knobs that the  ``voice_switching_loop`` rule
    will need.
    """
    text = _extract_text(args, kwargs)
    voice_id = _extract_voice_id(args, kwargs)
    model_id = _extract_kwarg_str(kwargs, "model_id")
    output_format = _extract_kwarg_str(kwargs, "output_format")
    optimize_streaming_latency = kwargs.get("optimize_streaming_latency")

    # Char count is the prompt_tokens proxy. ElevenLabs bills per-character
    # of INPUT, so this is also the billing dimension on the customer side
    # — same as Voyage's char-count proxy for embed inputs.
    char_count = len(text) if isinstance(text, str) else 0

    # Hash of the literal text — keyed for the voice_switching_loop
    # rule which compares text across calls without ever needing to read
    # the text. SHA-256 truncated to 16 hex chars (64 bits) matches the
    # Sentinel-side dedup key length and gives collision-free behavior
    # at any realistic session scale.
    text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    # ``request_hash`` is the full-fidelity per-call hash (used for retry
    # storm / duplicate detection). It DOES include the text — different
    # text is a different call. The truncated ``text_hash`` above is what
    # the  rule reads from ``usage_extra``; the longer ``request_hash``
    # is the SDK-wide convention for retry detection.
    request_hash = hashlib.sha256(
        json.dumps(
            {
                "method": method_label,
                "voice_id": voice_id,
                "model_id": model_id,
                "output_format": output_format,
                "text": text,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    raw_request = _redact_text(
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        optimize_streaming_latency=optimize_streaming_latency,
        char_count=char_count,
    )

    usage_extra = {
        "dimension_kind": "per_character",
        "dimension_value": char_count,
        "model_specific_meta": {
            "voice_id": voice_id,
            "model_id": model_id,
            "output_format": output_format,
            # text_hash: feeds the voice_switching_loop rule. Kept on
            # the per_character meta so a future rule can correlate across
            # calls (same text + different voice_id within 10s = loop).
            "text_hash": text_hash,
        },
    }

    raw_response_meta: dict[str, Any] = {
        "bytes_yielded": bytes_yielded,
    }

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="elevenlabs",
        # ``voice_id`` is the model in ElevenLabs (one voice maps to one
        # billing-rate target on the customer side; the ``model_id`` knob
        # selects the synthesis engine, which is a separate axis the
        # rule reads from ``model_specific_meta.model_id``). Surfacing
        # voice_id as the .model field keeps per-model dashboards
        # meaningful — customers ask "which voice am I burning the most
        # characters on?" and that's what they get back.
        model=voice_id or "unknown",
        method=method_label,
        prompt_tokens=char_count,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=True,
        raw_request=raw_request,
        raw_response_meta=raw_response_meta,
        usage_extra=usage_extra,
    )


# ---------------------------------------------------------------------------
# Argument extraction helpers
# ---------------------------------------------------------------------------


def _extract_voice_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Pull ``voice_id`` from the call, kwargs first, then positional.

    ElevenLabs' ``convert(voice_id, *, text=..., ...)`` accepts voice_id as
    the first positional argument; many customers invoke via kwarg too.
    Defensive: coerce to string, default to empty.
    """
    raw = kwargs.get("voice_id")
    if raw is None and args:
        raw = args[0]
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else str(raw)


def _extract_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Pull the ``text`` kwarg. ElevenLabs requires it as keyword-only.

    Defensive: if a customer passes a non-string (bytes, an object with
    __str__, etc.) we coerce. Empty/missing text returns "".
    """
    raw = kwargs.get("text")
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else str(raw)


def _extract_kwarg_str(kwargs: dict[str, Any], key: str) -> str | None:
    """Pull a kwarg and coerce to str (or None).

    ElevenLabs' Fern-generated SDK uses ``Ellipsis`` (``...``) as a sentinel
    for "kwarg not passed", separate from ``None``. We treat both as
    "missing" — the customer didn't make a choice, and we should not
    record a stale default.
    """
    raw = kwargs.get(key)
    if raw is None or raw is Ellipsis:
        return None
    return raw if isinstance(raw, str) else str(raw)


def _redact_text(
    *,
    voice_id: str,
    model_id: str | None,
    output_format: str | None,
    optimize_streaming_latency: Any,
    char_count: int,
) -> dict[str, Any]:
    """Build a ``raw_request`` dict that strips the literal text content.

    The customer's text can be sensitive (emails, financial summaries,
    medical content). Including the literal text on ``CallRecord.raw_request``
    would ship it to the cloud sink and increase the data-residency surface.
    We keep the configuration knobs and replace text with a redaction
    marker that still records the char count (useful for triage).
    """
    return {
        "voice_id": voice_id,
        "model_id": model_id,
        "output_format": output_format,
        "optimize_streaming_latency": optimize_streaming_latency,
        # Literal text is stripped; marker preserves char count for triage.
        "text": f"<redacted:{char_count}_chars>",
    }
