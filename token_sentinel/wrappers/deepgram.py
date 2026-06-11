"""Wrap a Deepgram ``deepgram-sdk`` client to capture call records.

Pattern: in-place mutation of ``client.listen.v1.media.transcribe_file``,
``client.listen.v1.media.transcribe_url``, and instrumentation of the live
streaming context manager returned by ``client.listen.v1.connect``. The
original is captured in a closure so the instrumented version can delegate.
This preserves type hints in IDEs because we mutate the live instance.

Reference pattern: see ``wrappers/bedrock.py`` for the streaming pattern.
Deepgram's live transcription socket is structurally identical to Bedrock's
``converse_stream`` EventStream — we wrap the iterator that yields events
until a terminal event (``ListenV1Metadata`` close-of-stream summary) fires,
emitting one ``CallRecord`` at stream close. The Bedrock wrapper's
``_StreamUsageAccumulator`` / ``_EventStreamProxy`` are the template here.

Four cases handled by ``wrap_deepgram``:
  1. Sync pre-recorded file: ``client.listen.v1.media.transcribe_file(...)``
     -> ``ListenV1Response`` with ``metadata.duration`` (audio seconds, the
     billable unit), ``metadata.channels``, and a ``results`` block.
  2. Sync pre-recorded URL: ``client.listen.v1.media.transcribe_url(...)``
     -> same response shape.
  3. Async pre-recorded (both file + url): ``AsyncDeepgramClient`` exposes
     the same methods as coroutines.
  4. Live streaming: ``client.listen.v1.connect(...)`` returns a context
     manager yielding a ``V1SocketClient``. The socket iterates yielding
     ``ListenV1Results`` / ``ListenV1Metadata`` / ``ListenV1UtteranceEnd`` /
     ``ListenV1SpeechStarted`` events; the closing ``ListenV1Metadata``
     carries the total billable duration. We wrap ``__iter__`` so events
     pass through observable, and emit a single ``CallRecord`` when the
     stream closes.

**Streaming-emit timing.** We emit ONE ``CallRecord`` per live connection,
on close — not per Final Result event. Rationale: a single live connection
is a single billing unit on the Deepgram side (one continuous duration meter
running for the lifetime of the websocket). Emitting per Final Result would
fragment what is logically one call into N records and would multiply rule
load by a factor proportional to the number of utterances in the call,
without any matching billing event. The terminal ``ListenV1Metadata``
(emitted when the server closes the stream) carries the authoritative
``duration`` and ``channels`` — we accumulate transcript text during the
stream for the ``user_facing_output`` flag and finalize on close. Bedrock
``converse_stream`` follows the same emit-on-terminal-event discipline; this
keeps the two streaming providers consistent.

**Novel leak pattern: multi-channel stereo doubling.** When a customer sets
``multichannel=True`` on a stereo audio file (``metadata.channels >= 2``),
Deepgram bills per-second-PER-channel — a 60s stereo file with
``multichannel=True`` is billed as 120 seconds. This is a real
dollar-meaningful leak that today's observability tools do not catch (the
vendor dashboard shows aggregate minutes, not channel-level breakdown).
A  rule named ``audio_multichannel_doubling`` would fire when
``usage_extra.model_specific_meta.channels >= 2`` AND
``usage_extra.model_specific_meta.multichannel is True`` AND the per-channel
transcripts have high similarity (signal: the second channel is redundant
content from a mixer bug, not a separate speaker). this wrapper
captures the underlying telemetry on ``CallRecord.usage_extra``; the rule
itself is  scope.

Failure isolation: standard two-level safety boundary mirrors
``wrappers/bedrock.py`` and ``wrappers/openai.py``. Record-building errors
are swallowed; ``record_call`` exceptions are caught EXCEPT ``LeakDetected``,
which must propagate so block mode works.

SDK shape note: the research described an older v3 SDK API
(``client.listen.prerecorded.v("1").transcribe_file``). The current v7
``deepgram-sdk`` (Fern-generated) reorganizes this as ``client.listen.v1.media.*``.
We wrap the v7 shape since that is what customers ``pip install`` today;
older v3 clients lack the ``listen.v1.media`` accessor and the
attribute-check in ``wrap_deepgram`` simply skips them (silent no-op rather
than crash).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from collections.abc import Iterator

    from token_sentinel.sentinel import Sentinel


# Module-level constant — the "method" string written to CallRecord for each
# Deepgram entry point. Kept here so tests and rule code can import a single
# source of truth instead of string-literal-ing it.
METHOD_TRANSCRIBE_FILE = "transcribe_file"
METHOD_TRANSCRIBE_URL = "transcribe_url"
METHOD_TRANSCRIBE_LIVE = "transcribe_live"


def wrap_deepgram(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a ``deepgram.DeepgramClient`` / ``AsyncDeepgramClient``. Mutates
    ``client.listen.v1.media.transcribe_file``,
    ``client.listen.v1.media.transcribe_url``, and
    ``client.listen.v1.connect`` in place.

    The accessor chain ``client.listen.v1.media`` is exercised lazily; on the
    real SDK ``media`` is a ``@property`` so first-access instantiates the
    ``MediaClient``. We instrument the resulting bound methods so the
    mutation outlives subsequent ``client.listen.v1.media`` accesses (those
    return the same memoised instance).

    Returns the same client object with instrumented methods.
    """
    # Resolve client.listen.v1 — defensively, since some integration tests
    # build a minimal mock that lacks the chain.
    listen = getattr(client, "listen", None)
    if listen is None:
        return client
    v1 = getattr(listen, "v1", None)
    if v1 is None:
        return client

    is_async = _is_async_client(client)

    # --- listen.v1.media.{transcribe_file, transcribe_url} ------------------
    media = getattr(v1, "media", None)
    if media is not None:
        _patch_transcribe(media, sentinel, "transcribe_file", METHOD_TRANSCRIBE_FILE, is_async)
        _patch_transcribe(media, sentinel, "transcribe_url", METHOD_TRANSCRIBE_URL, is_async)

    # --- listen.v1.connect (live streaming) ---------------------------------
    original_connect = getattr(v1, "connect", None)
    if original_connect is not None and callable(original_connect):
        try:
            if is_async:
                v1.connect = _make_async_connect(original_connect, sentinel)
            else:
                v1.connect = _make_sync_connect(original_connect, sentinel)
        except Exception:
            # If the SDK forbids attribute assignment on the v1 client (frozen
            # / slotted), skip rather than crash. Pre-recorded instrumentation
            # still works.
            pass

    return client


def _is_async_client(client: Any) -> bool:
    """Return True if the given Deepgram client is an async client.

    Checks both the class name (cheapest, works for the official SDK) and
    the coroutine-ness of ``client.listen.v1.media.transcribe_file`` for
    forks / subclasses.
    """
    try:
        if type(client).__name__ == "AsyncDeepgramClient":
            return True
    except Exception:
        pass
    try:
        listen = getattr(client, "listen", None)
        v1 = getattr(listen, "v1", None)
        media = getattr(v1, "media", None)
        tf = getattr(media, "transcribe_file", None)
        if tf is not None and inspect.iscoroutinefunction(tf):
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# transcribe_file / transcribe_url — sync + async (pre-recorded)
# ---------------------------------------------------------------------------


def _patch_transcribe(
    media: Any,
    sentinel: Sentinel,
    attr_name: str,
    method_label: str,
    is_async: bool,
) -> None:
    """Replace ``media.<attr_name>`` with an instrumented variant."""
    original = getattr(media, attr_name, None)
    if original is None or not callable(original):
        return

    if is_async:

        @functools.wraps(original)
        async def instrumented_async(*args: Any, **kwargs: Any) -> Any:
            session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
            captured_kwargs = dict(kwargs)
            start = time.perf_counter()
            try:
                response = await original(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

            try:
                record = _build_record_from_response(
                    session_id=session_id,
                    kwargs=captured_kwargs,
                    response=response,
                    latency_ms=elapsed_ms,
                    method=method_label,
                )
            except Exception:
                return response
            try:
                sentinel.record_call(record)
            except LeakDetected:
                raise
            except Exception:
                pass
            return response

        try:
            setattr(media, attr_name, instrumented_async)
        except Exception:
            return
        return

    @functools.wraps(original)
    def instrumented_sync(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        captured_kwargs = dict(kwargs)
        start = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_record_from_response(
                session_id=session_id,
                kwargs=captured_kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method=method_label,
            )
        except Exception:
            return response
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass
        return response

    try:
        setattr(media, attr_name, instrumented_sync)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Live streaming — connect()
# ---------------------------------------------------------------------------
#
# ``client.listen.v1.connect`` is a ``@contextmanager``-decorated generator
# function. Calling it returns a context manager; entering it yields a
# ``V1SocketClient`` whose ``__iter__`` yields the four event types listed at
# the top of this module. We wrap the socket in a ``_LiveSocketProxy`` that
# observes events as they pass through and finalizes a CallRecord on close.
#
# The Bedrock ``converse_stream`` wrapper does the same thing: wrap an
# iterator that yields server-pushed events, accumulate usage as they pass,
# flush exactly once at stream end. We reuse that mental model directly.


class _LiveAccumulator:
    """Aggregate live-streaming events into a billing summary.

    The authoritative duration + channel count arrive in the terminal
    ``ListenV1Metadata`` event when the server closes the stream. Final
    ``ListenV1Results`` events (with ``is_final=True``) carry per-utterance
    transcript text; we observe those to flip ``has_text_output`` so the
    ``user_facing_output`` flag on the CallRecord is accurate.

    ``KeepAlive`` events from the SDK are ignored — they are client→server
    heartbeats that should never make it back to the client, but if a
    mock test (or a buggy server) emits one, we simply skip it without
    triggering an emit.
    """

    def __init__(self) -> None:
        self.duration: float = 0.0
        self.channels: int = 1
        self.has_text_output: bool = False
        self.final_result_count: int = 0
        self.metadata_seen: bool = False
        self.request_id: str | None = None

    def observe(self, event: Any) -> None:
        try:
            event_type = _get_event_type(event)
            if event_type == "Metadata":
                # Terminal close-of-stream event with billing-authoritative
                # duration + channel count.
                duration = _get_attr_or_key(event, "duration")
                if isinstance(duration, (int, float)) and duration >= 0:
                    self.duration = float(duration)
                channels = _get_attr_or_key(event, "channels")
                if isinstance(channels, int) and channels >= 1:
                    self.channels = channels
                req_id = _get_attr_or_key(event, "request_id")
                if isinstance(req_id, str):
                    self.request_id = req_id
                self.metadata_seen = True
                return
            if event_type == "Results":
                # is_final=True utterances carry the user-facing transcript.
                is_final = _get_attr_or_key(event, "is_final")
                if is_final:
                    self.final_result_count += 1
                    transcript = _extract_transcript_text(event)
                    if transcript:
                        self.has_text_output = True
                return
            if event_type == "KeepAlive":
                # Heartbeat — never emit, never accumulate.
                return
            # SpeechStarted / UtteranceEnd / unknown types: ignore. The
            # close-of-stream Metadata event is what we wait for.
        except Exception:
            # Never let event-observation crash the user's iteration.
            pass


def _get_event_type(event: Any) -> str | None:
    """Best-effort extract of the event's ``type`` field.

    The real SDK returns Pydantic models with a ``type`` attribute; tests
    typically pass plain dicts or SimpleNamespace. We handle both.
    """
    try:
        t = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        if isinstance(t, str):
            return t
    except Exception:
        pass
    return None


def _get_attr_or_key(obj: Any, name: str) -> Any:
    """Return ``obj.name`` if it's an attribute, else ``obj[name]`` if dict.

    Defensive helper for tests that pass dicts in place of Pydantic models.
    """
    try:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
    except Exception:
        return None


def _extract_transcript_text(result_event: Any) -> str:
    """Pull the transcript string out of a ``ListenV1Results`` event.

    Real shape:
        event.channel.alternatives[0].transcript

    Tests may pass simpler shapes; we walk defensively.
    """
    try:
        channel = _get_attr_or_key(result_event, "channel")
        if channel is None:
            return ""
        alternatives = _get_attr_or_key(channel, "alternatives")
        if not alternatives:
            return ""
        first = alternatives[0]
        transcript = _get_attr_or_key(first, "transcript")
        if isinstance(transcript, str):
            return transcript
    except Exception:
        pass
    return ""


class _LiveSocketProxy:
    """Proxy around a ``V1SocketClient`` that observes events on iteration.

    Forwards attribute access to the wrapped socket so callers using
    ``send_media``/``send_finalize``/``send_close_stream``/``recv`` keep
    working. Re-implements ``__iter__`` so we can siphon events into the
    accumulator. The dunder is defined on the *class* — Python's
    special-method lookup bypasses instance attributes for dunders, so
    this is the only correct place for it (matches the Bedrock
    ``_EventStreamProxy`` pattern).

    On stream end (iterator exhausted, ``__exit__`` from the outer
    context manager, or ``__del__``), we call ``_finalize`` exactly once,
    which builds and records the CallRecord. The "exactly once" is enforced
    by ``_flushed``.
    """

    __slots__ = (
        "_socket",
        "_accumulator",
        "_sentinel",
        "_kwargs",
        "_session_id",
        "_start",
        "_flushed",
    )

    def __init__(
        self,
        socket: Any,
        accumulator: _LiveAccumulator,
        sentinel: Sentinel,
        kwargs: dict[str, Any],
        session_id: str,
        start: float,
    ) -> None:
        object.__setattr__(self, "_socket", socket)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_sentinel", sentinel)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_flushed", False)

    def __iter__(self) -> Iterator[Any]:
        accumulator = self._accumulator
        try:
            for event in self._socket:
                accumulator.observe(event)
                yield event
        finally:
            self._finalize()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup misses, so this
        # transparently exposes everything the underlying socket offers
        # (``send_media``, ``send_finalize``, ``recv``, internal cursors,
        # etc.) without us enumerating them.
        return getattr(self._socket, name)

    def __del__(self) -> None:
        # Last-resort flush. Users who break out of iteration early without
        # the outer ``with client.listen.v1.connect(...)`` cleanly closing
        # still get their CallRecord at GC time. ``__del__`` must never
        # raise — Python silently swallows ``__del__`` exceptions anyway,
        # which means ``LeakDetected`` from block mode would vanish without
        # the user's app halting. Suppress ``LeakDetected`` explicitly here
        # and emit a warning so the user knows block mode was best-effort
        # on this path.
        try:
            self._finalize(suppress_block=True)
        except Exception:
            pass

    def _finalize(self, *, suppress_block: bool = False) -> None:
        if self._flushed:
            return
        # Mark flushed BEFORE building the record so a record-building
        # exception that triggers ``__del__`` recursion can't double-fire.
        object.__setattr__(self, "_flushed", True)

        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            record = _build_record_from_live(
                session_id=self._session_id,
                kwargs=self._kwargs,
                accumulator=self._accumulator,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            if suppress_block:
                # Called from ``__del__`` where Python would swallow the
                # exception anyway. Warn so the user knows their leak fired
                # but block mode could not halt the (already-abandoned)
                # connection.
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in Deepgram live-stream "
                    "GC path (block mode is best-effort on abandoned sockets; use "
                    "'with client.listen.v1.connect(...) as socket:' and iterate "
                    "to completion for block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass


def _make_sync_connect(original_connect: Any, sentinel: Sentinel) -> Any:
    """Wrap sync ``listen.v1.connect`` — a ``@contextmanager`` function.

    The original is decorated with ``@contextmanager`` so it's a function
    returning a context manager (not a coroutine). We replace it with our
    own context manager that delegates to the original and yields a
    ``_LiveSocketProxy`` instead of the raw socket.
    """

    @functools.wraps(original_connect)
    @contextmanager
    def instrumented(*args: Any, **kwargs: Any) -> Iterator[Any]:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        captured_kwargs = dict(kwargs)
        start = time.perf_counter()
        accumulator = _LiveAccumulator()
        proxy_holder: dict[str, _LiveSocketProxy] = {}
        try:
            with original_connect(*args, **kwargs) as socket:
                try:
                    proxy = _LiveSocketProxy(
                        socket=socket,
                        accumulator=accumulator,
                        sentinel=sentinel,
                        kwargs=captured_kwargs,
                        session_id=session_id,
                        start=start,
                    )
                    proxy_holder["proxy"] = proxy
                except Exception:
                    # If wrapping fails for any reason, hand back the raw
                    # socket so user code never breaks because of us.
                    yield socket
                    return
                yield proxy
        finally:
            # On context-manager exit (normal or exceptional), ensure the
            # proxy finalizes exactly once. ``__iter__``'s finally also
            # triggers _finalize; ``_flushed`` makes the second call a no-op.
            stored_proxy = proxy_holder.get("proxy")
            if stored_proxy is not None:
                stored_proxy._finalize()

    return instrumented


def _make_async_connect(original_connect: Any, sentinel: Sentinel) -> Any:
    """Async counterpart of ``_make_sync_connect``.

    The async SDK exposes ``connect`` as an async context manager. We
    mirror the sync path with an ``@asynccontextmanager``-style
    implementation.
    """
    from contextlib import asynccontextmanager

    @functools.wraps(original_connect)
    @asynccontextmanager
    async def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        captured_kwargs = dict(kwargs)
        start = time.perf_counter()
        accumulator = _LiveAccumulator()
        proxy_holder: dict[str, _LiveSocketProxy] = {}
        try:
            async with original_connect(*args, **kwargs) as socket:
                try:
                    proxy = _LiveSocketProxy(
                        socket=socket,
                        accumulator=accumulator,
                        sentinel=sentinel,
                        kwargs=captured_kwargs,
                        session_id=session_id,
                        start=start,
                    )
                    proxy_holder["proxy"] = proxy
                except Exception:
                    yield socket
                    return
                yield proxy
        finally:
            stored_proxy = proxy_holder.get("proxy")
            if stored_proxy is not None:
                stored_proxy._finalize()

    return instrumented


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _request_hash(kwargs: dict[str, Any], *, method: str) -> str:
    """SHA-256 of the request shape for retry-storm / re-transcribe detection.

    For pre-recorded paths we include the ``url`` (transcribe_url) or a
    digest of the audio bytes (transcribe_file) plus model/diarize/channels/
    multichannel flags. Audio bytes themselves are NOT included in the hash
    input verbatim — we hash them separately and include the digest — so a
    100MB audio file doesn't bloat the hash input.

    For live streaming we include the model + the streaming feature flags
    (channels, diarize, multichannel) since the audio payload streams in
    after connect-time and isn't available here.
    """
    model = kwargs.get("model", "unknown")
    diarize = bool(kwargs.get("diarize", False))
    channels = kwargs.get("channels")
    multichannel = bool(kwargs.get("multichannel", False))
    language = kwargs.get("language")

    audio_digest: str = ""
    url = kwargs.get("url")
    if isinstance(url, str):
        # transcribe_url path
        payload_key = {"url": url}
    elif "request" in kwargs:
        # transcribe_file path — hash the bytes/iterator-marker
        request = kwargs.get("request")
        if isinstance(request, (bytes, bytearray)):
            audio_digest = hashlib.sha256(bytes(request)).hexdigest()
        elif request is not None:
            # Iterator / async iterator — we can't peek without consuming.
            # Hash its id() as a per-call distinguisher; re-transcribe
            # detection in the embedding_waste rule will rely on file bytes
            # if the customer hands us bytes, and ignore iterator-based
            # uploads (they are inherently streaming).
            audio_digest = f"iter:{id(request):x}"
        payload_key = {"audio_digest": audio_digest}
    else:
        # Live streaming — no audio payload at connect time.
        payload_key = {}

    return hashlib.sha256(
        json.dumps(
            {
                "method": method,
                "model": model,
                "diarize": diarize,
                "channels": channels,
                "multichannel": multichannel,
                "language": language,
                **payload_key,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _redact_audio_bytes(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a ``raw_request`` dict that strips audio bytes / iterators.

    The customer's audio payload can be megabytes or unbounded (live
    iterator). Including it verbatim on ``CallRecord.raw_request`` would
    bloat memory and ship binary blobs to the cloud sink. We keep the
    config flags and replace ``request`` with a marker.
    """
    redacted: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "request":
            # Pre-recorded file path — bytes or iterator.
            if isinstance(v, (bytes, bytearray)):
                redacted["request"] = f"<redacted:{len(v)}_bytes>"
            else:
                redacted["request"] = "<redacted:audio_iterator>"
        elif k == "url":
            # Keep URL — it's small, not sensitive in the same way bytes are.
            redacted["url"] = v
        else:
            redacted[k] = v
    return redacted


def _build_record_from_response(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
    method: str,
) -> CallRecord:
    """Build a CallRecord from a Deepgram pre-recorded response.

    Deepgram pre-recorded response shape:
        ListenV1Response(
            metadata=ListenV1ResponseMetadata(
                duration: float (seconds, the billable unit),
                channels: int,
                models: List[str],
                ...
            ),
            results=ListenV1ResponseResults(...),
        )
    """
    # ``model`` arrives in kwargs (the request), but the metadata also
    # echoes it back via ``models: List[str]``. We prefer the request
    # value (deterministic regardless of server-side defaults).
    model = kwargs.get("model", "unknown")
    if isinstance(model, list) and model:
        model = model[0]
    if not isinstance(model, str):
        model = str(model) if model else "unknown"

    duration_seconds: float = 0.0
    channels: int = 1
    request_id: str | None = None

    metadata = _get_attr_or_key(response, "metadata")
    if metadata is not None:
        dur = _get_attr_or_key(metadata, "duration")
        if isinstance(dur, (int, float)) and dur >= 0:
            duration_seconds = float(dur)
        ch = _get_attr_or_key(metadata, "channels")
        if isinstance(ch, int) and ch >= 1:
            channels = ch
        rid = _get_attr_or_key(metadata, "request_id")
        if isinstance(rid, str):
            request_id = rid
        # Server-echoed model wins only if the request didn't carry one.
        if model == "unknown":
            models = _get_attr_or_key(metadata, "models")
            if isinstance(models, list) and models and isinstance(models[0], str):
                model = models[0]

    # User-facing output check: any non-empty transcript present?
    has_text_output = _response_has_transcript(response)

    diarize = bool(kwargs.get("diarize", False))
    multichannel = bool(kwargs.get("multichannel", False))

    raw_response_meta = {
        "duration_seconds": duration_seconds,
        "channels": channels,
        "request_id": request_id,
    }
    # non-token pricing dimension. Deepgram bills per-second of audio,
    # so ``dimension_kind="per_second"`` with ``dimension_value=duration``.
    # The ``model_specific_meta`` carries the flags the
    # ``audio_multichannel_doubling`` rule (deferred — see module docstring)
    # will dispatch on.
    usage_extra = {
        "dimension_kind": "per_second",
        "dimension_value": duration_seconds,
        "model_specific_meta": {
            "channels": channels,
            "diarize": diarize,
            "multichannel": multichannel,
        },
    }

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="deepgram",
        model=model,
        method=method,
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs, method=method),
        tool_calls=[],
        user_facing_output=has_text_output,
        raw_request=_redact_audio_bytes(kwargs),
        raw_response_meta=raw_response_meta,
        usage_extra=usage_extra,
    )


def _response_has_transcript(response: Any) -> bool:
    """Walk the Deepgram response to see if any alternative has text.

    Shape::

        response.results.channels[*].alternatives[*].transcript

    Tests may pass plain dicts; we walk defensively. An empty-string
    transcript counts as no output (a voicemail-short clip the model could
    not transcribe).
    """
    try:
        results = _get_attr_or_key(response, "results")
        if results is None:
            return False
        channels = _get_attr_or_key(results, "channels")
        if not channels:
            return False
        for channel in channels:
            alternatives = _get_attr_or_key(channel, "alternatives")
            if not alternatives:
                continue
            for alt in alternatives:
                transcript = _get_attr_or_key(alt, "transcript")
                if isinstance(transcript, str) and transcript.strip():
                    return True
    except Exception:
        pass
    return False


def _build_record_from_live(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    accumulator: _LiveAccumulator,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a live-streaming connection's terminal state.

    Emitted ONCE per connection, on close. See module docstring for the
    "per close, not per Final Result" decision rationale.
    """
    model = kwargs.get("model", "unknown")
    if isinstance(model, list) and model:
        model = model[0]
    if not isinstance(model, str):
        model = str(model) if model else "unknown"

    diarize = bool(kwargs.get("diarize", False))
    multichannel = bool(kwargs.get("multichannel", False))

    raw_response_meta = {
        "duration_seconds": accumulator.duration,
        "channels": accumulator.channels,
        "request_id": accumulator.request_id,
        "final_result_count": accumulator.final_result_count,
        "streamed": True,
    }
    usage_extra = {
        "dimension_kind": "per_second",
        "dimension_value": accumulator.duration,
        "model_specific_meta": {
            "channels": accumulator.channels,
            "diarize": diarize,
            "multichannel": multichannel,
        },
    }

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="deepgram",
        model=model,
        method=METHOD_TRANSCRIBE_LIVE,
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs, method=METHOD_TRANSCRIBE_LIVE),
        tool_calls=[],
        user_facing_output=accumulator.has_text_output,
        raw_request=_redact_audio_bytes(kwargs),
        raw_response_meta=raw_response_meta,
        usage_extra=usage_extra,
    )
