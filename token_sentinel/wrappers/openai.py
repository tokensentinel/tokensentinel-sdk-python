"""Wrap an OpenAI client to capture call records.

Pattern: in-place mutation of `client.chat.completions.create`,
`client.embeddings.create`, `client.audio.transcriptions.create` (Whisper STT),
and `client.audio.translations.create` (Whisper STT + auto-translate). The
original is captured in a closure so the instrumented version can delegate.
This preserves all type hints in IDEs because we mutate the live instance,
not a subclass.

Reference pattern: LangSmith's `wrap_openai` in langchain-ai/langsmith-sdk
(see: python/langsmith/wrappers/_openai.py). Both `OpenAI` and `AsyncOpenAI`
are supported here -- we detect async via `inspect.iscoroutinefunction` on
the original method.

Streaming (`stream=True`) is fully instrumented as of stable release: the wrapper
returns a proxy iterator that siphons each ``ChatCompletionChunk`` into a
usage accumulator, then flushes a CallRecord on iteration end / close / GC.
The proxy mirrors the patterns used by ``wrappers/anthropic.py`` (sync +
async iteration), ``wrappers/gemini.py`` (GeneratorExit suppression in
GC-driven closes), and ``wrappers/bedrock.py`` (tool-call stitching by
index across delta chunks). The block-mode warning machinery
(``_warn_block_mode_stream_once``) is retained as a defensive fallback for
the rare case where proxy construction itself fails (e.g. a non-iterable
return shape from a mocked SDK); under normal streaming usage it never
fires because the streams ARE instrumented.

Two OpenAI-streaming quirks the wrapper handles:

  1. Token usage is on the FINAL chunk only and ONLY when the user passes
     ``stream_options={"include_usage": True}``. Without that, the wrapper
     emits ``prompt_tokens=0, completion_tokens=0`` and sets
     ``raw_response_meta['usage_unavailable'] = True`` so customers can
     detect the situation in their leak handlers / dashboards.

  2. ``tool_calls`` arrive as ``ChoiceDeltaToolCall`` objects with an
     ``index`` field. The ``function.name`` typically lands on the first
     delta for that index, while ``function.arguments`` is a JSON-encoded
     string streamed in chunks. We stitch by index (mirrors Bedrock's
     ``_StreamUsageAccumulator._tool_blocks[idx]``) and ``json.loads`` the
     final argument string with a raw-string fallback.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import sys
import threading
import time
import uuid
import warnings
import weakref
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from token_sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# Block-mode-on-streaming warning bookkeeping (defensive fallback only)
# ---------------------------------------------------------------------------
#
# As of stable release, OpenAI streaming IS instrumented (see ``_OpenAIStreamProxy``
# below). The warning machinery is retained as a defensive fallback: if for
# any reason the proxy cannot be constructed (a mocked client returning a
# non-iterable, a future SDK shape change, etc.) we fall back to passthrough
# and warn under ``mode='block'`` so customers know their leak detection is
# bypassed for the uninstrumented stream.
#
# Requirements:
#   - Issued via ``warnings.warn(..., RuntimeWarning, stacklevel=2)`` so the
#     standard ``warnings.filterwarnings`` mechanism can silence it.
#   - Once per (Sentinel instance, sync/async path) -- not per call.
#
# We track via a ``WeakKeyDictionary`` keyed on the Sentinel instance so a
# torn-down Sentinel doesn't pin its entry. The value is a small set of
# path labels ("sync" / "async") so the sync and async branches each get
# one warning even on the same Sentinel.

_BLOCK_MODE_STREAM_MESSAGE = (
    "TokenSentinel: OpenAI streaming bypass -- mode='block' is not active "
    "on streamed chat.completions.create calls. Use Anthropic, Gemini, or "
    "Bedrock for block-mode-with-streaming. OpenAI streaming "
    "instrumentation is tracked for the roadmap."
)

_WARNED_INSTANCES: weakref.WeakKeyDictionary[Sentinel, set[str]] = weakref.WeakKeyDictionary()
# Lock guarding the membership test on _WARNED_INSTANCES. Without it, two
# threads racing on the first streamed call for a given Sentinel both pass
# the `path in seen` check before either adds the path -> both warn. Cosmetic
# (warning module is not an audit channel) but cheap to fix.
_WARN_LOCK = threading.Lock()


def _warn_block_mode_stream_once(sentinel: Sentinel, path: str) -> None:
    """Emit the block-mode-on-streaming warning at most once per (Sentinel, path).

    ``path`` is a small label ("sync" or "async"). The warning is suppressible
    via the standard ``warnings.filterwarnings(...)`` mechanism because we
    use ``warnings.warn`` with the ``RuntimeWarning`` category.
    """
    with _WARN_LOCK:
        try:
            seen = _WARNED_INSTANCES.setdefault(sentinel, set())
        except TypeError:
            # Defensive: WeakKeyDictionary requires the key be weakref-able.
            # Built-in ``Sentinel`` instances are; if a customer subclassed it
            # with __slots__ that exclude __weakref__, fall back to emitting
            # the warning unconditionally rather than crash their setup.
            warnings.warn(_BLOCK_MODE_STREAM_MESSAGE, RuntimeWarning, stacklevel=2)
            return
        if path in seen:
            return
        seen.add(path)
    # Emit OUTSIDE the lock so warning filters that do something expensive
    # (write to a file, etc.) don't serialise other Sentinels' first warnings.
    # Membership in `seen` has already been claimed.
    warnings.warn(_BLOCK_MODE_STREAM_MESSAGE, RuntimeWarning, stacklevel=2)


def wrap_openai(client: Any, sentinel: Sentinel) -> Any:
    """Wrap an OpenAI client. Mutates `chat.completions.create`,
    `embeddings.create`, and `audio.{transcriptions,translations}.create`.

    Supports both `openai.OpenAI` and `openai.AsyncOpenAI`. Detects async via
    `inspect.iscoroutinefunction` on the underlying method.

    : ``audio.transcriptions.create`` and ``audio.translations.create``
    (Whisper STT) are also patched. Both record a CallRecord with
    ``usage_extra.dimension_kind == "per_second"`` — Whisper bills per-second
    of audio just like Deepgram. The audio paths are ADDITIVE: the existing
    chat/embeddings instrumentation is unchanged.
    """
    _patch_chat_completions(client, sentinel)
    _patch_embeddings(client, sentinel)
    # Whisper paths. Defensive: ``client.audio`` may not be present on
    # older openai SDK versions (<1.0) or on trimmed mocks. Silent skip
    # rather than crash the customer's setup.
    _patch_audio_transcriptions(client, sentinel)
    _patch_audio_translations(client, sentinel)
    return client


# ---------------------------------------------------------------------------
# Streaming: usage accumulator + sync/async iterator proxies
# ---------------------------------------------------------------------------
#
# OpenAI's ``chat.completions.create(stream=True)`` returns a ``Stream``
# iterable that yields ``ChatCompletionChunk`` objects. Each chunk has the
# shape:
#
#     chunk.choices[0].delta.content                   -> str | None
#     chunk.choices[0].delta.tool_calls                -> list[ChoiceDeltaToolCall] | None
#     chunk.choices[0].finish_reason                   -> str | None (final chunk)
#     chunk.usage                                      -> CompletionUsage | None
#                                                         (final chunk only,
#                                                         iff stream_options
#                                                         .include_usage=True)
#
# Each ``ChoiceDeltaToolCall`` has:
#     dt.index                                         -> int (mandatory; stable per call)
#     dt.id                                            -> str | None (first delta only)
#     dt.function.name                                 -> str | None (first delta only)
#     dt.function.arguments                            -> str | None (incremental JSON chunks)
#
# We accumulate by ``index`` so multi-tool calls are stitched into the same
# shape the non-streaming wrapper produces. ``arguments`` is JSON-decoded with
# the same raw-string fallback used by the non-streaming path.


class _OpenAIUsageAccumulator:
    """Aggregate usage and content from OpenAI streaming chunks.

    Mirrors ``wrappers.bedrock._StreamUsageAccumulator`` (tool stitching by
    index) and ``wrappers.gemini._UsageAccumulator`` (defensive observation
    that never lets a malformed chunk crash user iteration).
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.has_text_output: bool = False
        self.finish_reason: Any = None
        # True iff we observed a final chunk that lacked a ``usage`` block --
        # which means the user did not pass ``stream_options.include_usage``
        # and the wrapper has no token information. We surface this in
        # ``raw_response_meta`` so customers can detect/dashboard the gap.
        self.usage_unavailable: bool = True
        # Pending tool blocks keyed by ``index``. Each entry:
        #     {"name": str, "arguments_str": str}
        # The name typically arrives on the first delta for that index; the
        # arguments string is concatenated incrementally across deltas.
        self._tool_blocks: dict[int, dict[str, Any]] = {}

    def observe(self, chunk: Any) -> None:
        try:
            self._observe_choices(chunk)
            self._observe_usage(chunk)
        except Exception:
            # Never let observation crash the user's iteration.
            pass

    def _observe_choices(self, chunk: Any) -> None:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish is not None:
            self.finish_reason = finish
        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            self.has_text_output = True
        tool_calls = getattr(delta, "tool_calls", None) or []
        for tc in tool_calls:
            self._observe_tool_call_delta(tc)

    def _observe_tool_call_delta(self, tc: Any) -> None:
        # OpenAI requires ``index`` on every ``ChoiceDeltaToolCall`` (it's the
        # only stable correlator across deltas). Fall back to 0 defensively.
        idx_raw = getattr(tc, "index", None)
        idx = idx_raw if isinstance(idx_raw, int) else 0
        block = self._tool_blocks.get(idx)
        if block is None:
            block = {"name": "", "arguments_str": ""}
            self._tool_blocks[idx] = block
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            if isinstance(name, str) and name and not block["name"]:
                # First delta for this index typically carries the name --
                # don't overwrite a name we've already captured.
                block["name"] = name
            arguments = getattr(fn, "arguments", None)
            if isinstance(arguments, str) and arguments:
                block["arguments_str"] += arguments

    def _observe_usage(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is None:
            return
        # Any chunk with a usage block satisfies the "usage_unavailable" gate.
        self.usage_unavailable = False
        prompt = getattr(usage, "prompt_tokens", None)
        if isinstance(prompt, int):
            self.input_tokens = max(self.input_tokens, prompt)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(completion, int):
            self.output_tokens = max(self.output_tokens, completion)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Materialise the accumulated tool_calls, parsing each arguments
        string with a raw-string fallback (matches the non-streaming wrapper's
        behavior on un-parseable JSON)."""
        result: list[dict[str, Any]] = []
        # Sort by index so multi-tool ordering is stable across runs.
        for idx in sorted(self._tool_blocks.keys()):
            block = self._tool_blocks[idx]
            raw = block.get("arguments_str", "")
            parsed: Any
            try:
                parsed = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                parsed = raw
            result.append({"name": block.get("name", ""), "arguments": parsed})
        return result


class _OpenAIStreamProxy:
    """Sync proxy around an OpenAI ``Stream``.

    Forwards attribute access to the wrapped stream so helpers like
    ``close()`` and ``response`` keep working. Re-implements ``__iter__``
    so each chunk is siphoned into the accumulator. Defined on the *class*
    -- Python's special-method lookup bypasses instance attributes for
    dunders, so this is the only correct place for it.
    """

    __slots__ = ("_stream", "_accumulator", "_finalize", "_finalized")

    def __init__(self, stream: Any, accumulator: _OpenAIUsageAccumulator, finalize: Any) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_finalize", finalize)
        object.__setattr__(self, "_finalized", False)

    def __iter__(self) -> Any:
        accumulator = self._accumulator
        try:
            for chunk in self._stream:
                accumulator.observe(chunk)
                yield chunk
        finally:
            self._safe_finalize()

    def close(self) -> Any:
        # OpenAI's Stream exposes ``close()`` to abort iteration early. Forward
        # to the underlying stream and finalize.
        try:
            closer = getattr(self._stream, "close", None)
            if callable(closer):
                return closer()
        finally:
            self._safe_finalize()

    def __enter__(self) -> Any:
        # OpenAI's Stream supports ``with stream:`` (it implements __enter__
        # / __exit__ delegating to close()). Forward where possible and
        # always return self so user code keeps working.
        try:
            entry = getattr(self._stream, "__enter__", None)
            if callable(entry):
                entry()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            exiter = getattr(self._stream, "__exit__", None)
            if callable(exiter):
                try:
                    return exiter(exc_type, exc, tb)
                except Exception:
                    return False
        finally:
            self._safe_finalize()
        return False

    def __del__(self) -> None:
        # Last-resort flush. Users who break out of iteration early without
        # calling close() / leaving the with-block still get their CallRecord
        # at GC time. ``__del__`` must never raise -- Python silently
        # swallows __del__ exceptions anyway, which means LeakDetected from
        # block mode would vanish without the user's app halting. Suppress
        # LeakDetected explicitly here and emit a warning so the user knows
        # block mode was best-effort on this path.
        try:
            self._safe_finalize(suppress_block=True)
        except Exception:
            pass

    def _safe_finalize(self, *, suppress_block: bool = False) -> None:
        # Only finalize once even if the user iterates the proxy more than
        # once or hits both close() and __exit__.
        if object.__getattribute__(self, "_finalized"):
            return
        object.__setattr__(self, "_finalized", True)
        # Detect generator-close path: when this finalize runs from the
        # ``finally`` of an iterator whose try-block was exited via
        # close()/GC, GeneratorExit is in flight. Python silently swallows
        # non-GeneratorExit exceptions raised in that state -- meaning
        # LeakDetected from block mode would vanish without halting the
        # user's app. Suppress with a warning instead, matching the Gemini
        # wrapper's GC-suppression pattern.
        is_close = sys.exc_info()[0] is GeneratorExit
        finalize = object.__getattribute__(self, "_finalize")
        try:
            finalize(self._accumulator)
        except LeakDetected:
            if is_close or suppress_block:
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in stream close path "
                    "(block mode is best-effort on abandoned streams; use "
                    "'with stream:' or fully iterate to get block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        # __getattr__ fires only on lookup miss, so this transparently
        # exposes anything the SDK's Stream offers (``response``, internal
        # cursors, etc.) without us enumerating it.
        return getattr(self._stream, name)


class _AsyncOpenAIStreamProxy:
    """Async counterpart of ``_OpenAIStreamProxy``.

    Wraps an ``AsyncStream`` (returned from ``await
    AsyncOpenAI().chat.completions.create(stream=True)``). Re-implements
    ``__aiter__`` so each chunk is siphoned into the accumulator; everything
    else is forwarded.
    """

    __slots__ = ("_stream", "_accumulator", "_finalize", "_finalized")

    def __init__(self, stream: Any, accumulator: _OpenAIUsageAccumulator, finalize: Any) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_finalize", finalize)
        object.__setattr__(self, "_finalized", False)

    def __aiter__(self) -> Any:
        # ``__aiter__`` is the *sync* hook in the async iteration protocol --
        # it must return an async iterator. We return an async generator,
        # which is the simplest async iterator Python can construct.
        return self._aiter_impl()

    async def _aiter_impl(self) -> Any:
        accumulator = self._accumulator
        try:
            async for chunk in self._stream:
                accumulator.observe(chunk)
                yield chunk
        finally:
            self._safe_finalize()

    async def aclose(self) -> Any:
        # OpenAI's AsyncStream exposes ``aclose()`` for early abort.
        try:
            closer = getattr(self._stream, "aclose", None)
            if callable(closer):
                res = closer()
                if inspect.isawaitable(res):
                    return await res
                return res
        finally:
            self._safe_finalize()

    async def __aenter__(self) -> Any:
        try:
            entry = getattr(self._stream, "__aenter__", None)
            if callable(entry):
                res = entry()
                if inspect.isawaitable(res):
                    await res
        except Exception:
            pass
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            exiter = getattr(self._stream, "__aexit__", None)
            if callable(exiter):
                try:
                    res = exiter(exc_type, exc, tb)
                    if inspect.isawaitable(res):
                        return await res
                    return res
                except Exception:
                    return False
        finally:
            self._safe_finalize()
        return False

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
        finalize = object.__getattribute__(self, "_finalize")
        try:
            finalize(self._accumulator)
        except LeakDetected:
            if is_close or suppress_block:
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in stream close path "
                    "(block mode is best-effort on abandoned streams; use "
                    "'async with stream:' or fully iterate to get block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# chat.completions.create -- sync and async (with streaming instrumentation)
# ---------------------------------------------------------------------------


def _patch_chat_completions(client: Any, sentinel: Sentinel) -> None:
    original_create = client.chat.completions.create

    if inspect.iscoroutinefunction(original_create):

        @functools.wraps(original_create)
        async def instrumented_create_async(*args: Any, **kwargs: Any) -> Any:
            session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))

            if kwargs.get("stream") is True:
                return await _instrumented_async_stream(
                    original_create=original_create,
                    sentinel=sentinel,
                    session_id=session_id,
                    args=args,
                    kwargs=kwargs,
                )

            start = time.perf_counter()
            try:
                response = await original_create(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

            try:
                record = _build_chat_record(
                    session_id=session_id,
                    kwargs=kwargs,
                    response=response,
                    latency_ms=elapsed_ms,
                )
            except Exception:
                return response
            # record_call: propagate LeakDetected (block mode), swallow others.
            try:
                sentinel.record_call(record)
            except LeakDetected:
                raise
            except Exception:
                pass
            return response

        client.chat.completions.create = instrumented_create_async
        return

    @functools.wraps(original_create)
    def instrumented_create(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))

        if kwargs.get("stream") is True:
            return _instrumented_sync_stream(
                original_create=original_create,
                sentinel=sentinel,
                session_id=session_id,
                args=args,
                kwargs=kwargs,
            )

        start = time.perf_counter()
        try:
            response = original_create(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_chat_record(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return response
        # record_call: propagate LeakDetected (block mode), swallow others.
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass
        return response

    client.chat.completions.create = instrumented_create


def _instrumented_sync_stream(
    *,
    original_create: Any,
    sentinel: Sentinel,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Construct a sync streaming proxy. Falls back to passthrough + warn
    (under block mode) if proxy construction fails for any reason."""
    start = time.perf_counter()
    captured_kwargs = dict(kwargs)
    # Let the underlying SDK call propagate exceptions -- if the SDK itself
    # rejects the kwargs (bad model, etc.), we don't try to record anything.
    raw_stream = original_create(*args, **kwargs)

    accumulator = _OpenAIUsageAccumulator()

    def finalize(acc: _OpenAIUsageAccumulator) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            record = _build_record_from_accumulator(
                session_id=session_id,
                kwargs=captured_kwargs,
                accumulator=acc,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return
        # record_call may raise LeakDetected in block mode -- propagate;
        # swallow other exceptions (rule/handler bugs must not crash).
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass

    try:
        return _OpenAIStreamProxy(raw_stream, accumulator, finalize)
    except Exception:
        # Defensive fallback: if proxy construction fails (a non-iterable
        # mock, a future SDK shape change, etc.) hand back the raw stream
        # and warn under block mode so the customer knows their leak
        # detection is bypassed for this surface.
        if sentinel.mode == "block":
            _warn_block_mode_stream_once(sentinel, "sync")
        return raw_stream


async def _instrumented_async_stream(
    *,
    original_create: Any,
    sentinel: Sentinel,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Construct an async streaming proxy. Falls back to passthrough + warn
    (under block mode) if proxy construction fails for any reason."""
    start = time.perf_counter()
    captured_kwargs = dict(kwargs)
    # ``client.chat.completions.create(stream=True)`` is itself a coroutine
    # function on AsyncOpenAI; await it once to get the AsyncStream.
    raw_stream = await original_create(*args, **kwargs)

    accumulator = _OpenAIUsageAccumulator()

    def finalize(acc: _OpenAIUsageAccumulator) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            record = _build_record_from_accumulator(
                session_id=session_id,
                kwargs=captured_kwargs,
                accumulator=acc,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass

    try:
        return _AsyncOpenAIStreamProxy(raw_stream, accumulator, finalize)
    except Exception:
        if sentinel.mode == "block":
            _warn_block_mode_stream_once(sentinel, "async")
        return raw_stream


def _patch_embeddings(client: Any, sentinel: Sentinel) -> None:
    original_create = client.embeddings.create

    if inspect.iscoroutinefunction(original_create):

        @functools.wraps(original_create)
        async def instrumented_embed_async(*args: Any, **kwargs: Any) -> Any:
            session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
            start = time.perf_counter()
            try:
                response = await original_create(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

            try:
                record = _build_embedding_record(
                    session_id=session_id,
                    kwargs=kwargs,
                    response=response,
                    latency_ms=elapsed_ms,
                )
            except Exception:
                return response
            # record_call: propagate LeakDetected (block mode), swallow others.
            try:
                sentinel.record_call(record)
            except LeakDetected:
                raise
            except Exception:
                pass
            return response

        client.embeddings.create = instrumented_embed_async
        return

    @functools.wraps(original_create)
    def instrumented_embed(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_create(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_embedding_record(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
            )
        except Exception:
            return response
        # record_call: propagate LeakDetected (block mode), swallow others.
        try:
            sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass
        return response

    client.embeddings.create = instrumented_embed


def _build_chat_record(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
) -> CallRecord:
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)

    request_hash = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    tool_calls: list[dict[str, Any]] = []
    has_text_output = False

    choices = getattr(response, "choices", []) or []
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn is not None else ""
                raw_args = getattr(fn, "arguments", "") if fn is not None else ""
                # OpenAI tool call arguments are a JSON-encoded string. Parse to
                # a dict so downstream rules (tool_loop) can hash/compare them
                # the same way they do for Anthropic tool_use blocks.
                parsed_args: Any
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except (TypeError, ValueError):
                    parsed_args = raw_args
                tool_calls.append({"name": name, "arguments": parsed_args})

            content = getattr(message, "content", None)
            if content:
                has_text_output = True

    user_facing_output = has_text_output and not tool_calls

    finish_reason = None
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model=model,
        method="chat.completions.create",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        raw_request={"messages": messages, "tools": tools, "max_tokens": max_tokens},
        raw_response_meta={"finish_reason": finish_reason},
    )


def _build_record_from_accumulator(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    accumulator: _OpenAIUsageAccumulator,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a streamed chat completion accumulator.

    Mirrors ``_build_chat_record`` for the streaming path: same provider,
    method, request_hash shape, and tool_call structure. ``raw_response_meta``
    additionally carries ``streamed=True`` and ``usage_unavailable`` so
    downstream rules and dashboards can distinguish streaming records and
    flag calls where ``stream_options.include_usage`` was not set.
    """
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)

    request_hash = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    tool_calls = accumulator.tool_calls
    user_facing_output = accumulator.has_text_output and not tool_calls

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model=model,
        method="chat.completions.create",
        prompt_tokens=accumulator.input_tokens,
        completion_tokens=accumulator.output_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        raw_request={"messages": messages, "tools": tools, "max_tokens": max_tokens},
        raw_response_meta={
            "finish_reason": accumulator.finish_reason,
            "streamed": True,
            "usage_unavailable": accumulator.usage_unavailable,
        },
    )


def _build_embedding_record(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
) -> CallRecord:
    model = kwargs.get("model", "unknown")
    input_value = kwargs.get("input", "")

    # The embedding_waste rule hashes the `input` to detect re-embeds. Hash the
    # same shape we expose so request_hash is stable.
    request_hash = hashlib.sha256(
        json.dumps(
            {"model": model, "input": input_value},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model=model,
        method="embeddings.create",
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=False,
        raw_request={"input": input_value, "model": model},
        raw_response_meta={},
    )


# ---------------------------------------------------------------------------
# audio.transcriptions.create / audio.translations.create — Whisper STT
# ---------------------------------------------------------------------------
#
# OpenAI's Whisper API is reached via:
#
#   client.audio.transcriptions.create(file=..., model="whisper-1", ...)
#       -> STT (audio in, text out in the audio's source language)
#   client.audio.translations.create(file=..., model="whisper-1", ...)
#       -> STT + auto-translate to English
#
# Both bill per-SECOND of audio (same pricing dimension as Deepgram). The
# CallRecord:
#
#   - prompt_tokens = 0 (audio input has no token concept)
#   - completion_tokens = char count of the transcribed text (proxy)
#   - usage_extra.dimension_kind = "per_second"
#   - usage_extra.dimension_value = audio duration in seconds
#
# Duration extraction strategy (in order of preference):
#   1. If the response is a verbose JSON object with a top-level ``duration``
#      field (response_format="verbose_json"), use it. This is authoritative.
#   2. Otherwise, attempt ``ffmpeg.probe`` / ``mutagen`` style metadata
#      extraction from the file argument — NOT done in  because it
#      drags in a heavyweight dep. We fall through to (3).
#   3. Default to 0.0 and set ``usage_extra.dimension_unavailable = True``
#      so dashboards can detect the gap.
#
# raw_request handling:
#   - The ``file`` kwarg may be a path string, a tuple, an open file
#     handle, or raw bytes. We replace it with a redaction marker that
#     preserves the byte length when known (a multi-MB audio file should
#     not ride along on every CallRecord).
#   - ``model`` / ``response_format`` / ``language`` are kept verbatim
#     (small strings, no PII risk).
#
# Failure isolation matches the chat/embeddings path: record-build errors
# are swallowed, record_call errors propagate only LeakDetected.


def _patch_audio_transcriptions(client: Any, sentinel: Sentinel) -> None:
    """Patch ``client.audio.transcriptions.create`` if it exists.

    Defensive: ``client.audio`` may not be present (older openai SDK,
    trimmed mock). Silent skip in that case — the rest of the wrapper
    is unaffected.
    """
    audio = getattr(client, "audio", None)
    if audio is None:
        return
    transcriptions = getattr(audio, "transcriptions", None)
    if transcriptions is None:
        return
    _patch_audio_create(
        accessor=transcriptions,
        sentinel=sentinel,
        method_label="audio.transcriptions.create",
    )


def _patch_audio_translations(client: Any, sentinel: Sentinel) -> None:
    """Patch ``client.audio.translations.create`` if it exists. Mirrors
    ``_patch_audio_transcriptions`` — same defensive posture."""
    audio = getattr(client, "audio", None)
    if audio is None:
        return
    translations = getattr(audio, "translations", None)
    if translations is None:
        return
    _patch_audio_create(
        accessor=translations,
        sentinel=sentinel,
        method_label="audio.translations.create",
    )


def _patch_audio_create(
    *,
    accessor: Any,
    sentinel: Sentinel,
    method_label: str,
) -> None:
    """Replace ``accessor.create`` with an instrumented variant.

    ``accessor`` is either ``client.audio.transcriptions`` or
    ``client.audio.translations``. Both expose a ``create`` method with
    the same call signature (``file``, ``model``, optional kwargs).
    """
    original = getattr(accessor, "create", None)
    if original is None or not callable(original):
        return

    if inspect.iscoroutinefunction(original):

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
                record = _build_audio_record(
                    session_id=session_id,
                    kwargs=captured_kwargs,
                    response=response,
                    latency_ms=elapsed_ms,
                    method_label=method_label,
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
            accessor.create = instrumented_async
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
            record = _build_audio_record(
                session_id=session_id,
                kwargs=captured_kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method_label=method_label,
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
        accessor.create = instrumented_sync
    except Exception:
        return


def _build_audio_record(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
    method_label: str,
) -> CallRecord:
    """Build a CallRecord from a Whisper transcription/translation response.

    Per the specification:
      - ``prompt_tokens`` = 0 (audio in, no token concept)
      - ``completion_tokens`` = char count of the transcribed text (proxy)
      - ``usage_extra.dimension_kind`` = "per_second"
      - ``usage_extra.dimension_value`` = audio duration (from response
        when ``response_format="verbose_json"``, else 0.0)

    Response shapes Whisper returns depending on ``response_format``:
      - "json" (default) / "text"  -> ``response.text`` only
      - "verbose_json"             -> ``response.text`` + ``response.duration`` +
                                     ``response.language`` + ``response.segments``
      - "srt" / "vtt"              -> raw string
      - stream=True                -> iterable of segments; the response
                                     object has no ``.duration``. We
                                     fall back to ``_probe_streaming_audio_duration``
                                     on the captured audio bytes .
    """
    model = kwargs.get("model", "unknown")
    if not isinstance(model, str):
        model = str(model)
    response_format = kwargs.get("response_format")
    language = kwargs.get("language")
    is_streaming = bool(kwargs.get("stream"))

    transcript_text = _extract_audio_text(response)
    completion_tokens = len(transcript_text)

    duration_seconds, dimension_unavailable = _extract_audio_duration(response)

    # streaming branch — streamed Whisper responses have no
    # ``response.duration`` at all (the response is an iterator of
    # segments, not a single object). ``_probe_audio_duration``
    # path took the ``file`` kwarg, but for streaming the SDK typically
    # captures audio bytes BEFORE handing the iterator back — so we
    # probe the captured bytes via the dedicated streaming helper that
    # ALSO surfaces the ``streaming_realtime`` flag when no bytes are
    # available (the real-time-microphone case).
    streaming_realtime = False
    streaming_evidence: dict[str, Any] = {}
    if is_streaming and dimension_unavailable:
        captured_bytes, mime_hint = _captured_audio_bytes(kwargs.get("file"))
        if captured_bytes is None:
            # Real-time microphone path — bytes don't exist before the
            # call returns. We can't probe; surface the flag so
            # dashboards distinguish "we tried and failed" from "this
            # was real-time streaming and probing isn't possible".
            streaming_realtime = True
            streaming_evidence["streaming_realtime"] = True
        else:
            probed = _probe_streaming_audio_duration(captured_bytes, mime_type=mime_hint)
            if probed is not None:
                duration_seconds = probed
                dimension_unavailable = False
            else:
                # Probe failed (mutagen absent / cannot parse). Keep
                # ``dimension_unavailable=True`` and record the byte
                # count so the operator can spot patterns in the gap.
                streaming_evidence["streaming_probe_failed"] = True
                streaming_evidence["streaming_byte_count"] = len(captured_bytes)
    elif not is_streaming and dimension_unavailable:
        # fallback: if Whisper didn't return ``response.duration``
        # (which only ships on ``response_format="verbose_json"``), try
        # to probe the audio file itself via the ``mutagen`` library.
        # Customers who use response_format=text/srt/vtt/json otherwise
        # lose all duration telemetry, which means downstream
        # per-second cost dashboards have holes. mutagen is an optional
        # dep — if it's not installed we keep the
        # ``dimension_unavailable=True`` path as the final fallback.
        probed = _probe_audio_duration(kwargs.get("file"))
        if probed is not None:
            duration_seconds = probed
            dimension_unavailable = False

    request_hash = hashlib.sha256(
        json.dumps(
            {
                "method": method_label,
                "model": model,
                "response_format": response_format,
                "language": language,
                "stream": is_streaming,
                # Don't include the audio bytes in the hash input — they
                # can be large. We hash a per-call marker so identical
                # consecutive transcribe calls (retry storm) hash to
                # the same value only when their kwargs match.
                "file_marker": _file_marker(kwargs.get("file")),
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    raw_request = _redact_audio_file(kwargs)
    raw_response_meta: dict[str, Any] = {
        "duration_seconds": duration_seconds,
        "transcript_chars": completion_tokens,
    }
    if is_streaming:
        raw_response_meta["streamed"] = True
        if streaming_realtime:
            raw_response_meta["streaming_realtime"] = True
        for k, v in streaming_evidence.items():
            raw_response_meta[k] = v
    if dimension_unavailable:
        # Surface the gap so dashboards can detect "we don't know the
        # billable duration on this call" (the customer didn't pass
        # response_format="verbose_json", and we don't probe the audio
        # file itself).
        raw_response_meta["dimension_unavailable"] = True

    usage_extra = {
        "dimension_kind": "per_second",
        "dimension_value": duration_seconds,
        "model_specific_meta": {
            "response_format": response_format,
            "language": language,
            "stream": is_streaming,
        },
    }

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="openai",
        model=model,
        method=method_label,
        prompt_tokens=0,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        # The transcribed text IS the user-facing output of an STT call.
        user_facing_output=bool(transcript_text),
        raw_request=raw_request,
        raw_response_meta=raw_response_meta,
        usage_extra=usage_extra,
    )


def _extract_audio_text(response: Any) -> str:
    """Extract the transcript string from a Whisper response.

    Whisper response shapes:
      - response_format="json" / "verbose_json": pydantic object with .text
      - response_format="text": raw str
      - response_format="srt" / "vtt": raw str
    Defensive: accept str, object-with-.text, and dict.
    """
    try:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            text = response.get("text")
            return text if isinstance(text, str) else ""
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else ""
    except Exception:
        return ""


def _extract_audio_duration(response: Any) -> tuple[float, bool]:
    """Return (duration_seconds, dimension_unavailable).

    Whisper exposes ``duration`` only on ``response_format="verbose_json"``
    responses. We try to read it (defensively, accepting both pydantic
    objects and dicts); if it isn't present we return ``(0.0, True)`` so
    the caller can attempt the  ``_probe_audio_duration`` fallback
    (mutagen-based audio-file metadata probing). If that fallback also
    fails (mutagen not installed, or unable to parse the file), the
    record carries ``dimension_unavailable=True`` for the
    customers' dashboards.
    """
    try:
        if isinstance(response, dict):
            dur = response.get("duration")
        else:
            dur = getattr(response, "duration", None)
        if isinstance(dur, (int, float)) and dur >= 0:
            return float(dur), False
    except Exception:
        pass
    return 0.0, True


def _probe_audio_duration(file_arg: Any) -> float | None:
    """fallback: read audio duration from the file via ``mutagen``.

    Returns the duration in seconds on success, ``None`` on any failure
    (mutagen not installed, file unreadable, format unsupported, parse
    error). The caller treats ``None`` as "fall through to the
    ``dimension_unavailable=True`` path", so all errors here are
    silent — instrumentation must NEVER break the user's call, and a
    missing optional dependency is the canonical failure mode.

    ``file_arg`` is the ``file`` kwarg the customer passed to
    ``transcriptions.create`` / ``translations.create``. Per the OpenAI
    SDK convention it can be:

      - a path string (``"audio.mp3"``) — pass directly to mutagen
      - a Python file object (``open("audio.mp3", "rb")``) — pass
        directly; mutagen reads from the underlying buffer
      - raw bytes — wrap in BytesIO and pass; mutagen sniffs the format
        from the magic header
      - a ``(filename, content, mime)`` tuple — extract ``content`` and
        wrap in BytesIO (the filename is metadata only; mutagen sniffs)

    For file objects and bytes paths we reset the read cursor afterwards
    when possible (``.seek(0)``) so the original ``file_arg`` is in the
    same state the customer left it — the upstream OpenAI SDK has
    already consumed and uploaded the buffer at this point, but a
    defensive reset costs nothing.

    On ImportError, we return ``None`` silently with NO warning: the
    optional-dependency contract is "install ``token-sentinel[audio-
    metadata]`` if you want this", not "we'll nag you about it". The
    ``dimension_unavailable`` flag still surfaces the gap on the
    dashboard, which is the customer-facing signal.
    """
    if file_arg is None:
        return None

    # Late-import mutagen so the wrapper module itself does not have a
    # hard dependency. Customers who don't install the audio-metadata
    # extra get None back and the dimension_unavailable flag stays set.
    try:
        import mutagen  # type: ignore[import-not-found]
    except ImportError:
        return None
    except Exception:
        # Any other import-time failure (transitive dep, broken install)
        # also degrades silently.
        return None

    try:
        target = _file_arg_for_mutagen(file_arg)
        if target is None:
            return None
        info = mutagen.File(target)  # type: ignore[attr-defined]
        if info is None:
            return None
        length = getattr(getattr(info, "info", None), "length", None)
        if isinstance(length, (int, float)) and length >= 0:
            return float(length)
    except Exception:
        # mutagen raised mid-parse — fall through to None.
        return None
    return None


# Whisper's documented file-size cap is 25 MB. If captured audio bytes
# exceed this, the OpenAI server would have rejected the call anyway —
# probing them locally wastes work and (in pathological cases) memory.
# Surfaced as a module constant so tests can assert the boundary.
_WHISPER_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


def _probe_streaming_audio_duration(
    audio_bytes: bytes | bytearray,
    *,
    mime_type: str | None = None,
) -> float | None:
    """streaming fallback: probe duration from captured audio bytes.

    Companion to :func:`_probe_audio_duration` for the streaming path.
    The OpenAI Whisper streaming response is an iterator of segments;
    no single ``response.duration`` is available. For SDK callers that
    pass a file-like or bytes payload, the wrapper captures those
    bytes BEFORE the streamed iterator returns — meaning we can probe
    duration via the SAME mutagen-based metadata extraction the
    non-streaming  path uses.

    Real-time microphone streaming is the one case this helper can't
    handle: the bytes simply don't exist before the call returns
    because they're sourced from an ongoing recorder. The caller
    (``_build_audio_record``) detects that via :func:`_captured_audio_bytes`
    returning ``None`` and surfaces ``streaming_realtime=True`` in the
    record's evidence instead of calling this function.

    Returns the probed duration in seconds, or ``None`` on any
    failure (mutagen absent, format unrecognised, parse error, or the
    byte buffer exceeds Whisper's documented 25 MB cap). All errors
    are silent — instrumentation must never break the user's call.

    ``mime_type`` is currently advisory only — mutagen sniffs the
    format from the magic header. We accept the kwarg so future
    audio-metadata libraries that benefit from a hint can plug in
    without changing the call site.
    """
    if audio_bytes is None:
        return None
    if not isinstance(audio_bytes, (bytes, bytearray)):
        return None
    # Empty bytes are not probe-able and certainly aren't valid audio.
    if len(audio_bytes) == 0:
        return None
    # Respect Whisper's file-size cap. Bytes beyond this would have
    # been rejected by the API anyway; probing locally is wasted work.
    if len(audio_bytes) > _WHISPER_MAX_FILE_SIZE_BYTES:
        return None

    try:
        import mutagen
    except ImportError:
        return None
    except Exception:
        return None

    try:
        import io

        buf = io.BytesIO(bytes(audio_bytes))
        info = mutagen.File(buf)  # type: ignore[attr-defined]
        if info is None:
            return None
        length = getattr(getattr(info, "info", None), "length", None)
        if isinstance(length, (int, float)) and length >= 0:
            return float(length)
    except Exception:
        return None
    return None


def _captured_audio_bytes(
    file_arg: Any,
) -> tuple[bytes | None, str | None]:
    """Extract raw audio bytes from a streaming ``file`` kwarg.

    Returns ``(bytes, mime_type_hint_or_None)`` when bytes are
    available, or ``(None, None)`` for the real-time-microphone
    case (the SDK was passed a non-seekable / synthesised stream
    and bytes don't exist before the call returns).

    Supported shapes (mirrors :func:`_file_arg_for_mutagen`):

      - raw bytes / bytearray         -> bytes returned verbatim
      - ``(filename, content, mime)`` tuple where ``content`` is
        bytes or has ``.read()``       -> bytes + mime hint
      - file-like object with seek+
        read (a regular file open in
        rb mode)                       -> read the buffer
      - path string                    -> read the file off disk
                                          (best-effort; failures
                                          collapse to (None, None))

    Anything else — a real-time microphone source, a generator, an
    object without seek/read — returns ``(None, None)`` so the caller
    can mark ``streaming_realtime=True``.
    """
    if file_arg is None:
        return None, None

    # Raw bytes — easy path. No mime hint available.
    if isinstance(file_arg, (bytes, bytearray)):
        return bytes(file_arg), None

    # OpenAI SDK tuple shape: (filename, content, mime).
    if isinstance(file_arg, tuple) and len(file_arg) >= 2:
        content = file_arg[1]
        mime = str(file_arg[2]) if len(file_arg) >= 3 and isinstance(file_arg[2], str) else None
        if isinstance(content, (bytes, bytearray)):
            return bytes(content), mime
        # File-like content — read the buffer (best effort).
        return _read_seekable_bytes(content), mime

    # Path string — read off disk (the SDK accepts paths and the
    # bytes-on-wire flow opens them server-side). We open with rb so
    # mutagen sees the raw stream; non-readable paths collapse to None.
    if isinstance(file_arg, str):
        try:
            # Cap at the Whisper file-size limit + 1 byte so we surface
            # oversize files via the probe's own cap check rather than
            # loading 10 GB into memory if the customer points at a
            # WAV mirror of their entire podcast catalogue.
            with open(file_arg, "rb") as fh:
                data = fh.read(_WHISPER_MAX_FILE_SIZE_BYTES + 1)
            return data, None
        except Exception:
            return None, None

    # File-like object — read the underlying buffer.
    return _read_seekable_bytes(file_arg), None


def _read_seekable_bytes(obj: Any) -> bytes | None:
    """Best-effort read of a file-like ``obj`` into bytes.

    Returns ``None`` if the object isn't readable or seek-able in the
    way we need (a real-time microphone source typically isn't). All
    exceptions collapse to ``None`` — instrumentation must never
    propagate IO errors back to the caller's audio path.
    """
    try:
        read = getattr(obj, "read", None)
        if not callable(read):
            return None
        # Best-effort rewind so we read from the start. Some streaming
        # sources don't support seek — that's fine, we just read what's
        # available. If the previous reader already drained the buffer
        # we'll get b"" back and surface that as None below.
        try:
            seek = getattr(obj, "seek", None)
            if callable(seek):
                seek(0)
        except Exception:
            pass
        data = read()
        if not isinstance(data, (bytes, bytearray)):
            return None
        if len(data) == 0:
            return None
        return bytes(data)
    except Exception:
        return None


def _file_arg_for_mutagen(file_arg: Any) -> Any:
    """Normalise the OpenAI SDK's ``file`` kwarg into something mutagen
    can read.

    Returns ``None`` if the shape is unsupported (caller then falls
    back to the dimension_unavailable path).
    """
    # Path string: pass through; mutagen.File accepts strings.
    if isinstance(file_arg, str):
        return file_arg

    # OpenAI SDK tuple shape: (filename, content, mime). Content can be
    # bytes or a file-like object. Filename is purely metadata.
    if isinstance(file_arg, tuple) and len(file_arg) >= 2:
        content = file_arg[1]
        if isinstance(content, (bytes, bytearray)):
            import io

            return io.BytesIO(bytes(content))
        # Treat anything else (file object, BytesIO, etc.) as a buffer.
        return _seekable(content)

    # Raw bytes: wrap in BytesIO so mutagen can sniff.
    if isinstance(file_arg, (bytes, bytearray)):
        import io

        return io.BytesIO(bytes(file_arg))

    # File-like object: pass through with a best-effort seek(0).
    return _seekable(file_arg)


def _seekable(obj: Any) -> Any:
    """Best-effort rewind so mutagen reads from the start of the buffer.

    OpenAI SDK has already consumed the buffer at this point; calling
    ``seek(0)`` makes a re-read possible. If the object doesn't expose
    ``seek``, we hand it to mutagen anyway — mutagen will fail
    gracefully and our caller returns None.
    """
    try:
        seek = getattr(obj, "seek", None)
        if callable(seek):
            seek(0)
    except Exception:
        pass
    return obj


def _file_marker(file_arg: Any) -> str:
    """Build a stable marker for the ``file`` kwarg, NOT including bytes.

    The ``file`` kwarg can be:
      - a path string ("audio.mp3") -> use the string
      - a (filename, fileobj, mimetype) tuple -> use the filename
      - an open file object -> use its name attribute if any
      - raw bytes -> hash a small prefix
      - None / missing -> ""

    We do NOT hash the full bytes (could be megabytes) — a length-bounded
    marker suffices for retry-storm detection. Two consecutive calls
    with the same file kwarg produce the same marker, which is what the
    rule machinery needs.
    """
    if file_arg is None:
        return ""
    if isinstance(file_arg, str):
        return file_arg
    if isinstance(file_arg, tuple) and file_arg:
        first = file_arg[0]
        return first if isinstance(first, str) else "tuple"
    if isinstance(file_arg, (bytes, bytearray)):
        # Hash a length prefix; identical bytes => identical marker.
        digest = hashlib.sha256(bytes(file_arg)[:1024]).hexdigest()[:16]
        return f"bytes:{len(file_arg)}:{digest}"
    name = getattr(file_arg, "name", None)
    if isinstance(name, str):
        return name
    return f"fileobj:{type(file_arg).__name__}"


def _redact_audio_file(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a ``raw_request`` dict that strips the audio file bytes.

    The customer's audio payload can be megabytes; including it verbatim on
    ``CallRecord.raw_request`` would bloat memory and ship binary blobs to
    the cloud sink. We keep the config flags and replace ``file`` with a
    marker.
    """
    redacted: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "file":
            if isinstance(v, (bytes, bytearray)):
                redacted["file"] = f"<redacted:{len(v)}_bytes>"
            elif isinstance(v, str):
                # Path string is small and not sensitive in the same way
                # bytes are — keep it for triage.
                redacted["file"] = v
            elif isinstance(v, tuple) and v:
                tuple_name: str = v[0] if isinstance(v[0], str) else "tuple"
                redacted["file"] = f"<redacted:tuple:{tuple_name}>"
            else:
                file_name = getattr(v, "name", None)
                if isinstance(file_name, str):
                    redacted["file"] = f"<redacted:fileobj:{file_name}>"
                else:
                    redacted["file"] = f"<redacted:{type(v).__name__}>"
        else:
            redacted[k] = v
    return redacted
