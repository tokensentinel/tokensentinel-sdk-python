"""Wrap an Anthropic client to capture call records.

Pattern: in-place mutation of `client.messages.create` and `client.messages.stream`.
The original is captured in a closure so the instrumented version can delegate.
This preserves all type hints in IDEs because we mutate the live instance, not a
subclass.

Reference pattern: LangSmith's `wrap_anthropic` in langchain-ai/langsmith-sdk
(see: python/langsmith/wrappers/_anthropic.py, lines ~327–537 for
`MessageStreamWrapper` / `AsyncMessageStreamWrapper`).

Four cases handled by `wrap_anthropic`:
  1. Sync non-streaming: `client.messages.create(...)` -> Message
  2. Sync streaming: `client.messages.stream(...)` -> ctx mgr yielding MessageStream
  3. Async non-streaming: `AsyncAnthropic().messages.create(...)` -> awaitable Message
  4. Async streaming: `AsyncAnthropic().messages.stream(...)` -> async ctx mgr

For streams, the `CallRecord` is built when the stream finalizes (`__exit__` /
`__aexit__`). Token usage is aggregated from `message_delta` events' `usage`
deltas plus the final `MessageStopEvent`'s `message.usage`. Failures in
instrumentation never propagate to user code.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from anthropic import Anthropic

    from token_sentinel.sentinel import Sentinel


def wrap_anthropic(client: Anthropic, sentinel: Sentinel) -> Anthropic:
    """Wrap an Anthropic client. Mutates the client's `messages.create` and
    `messages.stream` in place.

    Detects sync vs async by inspecting `client.messages.create`. Returns the
    same client object with instrumented methods.
    """
    is_async = _is_async_client(client)

    # --- messages.create ---
    original_create = client.messages.create
    if is_async:
        client.messages.create = _make_async_create(  # type: ignore[method-assign]
            original_create, sentinel
        )
    else:
        client.messages.create = _make_sync_create(  # type: ignore[method-assign]
            original_create, sentinel
        )

    # --- messages.stream (only if present on this client surface) ---
    original_stream = getattr(client.messages, "stream", None)
    if original_stream is not None:
        if is_async:
            client.messages.stream = _make_async_stream(  # type: ignore[method-assign]
                original_stream, sentinel
            )
        else:
            client.messages.stream = _make_sync_stream(  # type: ignore[method-assign]
                original_stream, sentinel
            )

    return client


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _is_async_client(client: Any) -> bool:
    """Return True if the given Anthropic client is an async client.

    Checks both the class name (cheapest, works for the official SDK) and the
    coroutine-ness of `messages.create` (covers subclasses / forks).
    """
    try:
        if type(client).__name__ == "AsyncAnthropic":
            return True
    except Exception:
        pass
    try:
        if inspect.iscoroutinefunction(client.messages.create):
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# messages.create — sync and async
# ---------------------------------------------------------------------------


def _make_sync_create(original_create: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_create)
    def instrumented_create(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_create(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary:
        # - Record-building errors are swallowed (instrumentation must never
        #   break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected, which
        #   is the entire point of mode='block' and must propagate.
        try:
            record = _build_record_from_message(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method="messages.create",
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

    return instrumented_create


def _make_async_create(original_create: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_create)
    async def instrumented_create(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await original_create(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_record_from_message(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method="messages.create",
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

    return instrumented_create


# ---------------------------------------------------------------------------
# messages.stream — sync and async context-manager wrappers
# ---------------------------------------------------------------------------
#
# Anthropic's `messages.stream(...)` returns a *context manager* whose
# `__enter__` yields a `MessageStream`. The `MessageStream` itself is the
# iterator and has helper methods (`text_stream`, `get_final_message()`, etc.).
#
# We wrap the outer context manager so:
#   - `__enter__` returns a proxy that delegates attribute access to the
#     underlying `MessageStream` and re-implements `__iter__` to siphon every
#     event into our usage accumulator. Special-method lookup goes through
#     the type, so we MUST define `__iter__`/`__aiter__` on the wrapper class
#     — patching dunders on an instance does not work.
#   - `__exit__` builds the CallRecord using whatever usage we accumulated
#     plus the final message snapshot if available.
#
# This mirrors LangSmith's `MessageStreamWrapper`/`AsyncMessageStreamWrapper`.


class _UsageAccumulator:
    """Aggregate `input_tokens`/`output_tokens` from streaming events.

    Anthropic streams emit:
      - `message_start` with initial `message.usage` (input_tokens populated)
      - `message_delta` with `usage` carrying the running total of output_tokens
      - `message_stop` with the final accumulated message
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.stop_reason: Any = None
        self.tool_calls: list[dict[str, Any]] = []
        self.has_text_output: bool = False
        self._final_message: Any = None

    def observe(self, event: Any) -> None:
        try:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    inp = getattr(usage, "input_tokens", None)
                    if isinstance(inp, int):
                        self.input_tokens = max(self.input_tokens, inp)
                    out = getattr(usage, "output_tokens", None)
                    if isinstance(out, int):
                        self.output_tokens = max(self.output_tokens, out)
            elif etype == "message_delta":
                usage = getattr(event, "usage", None)
                if usage is not None:
                    # Anthropic emits the running total of output_tokens on
                    # message_delta. Take max() so we never regress on any
                    # SDK quirk.
                    out = getattr(usage, "output_tokens", None)
                    if isinstance(out, int):
                        self.output_tokens = max(self.output_tokens, out)
                    inp = getattr(usage, "input_tokens", None)
                    if isinstance(inp, int):
                        self.input_tokens = max(self.input_tokens, inp)
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None)
                if stop_reason is not None:
                    self.stop_reason = stop_reason
            elif etype == "message_stop":
                msg = getattr(event, "message", None)
                if msg is not None:
                    self._final_message = msg
        except Exception:
            # Never let observation crash the user's iteration.
            pass

    def finalize_from_message(self, final_message: Any) -> None:
        """Pull authoritative usage / content from the final Message snapshot.

        Called from __exit__ via `stream.get_final_message()` when available.
        """
        try:
            if final_message is None:
                final_message = self._final_message
            if final_message is None:
                return
            usage = getattr(final_message, "usage", None)
            if usage is not None:
                inp = getattr(usage, "input_tokens", None)
                if isinstance(inp, int):
                    self.input_tokens = max(self.input_tokens, inp)
                out = getattr(usage, "output_tokens", None)
                if isinstance(out, int):
                    self.output_tokens = max(self.output_tokens, out)
            if self.stop_reason is None:
                self.stop_reason = getattr(final_message, "stop_reason", None)
            for block in getattr(final_message, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    self.tool_calls.append(
                        {
                            "name": getattr(block, "name", ""),
                            "arguments": getattr(block, "input", {}),
                        }
                    )
                elif btype == "text":
                    self.has_text_output = True
        except Exception:
            pass


class _StreamProxy:
    """Sync proxy around a `MessageStream`.

    Forwards attribute access to the wrapped stream so helpers like
    `text_stream`, `get_final_message()`, `until_done()`, etc. all keep
    working. Re-implements `__iter__` so we can siphon events into the
    accumulator. Defined on the *class* — Python's special-method lookup
    bypasses instance attributes for dunders, so this is the only correct
    place for it.
    """

    __slots__ = ("_stream", "_accumulator")

    def __init__(self, stream: Any, accumulator: _UsageAccumulator) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_accumulator", accumulator)

    def __iter__(self) -> Any:
        accumulator = self._accumulator
        for event in self._stream:
            accumulator.observe(event)
            yield event

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup misses, so this
        # transparently exposes everything the SDK's MessageStream offers
        # without us having to enumerate it.
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    """Async counterpart of `_StreamProxy`. Re-implements `__aiter__` to
    observe events; forwards everything else."""

    __slots__ = ("_stream", "_accumulator")

    def __init__(self, stream: Any, accumulator: _UsageAccumulator) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_accumulator", accumulator)

    def __aiter__(self) -> Any:
        # `__aiter__` is the *sync* hook in the async iteration protocol —
        # it must return an async iterator. We return an async generator,
        # which is the simplest async iterator Python can construct.
        return self._aiter_impl()

    async def _aiter_impl(self) -> Any:
        accumulator = self._accumulator
        async for event in self._stream:
            accumulator.observe(event)
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _MessageStreamWrapper:
    """Wrap the sync context manager returned by `client.messages.stream(...)`.

    Mirrors LangSmith's `MessageStreamWrapper`. Delegates `__enter__`/`__exit__`
    to the wrapped CM; on `__exit__` builds and records the CallRecord.
    """

    def __init__(
        self,
        wrapped_cm: Any,
        sentinel: Sentinel,
        kwargs: dict[str, Any],
        session_id: str,
    ) -> None:
        self._cm = wrapped_cm
        self._sentinel = sentinel
        self._kwargs = kwargs
        self._session_id = session_id
        self._start = time.perf_counter()
        self._accumulator = _UsageAccumulator()
        self._stream: Any = None

    def __enter__(self) -> Any:
        stream = self._cm.__enter__()
        self._stream = stream
        try:
            return _StreamProxy(stream, self._accumulator)
        except Exception:
            return stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            result = self._cm.__exit__(exc_type, exc, tb)
        finally:
            self._finalize()
        return result

    def _finalize(self) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            final_message = None
            try:
                getter = getattr(self._stream, "get_final_message", None)
                if callable(getter):
                    final_message = getter()
            except Exception:
                final_message = None
            self._accumulator.finalize_from_message(final_message)
            record = _build_record_from_accumulator(
                session_id=self._session_id,
                kwargs=self._kwargs,
                accumulator=self._accumulator,
                latency_ms=elapsed_ms,
                method="messages.stream",
            )
        except Exception:
            return
        # record_call may raise LeakDetected in block mode — propagate; swallow
        # other exceptions (rule/handler bugs must not crash user code).
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass


class _AsyncMessageStreamWrapper:
    """Async counterpart of `_MessageStreamWrapper`. Wraps the async CM
    returned by `AsyncAnthropic().messages.stream(...)`.
    """

    def __init__(
        self,
        wrapped_cm: Any,
        sentinel: Sentinel,
        kwargs: dict[str, Any],
        session_id: str,
    ) -> None:
        self._cm = wrapped_cm
        self._sentinel = sentinel
        self._kwargs = kwargs
        self._session_id = session_id
        self._start = time.perf_counter()
        self._accumulator = _UsageAccumulator()
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        stream = await self._cm.__aenter__()
        self._stream = stream
        try:
            return _AsyncStreamProxy(stream, self._accumulator)
        except Exception:
            return stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            result = await self._cm.__aexit__(exc_type, exc, tb)
        finally:
            await self._finalize()
        return result

    async def _finalize(self) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            final_message = None
            try:
                getter = getattr(self._stream, "get_final_message", None)
                if callable(getter):
                    res = getter()
                    # `get_final_message` is async on AsyncMessageStream.
                    if inspect.isawaitable(res):
                        final_message = await res
                    else:
                        final_message = res
            except Exception:
                final_message = None
            self._accumulator.finalize_from_message(final_message)
            record = _build_record_from_accumulator(
                session_id=self._session_id,
                kwargs=self._kwargs,
                accumulator=self._accumulator,
                latency_ms=elapsed_ms,
                method="messages.stream",
            )
        except Exception:
            return
        # `record_call` is sync; safe to call from async since it does
        # in-process work only. May raise LeakDetected in block mode —
        # that's the entire point of block mode and must propagate. Other
        # exceptions are swallowed.
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass


def _make_sync_stream(original_stream: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_stream)
    def instrumented_stream(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        cm = original_stream(*args, **kwargs)
        try:
            return _MessageStreamWrapper(cm, sentinel, dict(kwargs), session_id)
        except Exception:
            # If wrapping fails for any reason, hand back the original CM
            # so user code never breaks because of us.
            return cm

    return instrumented_stream


def _make_async_stream(original_stream: Any, sentinel: Sentinel) -> Any:
    # `AsyncAnthropic.messages.stream` is itself a sync function returning an
    # async context manager (it does not need to be awaited). Mirror that.
    @functools.wraps(original_stream)
    def instrumented_stream(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        cm = original_stream(*args, **kwargs)
        try:
            return _AsyncMessageStreamWrapper(cm, sentinel, dict(kwargs), session_id)
        except Exception:
            return cm

    return instrumented_stream


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _request_hash(kwargs: dict[str, Any]) -> str:
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)
    return hashlib.sha256(
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


def _build_record_from_message(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
    method: str,
) -> CallRecord:
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    tool_calls: list[dict[str, Any]] = []
    has_text_output = False
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "name": getattr(block, "name", ""),
                    "arguments": getattr(block, "input", {}),
                }
            )
        elif block_type == "text":
            has_text_output = True

    user_facing_output = has_text_output and not tool_calls

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model=model,
        method=method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        raw_request={"messages": messages, "tools": tools, "max_tokens": max_tokens},
        raw_response_meta={"stop_reason": getattr(response, "stop_reason", None)},
    )


def _build_record_from_accumulator(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    accumulator: _UsageAccumulator,
    latency_ms: float,
    method: str,
) -> CallRecord:
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)

    user_facing_output = accumulator.has_text_output and not accumulator.tool_calls

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model=model,
        method=method,
        prompt_tokens=accumulator.input_tokens,
        completion_tokens=accumulator.output_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=list(accumulator.tool_calls),
        user_facing_output=user_facing_output,
        raw_request={"messages": messages, "tools": tools, "max_tokens": max_tokens},
        raw_response_meta={"stop_reason": accumulator.stop_reason, "streamed": True},
    )


# Kept for backward compatibility with anything that imported the old
# private helper directly. Internal-only; not part of the public API.
def _build_record(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
) -> CallRecord:
    return _build_record_from_message(
        session_id=session_id,
        kwargs=kwargs,
        response=response,
        latency_ms=latency_ms,
        method="messages.create",
    )
