"""Wrap a Google Gemini ``google-genai`` client to capture call records.

Pattern: in-place mutation of ``client.models.generate_content`` and
``client.models.generate_content_stream`` (plus their async counterparts on
``client.aio.models``). The original is captured in a closure so the
instrumented version can delegate. This preserves all type hints in IDEs
because we mutate the live instance, not a subclass.

Reference pattern: this is a structural twin of ``wrap_anthropic`` — same
two-level safety boundary (record-build errors are swallowed; ``record_call``
errors are swallowed except ``LeakDetected``, which must propagate so block
mode actually halts user code).

The ``google-genai`` SDK (NOT the legacy ``google-generativeai``) exposes:

  client = genai.Client(api_key=...)                              # direct
  client = genai.Client(vertexai=True, project=..., location=...) # Vertex backend
  client.models.generate_content(model='gemini-2.5-pro', contents='hi')
  client.models.generate_content_stream(...)                      # sync iter
  client.aio.models.generate_content(...)                         # async coro
  client.aio.models.generate_content_stream(...)                  # async iter

Four cases handled:
  1. Sync non-streaming: ``client.models.generate_content`` -> Response
  2. Sync streaming: ``client.models.generate_content_stream`` -> Iterator[chunk]
  3. Async non-streaming: ``client.aio.models.generate_content`` -> awaitable Response
  4. Async streaming: ``client.aio.models.generate_content_stream`` -> AsyncIterator[chunk]

Note on ``generate_content_stream``: the ``google-genai`` async variant is
itself an *async function* that returns an async iterator (you must
``await`` the call before iterating). We detect coroutine-ness on the
original to pick the right wrapper.
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
    from token_sentinel.sentinel import Sentinel


def wrap_gemini(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a ``google.genai.Client``. Mutates ``client.models.generate_content``,
    ``client.models.generate_content_stream``, and the corresponding
    ``client.aio.models.*`` methods in place.

    Returns the same client object with instrumented methods.
    """
    # Sync surface: client.models.{generate_content, generate_content_stream}
    models = getattr(client, "models", None)
    if models is not None:
        _patch_sync_generate_content(models, sentinel)
        _patch_sync_generate_content_stream(models, sentinel)

    # Async surface: client.aio.models.{generate_content, generate_content_stream}
    aio = getattr(client, "aio", None)
    if aio is not None:
        aio_models = getattr(aio, "models", None)
        if aio_models is not None:
            _patch_async_generate_content(aio_models, sentinel)
            _patch_async_generate_content_stream(aio_models, sentinel)

    return client


# ---------------------------------------------------------------------------
# generate_content — sync and async (non-streaming)
# ---------------------------------------------------------------------------


def _patch_sync_generate_content(models: Any, sentinel: Sentinel) -> None:
    original = getattr(models, "generate_content", None)
    if original is None:
        return

    @functools.wraps(original)
    def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary:
        # - Record-building errors are swallowed (instrumentation must never
        #   break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected, which
        #   is the entire point of mode='block' and must propagate.
        try:
            record = _build_record_from_response(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method="models.generate_content",
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

    models.generate_content = instrumented


def _patch_async_generate_content(aio_models: Any, sentinel: Sentinel) -> None:
    original = getattr(aio_models, "generate_content", None)
    if original is None:
        return

    @functools.wraps(original)
    async def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await original(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_record_from_response(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method="models.generate_content",
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

    aio_models.generate_content = instrumented


# ---------------------------------------------------------------------------
# generate_content_stream — sync and async iterator wrappers
# ---------------------------------------------------------------------------
#
# The streaming variants in google-genai do NOT return a context manager (as
# anthropic does); they return a plain iterator (sync) or async iterator
# (async). We wrap the iterator itself so we can siphon chunks into a
# usage accumulator and finalize the CallRecord when iteration ends.
#
# Special-method lookup (``__iter__`` / ``__aiter__``) is performed on the
# *type*, never on the instance, so the wrappers must be defined on a class.


class _UsageAccumulator:
    """Aggregate token usage and content from streaming chunks.

    google-genai streaming chunks each carry a ``usage_metadata`` (cumulative,
    per the docs) and a ``candidates[0].content.parts`` list with text /
    function_call entries. We take ``max()`` for token counts so we never
    regress on any SDK quirk where a later chunk reports lower numbers.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.has_text_output: bool = False
        self.finish_reason: Any = None
        # per-modality breakdown when Gemini surfaces it on the
        # cumulative ``usage_metadata`` of any chunk. Latest non-empty
        # wins (the SDK emits the same final breakdown on later chunks).
        self.prompt_tokens_details: list[dict[str, Any]] | None = None

    def observe(self, chunk: Any) -> None:
        try:
            self._observe_usage(chunk)
            self._observe_content(chunk)
        except Exception:
            # Never let observation crash the user's iteration.
            pass

    def _observe_usage(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage_metadata", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_token_count", None)
        if isinstance(prompt, int):
            self.prompt_tokens = max(self.prompt_tokens, prompt)
        completion = getattr(usage, "candidates_token_count", None)
        if isinstance(completion, int):
            self.completion_tokens = max(self.completion_tokens, completion)
        # Refresh the modality breakdown from the latest chunk that
        # surfaces one. ``_extract_prompt_tokens_details`` returns None
        # when the field is absent, so this is a no-op for chunks that
        # don't include it (typical: only the final chunk carries it).
        details = _extract_prompt_tokens_details(usage)
        if details:
            self.prompt_tokens_details = details

    def _observe_content(self, chunk: Any) -> None:
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            return
        candidate = candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        if finish is not None:
            self.finish_reason = finish
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                name = getattr(fn_call, "name", "") or ""
                args = getattr(fn_call, "args", None)
                # google-genai exposes args as a dict-like; coerce to dict so
                # downstream rules (tool_loop) can hash/compare consistently.
                if args is None:
                    args_value: Any = {}
                elif isinstance(args, dict):
                    args_value = args
                else:
                    try:
                        args_value = dict(args)
                    except Exception:
                        args_value = args
                # Avoid accumulating duplicates if the SDK emits the same
                # function_call across multiple chunks (it shouldn't, but be
                # defensive — same name+args wins once).
                entry = {"name": name, "arguments": args_value}
                if entry not in self.tool_calls:
                    self.tool_calls.append(entry)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    self.has_text_output = True


class _StreamProxy:
    """Sync proxy around a ``generate_content_stream`` iterator.

    Forwards attribute access to the wrapped iterator and re-implements
    ``__iter__`` so we can siphon chunks into the accumulator. Defined on
    the class because Python's special-method lookup bypasses instance
    attributes for dunders.
    """

    __slots__ = ("_iterator", "_accumulator", "_finalize", "_finalized")

    def __init__(self, iterator: Any, accumulator: _UsageAccumulator, finalize: Any) -> None:
        object.__setattr__(self, "_iterator", iterator)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_finalize", finalize)
        object.__setattr__(self, "_finalized", False)

    def __iter__(self) -> Any:
        accumulator = self._accumulator
        try:
            for chunk in self._iterator:
                accumulator.observe(chunk)
                yield chunk
        finally:
            self._safe_finalize()

    def _safe_finalize(self) -> None:
        # Only finalize once even if the user iterates the proxy more than
        # once (which would be unusual, but cheap to guard against).
        if object.__getattribute__(self, "_finalized"):
            return
        object.__setattr__(self, "_finalized", True)
        # Detect generator-close path: when this finalize runs from a `finally`
        # whose try-block was exited via close()/GC, GeneratorExit is in flight.
        # Python silently swallows non-GeneratorExit exceptions raised in that
        # state — meaning LeakDetected from block mode would vanish without
        # halting the user's app. Suppress with a warning instead.
        is_close = sys.exc_info()[0] is GeneratorExit
        finalize = object.__getattribute__(self, "_finalize")
        try:
            finalize(self._accumulator)
        except LeakDetected:
            if is_close:
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
        # exposes anything the SDK iterator offers without enumerating it.
        return getattr(self._iterator, name)


class _AsyncStreamProxy:
    """Async counterpart of ``_StreamProxy``."""

    __slots__ = ("_iterator", "_accumulator", "_finalize", "_finalized")

    def __init__(self, iterator: Any, accumulator: _UsageAccumulator, finalize: Any) -> None:
        object.__setattr__(self, "_iterator", iterator)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_finalize", finalize)
        object.__setattr__(self, "_finalized", False)

    def __aiter__(self) -> Any:
        # ``__aiter__`` is sync and must return an async iterator. An async
        # generator IS an async iterator and is the simplest construction.
        return self._aiter_impl()

    async def _aiter_impl(self) -> Any:
        accumulator = self._accumulator
        try:
            async for chunk in self._iterator:
                accumulator.observe(chunk)
                yield chunk
        finally:
            self._safe_finalize()

    def _safe_finalize(self) -> None:
        if object.__getattribute__(self, "_finalized"):
            return
        object.__setattr__(self, "_finalized", True)
        finalize = object.__getattribute__(self, "_finalize")
        try:
            finalize(self._accumulator)
        except LeakDetected:
            raise
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)


def _patch_sync_generate_content_stream(models: Any, sentinel: Sentinel) -> None:
    original = getattr(models, "generate_content_stream", None)
    if original is None:
        return

    @functools.wraps(original)
    def instrumented(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        captured_kwargs = dict(kwargs)
        try:
            iterator = original(*args, **kwargs)
        except Exception:
            raise

        accumulator = _UsageAccumulator()

        def finalize(acc: _UsageAccumulator) -> None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            try:
                record = _build_record_from_accumulator(
                    session_id=session_id,
                    kwargs=captured_kwargs,
                    accumulator=acc,
                    latency_ms=elapsed_ms,
                    method="models.generate_content_stream",
                )
            except Exception:
                return
            # record_call may raise LeakDetected in block mode — propagate;
            # swallow other exceptions (rule/handler bugs must not crash).
            try:
                sentinel.record_call(record)
            except LeakDetected:
                raise
            except Exception:
                pass

        try:
            return _StreamProxy(iterator, accumulator, finalize)
        except Exception:
            # If wrapping fails for any reason, hand back the original
            # iterator so user code never breaks because of us.
            return iterator

    models.generate_content_stream = instrumented


def _patch_async_generate_content_stream(aio_models: Any, sentinel: Sentinel) -> None:
    original = getattr(aio_models, "generate_content_stream", None)
    if original is None:
        return

    # ``aio.models.generate_content_stream`` may be either:
    #   (a) a coroutine function: returns an async iterator after `await`
    #   (b) a plain sync function returning an async iterator directly
    # Real google-genai is (a); we still handle (b) to be defensive against
    # future SDK shape drift.
    is_coroutine = inspect.iscoroutinefunction(original)

    if is_coroutine:

        @functools.wraps(original)
        async def instrumented(*args: Any, **kwargs: Any) -> Any:
            session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
            start = time.perf_counter()
            captured_kwargs = dict(kwargs)
            iterator = await original(*args, **kwargs)

            accumulator = _UsageAccumulator()

            def finalize(acc: _UsageAccumulator) -> None:
                elapsed_ms = (time.perf_counter() - start) * 1000
                try:
                    record = _build_record_from_accumulator(
                        session_id=session_id,
                        kwargs=captured_kwargs,
                        accumulator=acc,
                        latency_ms=elapsed_ms,
                        method="models.generate_content_stream",
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
                return _AsyncStreamProxy(iterator, accumulator, finalize)
            except Exception:
                return iterator

        aio_models.generate_content_stream = instrumented
        return

    @functools.wraps(original)
    def instrumented_sync(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        captured_kwargs = dict(kwargs)
        iterator = original(*args, **kwargs)

        accumulator = _UsageAccumulator()

        def finalize(acc: _UsageAccumulator) -> None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            try:
                record = _build_record_from_accumulator(
                    session_id=session_id,
                    kwargs=captured_kwargs,
                    accumulator=acc,
                    latency_ms=elapsed_ms,
                    method="models.generate_content_stream",
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
            return _AsyncStreamProxy(iterator, accumulator, finalize)
        except Exception:
            return iterator

    aio_models.generate_content_stream = instrumented_sync


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _request_hash(kwargs: dict[str, Any]) -> str:
    """SHA-256 of the request shape — same shape as the anthropic wrapper.

    We hash ``(model, contents, tools, generation_config)``. ``tools`` may
    arrive top-level or under ``config.tools`` depending on how the user
    called the SDK; we hash the value we see (don't try to merge — the rule
    only cares about a stable per-shape hash).
    """
    model = kwargs.get("model", "unknown")
    contents = kwargs.get("contents", "")
    tools = kwargs.get("tools", [])
    generation_config = kwargs.get("generation_config", kwargs.get("config"))
    return hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "contents": contents,
                "tools": tools,
                "generation_config": generation_config,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _extract_prompt_tokens_details(usage: Any) -> list[dict[str, Any]] | None:
    """Pull ``usage_metadata.prompt_tokens_details`` into a JSON-friendly list.

    Gemini exposes the per-modality token breakdown as a list of
    ``ModalityTokenCount`` entries. Each entry carries ``.modality`` (a
    string like ``"TEXT"`` / ``"IMAGE"`` / ``"VIDEO"`` / ``"AUDIO"``) and
    ``.token_count`` (int). We coerce both into native Python types so
    the resulting dict survives ``json.dumps`` cleanly in downstream
    transport (cloud ingest, on-leak handlers, etc.). Returns ``None``
    when the field is absent so callers can distinguish "no breakdown
    available" from "breakdown says zero image tokens".
    """
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    out: list[dict[str, Any]] = []
    try:
        for entry in details:
            modality = getattr(entry, "modality", None)
            count = getattr(entry, "token_count", None)
            if modality is None and isinstance(entry, dict):
                modality = entry.get("modality")
                count = entry.get("token_count")
            if not isinstance(modality, str):
                # Modality may arrive as an enum; coerce to str.
                modality = str(modality) if modality is not None else None
            # Strip ``MediaModality.`` enum prefix if present so the
            # rule's case-insensitive ``"image"`` check still hits.
            if isinstance(modality, str) and "." in modality:
                modality = modality.rsplit(".", 1)[-1]
            if not isinstance(count, int):
                continue
            out.append({"modality": modality, "token_count": count})
    except Exception:
        # Never let the breakdown crash record-building.
        return None
    return out or None


def _extract_tool_calls_and_text(response: Any) -> tuple[list[dict[str, Any]], bool]:
    """Pull tool_calls and a has_text flag out of a Gemini response.

    Tool calls live at ``response.candidates[0].content.parts[*].function_call``;
    text lives at the same parts under ``.text``.
    """
    tool_calls: list[dict[str, Any]] = []
    has_text_output = False

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return tool_calls, has_text_output

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        fn_call = getattr(part, "function_call", None)
        if fn_call is not None:
            name = getattr(fn_call, "name", "") or ""
            args = getattr(fn_call, "args", None)
            if args is None:
                args_value: Any = {}
            elif isinstance(args, dict):
                args_value = args
            else:
                try:
                    args_value = dict(args)
                except Exception:
                    args_value = args
            tool_calls.append({"name": name, "arguments": args_value})
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                has_text_output = True

    return tool_calls, has_text_output


def _build_record_from_response(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
    method: str,
) -> CallRecord:
    model = kwargs.get("model", "unknown")
    contents = kwargs.get("contents", "")
    tools = kwargs.get("tools", [])

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    # Defensive: SDK occasionally returns None for these; coerce to int.
    if not isinstance(prompt_tokens, int):
        prompt_tokens = 0
    if not isinstance(completion_tokens, int):
        completion_tokens = 0
    # surface the per-modality breakdown when Gemini ships it
    # (``prompt_tokens_details`` is the only provider-side text/image
    # split available; the ``vision_cost_concentration`` rule needs it).
    prompt_tokens_details = _extract_prompt_tokens_details(usage)

    tool_calls, has_text_output = _extract_tool_calls_and_text(response)
    user_facing_output = has_text_output and not tool_calls

    finish_reason = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)

    meta: dict[str, Any] = {"finish_reason": finish_reason}
    if prompt_tokens_details is not None:
        meta["prompt_tokens_details"] = prompt_tokens_details

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="gemini",
        model=model,
        method=method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        raw_request={"model": model, "contents": contents, "tools": tools},
        raw_response_meta=meta,
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
    contents = kwargs.get("contents", "")
    tools = kwargs.get("tools", [])

    user_facing_output = accumulator.has_text_output and not accumulator.tool_calls

    meta: dict[str, Any] = {
        "finish_reason": accumulator.finish_reason,
        "streamed": True,
    }
    if accumulator.prompt_tokens_details is not None:
        meta["prompt_tokens_details"] = accumulator.prompt_tokens_details

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="gemini",
        model=model,
        method=method,
        prompt_tokens=accumulator.prompt_tokens,
        completion_tokens=accumulator.completion_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=list(accumulator.tool_calls),
        user_facing_output=user_facing_output,
        raw_request={"model": model, "contents": contents, "tools": tools},
        raw_response_meta=meta,
    )
