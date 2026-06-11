"""Wrap an AWS Bedrock (boto3) ``bedrock-runtime`` client to capture call records.

Pattern: in-place mutation of ``client.converse`` and ``client.converse_stream``.
The original is captured in a closure so the instrumented version can delegate.
boto3 attaches service operations dynamically at client-creation time, so we
attribute-check before wrapping.

Reference pattern: see ``wrappers/anthropic.py`` and ``wrappers/openai.py``.

Three cases handled by ``wrap_bedrock``:
  1. Non-streaming converse: ``client.converse(...)`` -> dict with ``output``,
     ``usage``, ``stopReason``.
  2. Streaming converse: ``client.converse_stream(...)`` -> dict whose
     ``stream`` key is an EventStream that yields events
     (``messageStart`` / ``contentBlockStart`` / ``contentBlockDelta`` /
     ``contentBlockStop`` / ``messageStop`` / ``metadata`` carrying usage).
  3. ``invoke_model`` — not yet instrumented. Its body is a JSON string with
     a per-provider shape (Anthropic, Cohere, AI21, Mistral, Meta Llama,
     Amazon Titan all differ); a per-vendor parser registry is required
     before it's safe to wire up. ``converse`` / ``converse_stream`` cover
     all current Bedrock-supported model families and should be preferred.

For streams, the ``CallRecord`` is built when the EventStream finalizes — we
wrap it so iteration is observable, accumulate usage from the ``metadata``
event, and flush in ``__exit__`` / ``close()`` / ``__del__`` (whichever fires
first). Failures in instrumentation never propagate to user code — they are
swallowed with the same two-level safety boundary used by the Anthropic and
OpenAI wrappers (record-building errors swallowed; ``record_call`` exceptions
swallowed EXCEPT ``LeakDetected``, which must propagate so block mode works).

Async note: Bedrock's official ``boto3`` is sync-only. There is no
``client.aio`` surface like the Anthropic/OpenAI SDKs expose. ``aioboto3`` is
a third-party shim that wraps boto3 — instrumenting it would require a
separate code path. Not currently instrumented.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from token_sentinel.sentinel import Sentinel


def wrap_bedrock(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a boto3 ``bedrock-runtime`` client. Mutates ``converse`` and
    ``converse_stream`` in place.

    boto3 attaches service operations dynamically at client-creation time, so
    we attribute-check each method before instrumenting. If a method isn't
    present (older botocore, restricted service surface, etc.) we silently
    skip it rather than crash.

    Returns the same client object with instrumented methods.
    """
    original_converse = getattr(client, "converse", None)
    if original_converse is not None and callable(original_converse):
        client.converse = _make_converse(original_converse, sentinel)

    original_converse_stream = getattr(client, "converse_stream", None)
    if original_converse_stream is not None and callable(original_converse_stream):
        client.converse_stream = _make_converse_stream(original_converse_stream, sentinel)

    # ``client.invoke_model`` and ``client.invoke_model_with_response_stream``
    # are not yet instrumented. Their request body is a JSON-encoded string
    # with a per-provider shape (Anthropic, Cohere, AI21, Mistral, Meta Llama,
    # Amazon Titan all differ), and the response body is likewise per-provider.
    # A per-vendor parser registry is needed before this is safe to wire up; for
    # now we skip it so we don't emit malformed CallRecords. ``converse`` /
    # ``converse_stream`` cover all current Bedrock-supported model families.

    return client


# ---------------------------------------------------------------------------
# converse — non-streaming
# ---------------------------------------------------------------------------


def _make_converse(original_converse: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_converse)
    def instrumented_converse(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_converse(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary mirrors anthropic.py / openai.py:
        # - Record-building errors are swallowed (instrumentation must never
        #   break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected, which
        #   is the entire point of mode='block' and must propagate.
        try:
            record = _build_record_from_converse(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
                latency_ms=elapsed_ms,
                method="converse",
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

    return instrumented_converse


# ---------------------------------------------------------------------------
# converse_stream — streaming
# ---------------------------------------------------------------------------
#
# Bedrock's converse_stream returns a dict like:
#     {'ResponseMetadata': {...}, 'stream': <EventStream object>}
# The EventStream is iterable and yields dicts shaped:
#     {'messageStart': {'role': 'assistant'}}
#     {'contentBlockStart': {'start': {'toolUse': {'toolUseId': ..., 'name': ...}}, 'contentBlockIndex': 0}}
#     {'contentBlockDelta': {'delta': {'text': '...'} | {'toolUse': {'input': '...'}}, 'contentBlockIndex': 0}}
#     {'contentBlockStop': {'contentBlockIndex': 0}}
#     {'messageStop': {'stopReason': '...'}}
#     {'metadata': {'usage': {'inputTokens': ..., 'outputTokens': ..., 'totalTokens': ...},
#                   'metrics': {'latencyMs': ...}}}
#
# We replace ``response['stream']`` with a proxy that observes each event on
# the way through, then flushes a CallRecord on stream end.


class _StreamUsageAccumulator:
    """Aggregate token usage and tool-use blocks from converse_stream events.

    The authoritative usage numbers arrive in the ``metadata`` event near the
    end of the stream. ``contentBlockStart``/``contentBlockDelta`` give us the
    in-flight tool-use names + their JSON-stringified arguments which we
    accumulate so we can emit the same ``tool_calls`` shape as the
    non-streaming path.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.stop_reason: Any = None
        self.has_text_output: bool = False
        # Bedrock's contentBlockStart for a toolUse arrives separately from the
        # JSON-string deltas (which arrive incrementally on contentBlockDelta).
        # Key by contentBlockIndex so we can stitch them back together when
        # the block stops.
        self._tool_blocks: dict[int, dict[str, Any]] = {}
        self.tool_calls: list[dict[str, Any]] = []
        self.metadata_seen: bool = False

    def observe(self, event: Any) -> None:
        try:
            if not isinstance(event, dict):
                return
            if "messageStart" in event:
                # Nothing to record — just role, no usage info.
                return
            if "contentBlockStart" in event:
                block = event["contentBlockStart"]
                idx = block.get("contentBlockIndex", 0)
                start = block.get("start", {}) or {}
                tool_use = start.get("toolUse")
                if tool_use:
                    self._tool_blocks[idx] = {
                        "name": tool_use.get("name", ""),
                        "input_str": "",
                    }
                return
            if "contentBlockDelta" in event:
                block = event["contentBlockDelta"]
                idx = block.get("contentBlockIndex", 0)
                delta = block.get("delta", {}) or {}
                if "text" in delta:
                    if delta.get("text"):
                        self.has_text_output = True
                elif "toolUse" in delta:
                    tu = delta["toolUse"] or {}
                    chunk = tu.get("input", "")
                    if idx not in self._tool_blocks:
                        # Defensive: some streams emit deltas before start —
                        # seed an empty block so we still capture the JSON.
                        self._tool_blocks[idx] = {"name": "", "input_str": ""}
                    self._tool_blocks[idx]["input_str"] += chunk
                return
            if "contentBlockStop" in event:
                block = event["contentBlockStop"]
                idx = block.get("contentBlockIndex", 0)
                if idx in self._tool_blocks:
                    pending = self._tool_blocks.pop(idx)
                    raw = pending.get("input_str", "")
                    parsed: Any
                    try:
                        parsed = json.loads(raw) if raw else {}
                    except (TypeError, ValueError):
                        parsed = raw
                    self.tool_calls.append({"name": pending.get("name", ""), "arguments": parsed})
                return
            if "messageStop" in event:
                stop = event["messageStop"] or {}
                if stop.get("stopReason") is not None:
                    self.stop_reason = stop.get("stopReason")
                return
            if "metadata" in event:
                meta = event["metadata"] or {}
                usage = meta.get("usage", {}) or {}
                inp = usage.get("inputTokens")
                if isinstance(inp, int):
                    self.input_tokens = max(self.input_tokens, inp)
                out = usage.get("outputTokens")
                if isinstance(out, int):
                    self.output_tokens = max(self.output_tokens, out)
                self.metadata_seen = True
                return
        except Exception:
            # Never let observation crash the user's iteration.
            pass


class _EventStreamProxy:
    """Proxy around the boto3 EventStream returned in
    ``converse_stream(...)['stream']``.

    Forwards attribute access to the wrapped stream so callers using helpers
    like ``close()`` keep working. Re-implements ``__iter__`` so we can siphon
    events into the accumulator. The dunder is defined on the *class* —
    Python's special-method lookup bypasses instance attributes for dunders,
    so this is the only correct place for it (matches the Anthropic
    ``_StreamProxy`` pattern).

    On stream end (iterator exhausted, ``close()`` called, or ``__del__``),
    we call ``_finalize`` exactly once, which builds and records the
    CallRecord. The "exactly once" is enforced by ``_flushed``.
    """

    __slots__ = (
        "_stream",
        "_accumulator",
        "_sentinel",
        "_kwargs",
        "_session_id",
        "_start",
        "_flushed",
    )

    def __init__(
        self,
        stream: Any,
        accumulator: _StreamUsageAccumulator,
        sentinel: Sentinel,
        kwargs: dict[str, Any],
        session_id: str,
        start: float,
    ) -> None:
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_accumulator", accumulator)
        object.__setattr__(self, "_sentinel", sentinel)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_flushed", False)

    def __iter__(self) -> Any:
        accumulator = self._accumulator
        try:
            for event in self._stream:
                accumulator.observe(event)
                yield event
        finally:
            self._finalize()

    def close(self) -> Any:
        # boto3 EventStream may expose ``close()``; forward and finalize.
        try:
            closer = getattr(self._stream, "close", None)
            if callable(closer):
                return closer()
        finally:
            self._finalize()

    def __enter__(self) -> Any:
        # Some boto3 EventStreams support context-manager use. Forward where
        # possible; otherwise return self so ``with stream as s:`` still works.
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
            self._finalize()
        return False

    def __del__(self) -> None:
        # Last-resort flush. Users who break out of iteration early without
        # calling close() still get their CallRecord at GC time. __del__ must
        # never raise — Python silently swallows __del__ exceptions anyway,
        # which means LeakDetected from block mode would vanish without the
        # user's app halting. Suppress LeakDetected explicitly here and emit
        # a warning so the user knows block mode was best-effort on this path.
        try:
            self._finalize(suppress_block=True)
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup misses, so this
        # transparently exposes everything the underlying EventStream offers
        # (``close``, internal cursors, etc.) without us enumerating.
        return getattr(self._stream, name)

    def _finalize(self, *, suppress_block: bool = False) -> None:
        if self._flushed:
            return
        # Mark flushed BEFORE building the record so a record-building
        # exception that triggers __del__ recursion can't double-fire.
        object.__setattr__(self, "_flushed", True)

        elapsed_ms = (time.perf_counter() - self._start) * 1000
        try:
            record = _build_record_from_accumulator(
                session_id=self._session_id,
                kwargs=self._kwargs,
                accumulator=self._accumulator,
                latency_ms=elapsed_ms,
                method="converse_stream",
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            if suppress_block:
                # Called from __del__ where Python would swallow the exception
                # anyway. Warn so the user knows their leak fired but block
                # mode could not halt the (already-abandoned) caller.
                warnings.warn(
                    "TokenSentinel: LeakDetected suppressed in stream GC path "
                    "(block mode is best-effort on abandoned streams; use "
                    "'with stream:' or fully iterate to get block-mode halts).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        except Exception:
            pass


def _make_converse_stream(original_stream: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_stream)
    def instrumented_converse_stream(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_stream(*args, **kwargs)
        except Exception:
            # Underlying boto3 call failed — don't try to record, just re-raise.
            raise

        # The response is a dict with a 'stream' EventStream. If for some
        # reason it isn't (mocked, unusual surface), hand back the original
        # response untouched.
        try:
            if not isinstance(response, dict) or "stream" not in response:
                return response
            accumulator = _StreamUsageAccumulator()
            proxy = _EventStreamProxy(
                stream=response["stream"],
                accumulator=accumulator,
                sentinel=sentinel,
                kwargs=dict(kwargs),
                session_id=session_id,
                start=start,
            )
            response["stream"] = proxy
        except Exception:
            # If wrapping fails for any reason, hand back the original
            # response so user code never breaks because of us.
            return response
        return response

    return instrumented_converse_stream


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _request_hash(kwargs: dict[str, Any]) -> str:
    """Hash a Bedrock converse request for retry_storm detection.

    Includes ``modelId``, ``messages``, ``toolConfig``, and ``inferenceConfig``
    — these are the four fields that uniquely identify a logical call. System
    prompts and ``additionalModelRequestFields`` are intentionally excluded
    so that retries with identical conversation but different metadata still
    coalesce. (If we find this is wrong in practice, we'll widen the hash.)
    """
    model_id = kwargs.get("modelId", "unknown")
    messages = kwargs.get("messages", [])
    tool_config = kwargs.get("toolConfig", {})
    inference_config = kwargs.get("inferenceConfig", {})
    return hashlib.sha256(
        json.dumps(
            {
                "modelId": model_id,
                "messages": messages,
                "toolConfig": tool_config,
                "inferenceConfig": inference_config,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _build_record_from_converse(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
    method: str,
) -> CallRecord:
    """Build a CallRecord from a Bedrock ``converse`` (non-stream) response.

    Bedrock's converse response shape:
        {
          'output': {'message': {'role': 'assistant',
                                  'content': [{'text': '...'} |
                                              {'toolUse': {'toolUseId': ...,
                                                            'name': ...,
                                                            'input': {...}}}]}},
          'stopReason': 'end_turn' | 'tool_use' | ...,
          'usage': {'inputTokens': N, 'outputTokens': M, 'totalTokens': N+M},
          'metrics': {'latencyMs': ...},
        }
    """
    model_id = kwargs.get("modelId", "unknown")
    messages = kwargs.get("messages", [])
    tool_config = kwargs.get("toolConfig", {})

    # Defensive get-paths — boto3 always returns plain dicts but tests may
    # hand us SimpleNamespace stand-ins.
    if isinstance(response, dict):
        usage = response.get("usage", {}) or {}
        prompt_tokens = usage.get("inputTokens", 0) or 0
        completion_tokens = usage.get("outputTokens", 0) or 0
        stop_reason = response.get("stopReason")
        output = response.get("output", {}) or {}
        message = output.get("message", {}) or {}
        content = message.get("content", []) or []
    else:
        usage = getattr(response, "usage", None)
        prompt_tokens = (usage.get("inputTokens", 0) if isinstance(usage, dict) else 0) or 0
        completion_tokens = (usage.get("outputTokens", 0) if isinstance(usage, dict) else 0) or 0
        stop_reason = getattr(response, "stopReason", None)
        content = []

    tool_calls: list[dict[str, Any]] = []
    has_text_output = False

    for block in content:
        if not isinstance(block, dict):
            continue
        if "toolUse" in block:
            tool_use = block["toolUse"] or {}
            tool_calls.append(
                {
                    "name": tool_use.get("name", ""),
                    "arguments": tool_use.get("input", {}),
                }
            )
        elif "text" in block:
            if block.get("text"):
                has_text_output = True

    user_facing_output = has_text_output and not tool_calls

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="bedrock",
        model=model_id,
        method=method,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        raw_request={
            "modelId": model_id,
            "messages": messages,
            "toolConfig": tool_config,
        },
        raw_response_meta={"stopReason": stop_reason},
    )


def _build_record_from_accumulator(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    accumulator: _StreamUsageAccumulator,
    latency_ms: float,
    method: str,
) -> CallRecord:
    """Build a CallRecord from a streamed converse_stream accumulator."""
    model_id = kwargs.get("modelId", "unknown")
    messages = kwargs.get("messages", [])
    tool_config = kwargs.get("toolConfig", {})

    user_facing_output = accumulator.has_text_output and not accumulator.tool_calls

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="bedrock",
        model=model_id,
        method=method,
        prompt_tokens=accumulator.input_tokens,
        completion_tokens=accumulator.output_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash(kwargs),
        tool_calls=list(accumulator.tool_calls),
        user_facing_output=user_facing_output,
        raw_request={
            "modelId": model_id,
            "messages": messages,
            "toolConfig": tool_config,
        },
        raw_response_meta={"stopReason": accumulator.stop_reason, "streamed": True},
    )
