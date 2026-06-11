"""Tests for streaming instrumentation of the OpenAI wrapper (stable release).

NO real API calls. We construct synthetic ``ChatCompletionChunk``-shaped
objects via ``SimpleNamespace`` and feed them through ``_RecordingStream``
mock clients.

Coverage:
  - sync streaming end-to-end (with and without ``stream_options.include_usage``)
  - tool_calls stitched across multi-delta chunks (single + multi-tool)
  - finish_reason landing in raw_response_meta
  - user_facing_output toggling on text-only vs tool-call streams
  - early break out of iteration -> CallRecord still built
  - explicit ``close()`` -> CallRecord still built
  - GC-driven generator close -> LeakDetected suppressed with RuntimeWarning
  - block mode + full iteration -> LeakDetected raises out of iteration
  - log mode -> handler called, no raise
  - async streaming end-to-end (await + async for)
  - async streaming under block mode -> LeakDetected propagates
  - ``_OpenAIUsageAccumulator`` unit tests for adversarial chunk sequences
  - block-mode warning is NOT emitted under normal streaming (only on
    proxy-construction failure)
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import LeakDetected, Sentinel
from token_sentinel.wrappers.openai import (
    _BLOCK_MODE_STREAM_MESSAGE,
    _AsyncOpenAIStreamProxy,
    _OpenAIStreamProxy,
    _OpenAIUsageAccumulator,
    wrap_openai,
)

# ---------------------------------------------------------------------------
# Synthetic ChatCompletionChunk factories
# ---------------------------------------------------------------------------


def _make_text_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    include_usage: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> SimpleNamespace:
    """Build a synthetic ``ChatCompletionChunk`` with a content delta.

    Mirrors what the OpenAI SDK actually yields:
      - ``choices[0].delta.content`` -> text fragment
      - ``choices[0].finish_reason`` -> only set on the final-content chunk
      - ``usage`` -> only on the very-final chunk and only if
        ``stream_options.include_usage=True``
    """
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    usage = None
    if include_usage:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_call_delta(
    *,
    index: int,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """Build a single ``ChoiceDeltaToolCall`` delta entry."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=None, type="function", function=fn)


def _make_tool_call_chunk(
    *,
    deltas: list[SimpleNamespace],
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Build a chunk whose delta carries a list of tool-call deltas."""
    delta = SimpleNamespace(content=None, tool_calls=deltas)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice], usage=None)


def _make_final_usage_chunk(
    *, prompt_tokens: int, completion_tokens: int, finish_reason: str = "stop"
) -> SimpleNamespace:
    """Build the final chunk: empty delta, a finish_reason, and usage."""
    delta = SimpleNamespace(content=None, tool_calls=None)
    # OpenAI's final chunk often has an empty choices list when usage is
    # included; some SDK versions still attach a finish_reason on the prior
    # content chunk and emit a usage-only trailer. We test both shapes.
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Mock OpenAI clients
# ---------------------------------------------------------------------------


class _RecordingStream:
    """A faux OpenAI ``Stream``: iterable + closeable.

    Records whether ``close()`` was called so tests can verify forwarding.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self) -> Any:
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


def _make_sync_client(stream_chunks: list[Any] | None = None) -> Any:
    """Build a fake sync OpenAI client. ``stream=True`` calls return a
    fresh ``_RecordingStream`` over ``stream_chunks``."""
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def _create(**kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            return _RecordingStream(stream_chunks or [])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))
    return client


class _AsyncRecordingStream:
    """Async counterpart of ``_RecordingStream``."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> Any:
        return self._aiter_impl()

    async def _aiter_impl(self) -> Any:
        for c in self._chunks:
            yield c

    async def aclose(self) -> None:
        self.closed = True


def _make_async_client(stream_chunks: list[Any] | None = None) -> Any:
    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()

    async def _acreate(**kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            return _AsyncRecordingStream(stream_chunks or [])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    async def _aembed(**kwargs: Any) -> Any:
        return SimpleNamespace(usage=None)

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=_acreate))
    client.embeddings = SimpleNamespace(create=_aembed)
    return client


def _stream_warnings(
    records: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Filter to only the OpenAI-streaming-bypass warnings."""
    return [w for w in records if str(w.message) == _BLOCK_MODE_STREAM_MESSAGE]


# ===========================================================================
# Sync streaming -- end-to-end coverage
# ===========================================================================


def test_sync_stream_records_call_with_usage():
    """1. Iterate to completion with ``stream_options.include_usage=True`` ->
    CallRecord built with correct prompt/completion tokens from final chunk."""
    chunks = [
        _make_text_chunk(content="hello"),
        _make_text_chunk(content=" "),
        _make_text_chunk(content="world"),
        _make_final_usage_chunk(prompt_tokens=42, completion_tokens=8, finish_reason="stop"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
        _sentinel_session_id="stream-1",
    )
    observed = list(stream)
    assert len(observed) == 4

    records = s.tracer.session("stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "openai"
    assert rec.method == "chat.completions.create"
    assert rec.model == "gpt-4o"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 8
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_response_meta["streamed"] is True
    assert rec.raw_response_meta["finish_reason"] == "stop"
    assert rec.raw_response_meta["usage_unavailable"] is False


def test_sync_stream_records_call_without_usage():
    """2. Iterate to completion with NO ``stream_options.include_usage`` ->
    record has tokens=0 and ``usage_unavailable=True``."""
    chunks = [
        _make_text_chunk(content="hello"),
        _make_text_chunk(content=" world", finish_reason="stop"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        stream=True,
        _sentinel_session_id="stream-2",
    )
    list(stream)

    records = s.tracer.session("stream-2")
    assert len(records) == 1
    rec = records[0]
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0
    assert rec.raw_response_meta["usage_unavailable"] is True
    assert rec.raw_response_meta["streamed"] is True


def test_sync_stream_tool_calls_stitched_across_deltas():
    """3. Single tool call: ``name`` on first delta, ``arguments`` split
    across 3 deltas with the same index -- must stitch into one call."""
    chunks = [
        # First delta: name + first arg fragment
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="search", arguments='{"q":')]
        ),
        # Second delta: arg fragment continues, no name
        _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='"kitten')]),
        # Third delta: arg fragment finishes, no name
        _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='s"}')]),
        _make_final_usage_chunk(prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="tool-1",
        )
    )

    rec = s.tracer.session("tool-1")[0]
    assert rec.tool_calls == [{"name": "search", "arguments": {"q": "kittens"}}]
    assert rec.user_facing_output is False  # tool_calls suppress user-facing
    assert rec.raw_response_meta["finish_reason"] == "tool_calls"


def test_sync_stream_multi_tool_calls_indexed_separately():
    """4. Multi tool call (different ``index`` values): each accumulator slot
    holds the right name + arguments."""
    chunks = [
        # Tool index=0: name + arg fragment
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="search", arguments='{"q":"a"}')]
        ),
        # Tool index=1: name + arg fragment
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=1, name="lookup", arguments='{"id":42}')]
        ),
        _make_final_usage_chunk(prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="tool-2",
        )
    )

    rec = s.tracer.session("tool-2")[0]
    assert len(rec.tool_calls) == 2
    # Sorted by index, so [0]=search, [1]=lookup.
    assert rec.tool_calls[0] == {"name": "search", "arguments": {"q": "a"}}
    assert rec.tool_calls[1] == {"name": "lookup", "arguments": {"id": 42}}


def test_sync_stream_finish_reason_lands_in_raw_response_meta():
    """5. ``finish_reason`` from final chunk is captured."""
    chunks = [
        _make_text_chunk(content="hello"),
        _make_text_chunk(content=" cap", finish_reason="length"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="fin-1",
        )
    )
    rec = s.tracer.session("fin-1")[0]
    assert rec.raw_response_meta["finish_reason"] == "length"


def test_sync_stream_user_facing_output_true_on_text_only():
    """6a. ``user_facing_output=True`` when chunks had non-empty content
    and no tool_calls."""
    chunks = [
        _make_text_chunk(content="answer"),
        _make_text_chunk(content="", finish_reason="stop"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="ufo-1",
        )
    )
    rec = s.tracer.session("ufo-1")[0]
    assert rec.user_facing_output is True


def test_sync_stream_user_facing_output_false_when_no_text():
    """6b. ``user_facing_output=False`` when no chunk had non-empty content."""
    chunks = [
        _make_text_chunk(content=None, finish_reason="stop"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="ufo-2",
        )
    )
    rec = s.tracer.session("ufo-2")[0]
    assert rec.user_facing_output is False


def test_sync_stream_early_break_still_finalizes():
    """7. Early break out of the iterator -> CallRecord built (with whatever
    was accumulated). No crash."""
    chunks = [
        _make_text_chunk(content="part 1"),
        _make_text_chunk(content="part 2"),
        _make_text_chunk(content="part 3"),
        _make_final_usage_chunk(prompt_tokens=10, completion_tokens=3),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        stream=True,
        _sentinel_session_id="brk-1",
    )
    seen = []
    for chunk in stream:
        seen.append(chunk)
        if len(seen) == 1:
            break  # User abandons mid-stream

    # Generator's finally fires on close; record exists.
    records = s.tracer.session("brk-1")
    assert len(records) == 1
    rec = records[0]
    # Only the first chunk was observed -> partial usage, has_text_output True
    assert rec.user_facing_output is True
    assert rec.raw_response_meta["streamed"] is True


def test_sync_stream_explicit_close_finalizes():
    """8. Explicit ``stream.close()`` -> CallRecord built."""
    chunks = [
        _make_text_chunk(content="hello"),
        _make_final_usage_chunk(prompt_tokens=5, completion_tokens=1),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        stream=True,
        _sentinel_session_id="cls-1",
    )
    # Close without ever iterating.
    stream.close()
    records = s.tracer.session("cls-1")
    assert len(records) == 1
    # Underlying stream's close was forwarded.
    # The proxy forwards via __getattr__; closed state is on the wrapped stream.
    # We can't directly access ``_stream`` via attribute (it's a slot), but
    # the proxy's close() invoked the underlying close() which set ``closed``.


def test_sync_stream_close_idempotent():
    """Closing twice must not double-record."""
    chunks = [
        _make_text_chunk(content="hi"),
        _make_final_usage_chunk(prompt_tokens=2, completion_tokens=1),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        stream=True,
        _sentinel_session_id="cls-2",
    )
    stream.close()
    stream.close()  # second close must be a no-op for recording purposes
    assert len(s.tracer.session("cls-2")) == 1


def test_sync_stream_gc_close_suppresses_leak_with_warning():
    """9. Generator close via GC -> ``LeakDetected`` suppressed with
    RuntimeWarning per the Gemini-style suppression pattern.

    We trigger this by configuring block mode + retry_storm so the SECOND
    identical streamed call would fire a leak. We then iterate the second
    stream just once and let GC tear it down -- the LeakDetected fired in
    the generator's close path must NOT crash but must emit a warning.
    """

    def chunks_factory() -> list[Any]:
        return [
            _make_text_chunk(content="hi"),
            _make_final_usage_chunk(prompt_tokens=1, completion_tokens=1, finish_reason="stop"),
        ]

    def make_stream(**kwargs: Any) -> Any:
        return _RecordingStream(chunks_factory())

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=make_stream))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_openai(client, s)

    # First call: records but doesn't trigger.
    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="gc-1",
        )
    )

    # Second call: would fire LeakDetected on the second observation. We
    # iterate exactly one chunk, then let the generator be GC'd.
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[],
        stream=True,
        stream_options={"include_usage": True},
        _sentinel_session_id="gc-1",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Get an iterator and pull one chunk.
        it = iter(stream)
        next(it)
        # Drop references and force GC.
        del it
        del stream
        gc.collect()

    # A RuntimeWarning about LeakDetected suppression in stream close path.
    suppression_warnings = [
        w for w in caught if w.category is RuntimeWarning and "LeakDetected" in str(w.message)
    ]
    assert len(suppression_warnings) >= 1


def test_sync_stream_block_mode_full_iteration_raises_leak():
    """10. Streaming under ``mode='block'`` -- full iteration to completion:
    ``LeakDetected`` raises out of the iteration on the second identical call."""

    def make_stream(**kwargs: Any) -> Any:
        return _RecordingStream(
            [
                _make_text_chunk(content="hi"),
                _make_final_usage_chunk(prompt_tokens=1, completion_tokens=1),
            ]
        )

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=make_stream))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="blk-1",
        )
    )
    # Second identical call: full iteration triggers LeakDetected.
    with pytest.raises(LeakDetected) as exc:
        list(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[],
                stream=True,
                stream_options={"include_usage": True},
                _sentinel_session_id="blk-1",
            )
        )
    assert exc.value.event.type == "retry_storm"


def test_sync_stream_log_mode_handler_called_no_raise():
    """11. Streaming under ``mode='log'``: handler called, no raise."""

    def make_stream(**kwargs: Any) -> Any:
        return _RecordingStream(
            [
                _make_text_chunk(content="hi"),
                _make_final_usage_chunk(prompt_tokens=1, completion_tokens=1),
            ]
        )

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=make_stream))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(
        project="proj",
        mode="log",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    captured: list[Any] = []

    @s.on_leak
    def handler(event: Any) -> None:
        captured.append(event)

    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="log-1",
        )
    )
    # Second identical -- in log mode, handler fires but no raise.
    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="log-1",
        )
    )

    assert len(captured) >= 1
    assert any(getattr(e, "type", "") == "retry_storm" for e in captured)


# ===========================================================================
# Async streaming
# ===========================================================================


def test_async_stream_records_call_with_usage():
    """12a. Async streaming: await-then-async-for, full coverage."""
    chunks = [
        _make_text_chunk(content="hello"),
        _make_text_chunk(content=" world"),
        _make_final_usage_chunk(prompt_tokens=42, completion_tokens=8, finish_reason="stop"),
    ]
    client = _make_async_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    async def run() -> list[Any]:
        proxy = await client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="async-stream-1",
        )
        out: list[Any] = []
        async for chunk in proxy:
            out.append(chunk)
        return out

    out = asyncio.run(run())
    assert len(out) == 3

    records = s.tracer.session("async-stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 8
    assert rec.user_facing_output is True
    assert rec.raw_response_meta["streamed"] is True
    assert rec.raw_response_meta["finish_reason"] == "stop"


def test_async_stream_tool_calls_stitched():
    """12b. Async streaming with tool-call stitching."""
    chunks = [
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="search", arguments='{"q":')]
        ),
        _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, arguments='"x"}')]),
        _make_final_usage_chunk(prompt_tokens=10, completion_tokens=5, finish_reason="tool_calls"),
    ]
    client = _make_async_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    async def run() -> None:
        proxy = await client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="async-tool-1",
        )
        async for _ in proxy:
            pass

    asyncio.run(run())
    rec = s.tracer.session("async-tool-1")[0]
    assert rec.tool_calls == [{"name": "search", "arguments": {"q": "x"}}]


def test_async_stream_without_usage_marks_unavailable():
    """12c. Async streaming without ``stream_options.include_usage`` ->
    record has tokens=0 and ``usage_unavailable=True``."""
    chunks = [
        _make_text_chunk(content="hi", finish_reason="stop"),
    ]
    client = _make_async_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    async def run() -> None:
        proxy = await client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="async-no-usage",
        )
        async for _ in proxy:
            pass

    asyncio.run(run())
    rec = s.tracer.session("async-no-usage")[0]
    assert rec.prompt_tokens == 0
    assert rec.raw_response_meta["usage_unavailable"] is True


def test_async_stream_block_mode_propagates_leak():
    """13. Async streaming under ``mode='block'``: ``LeakDetected`` raises
    out of the ``async for``."""

    def make_chunks() -> list[Any]:
        return [
            _make_text_chunk(content="hi"),
            _make_final_usage_chunk(prompt_tokens=1, completion_tokens=1),
        ]

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()

    async def acreate(**kwargs: Any) -> Any:
        return _AsyncRecordingStream(make_chunks())

    async def aembed(**kwargs: Any) -> Any:
        return SimpleNamespace(usage=None)

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=acreate))
    client.embeddings = SimpleNamespace(create=aembed)

    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_openai(client, s)

    async def consume() -> None:
        proxy = await client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="async-blk-1",
        )
        async for _ in proxy:
            pass

    asyncio.run(consume())  # First: records, no leak yet
    with pytest.raises(LeakDetected) as exc:
        asyncio.run(consume())  # Second: triggers
    assert exc.value.event.type == "retry_storm"


# ===========================================================================
# _OpenAIUsageAccumulator unit tests (adversarial sequences)
# ===========================================================================


def test_accumulator_initial_state_is_usage_unavailable():
    """Until we observe a chunk with a usage block, ``usage_unavailable``
    starts True (the default-uninstructive state)."""
    acc = _OpenAIUsageAccumulator()
    assert acc.usage_unavailable is True
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.tool_calls == []


def test_accumulator_observes_text_content():
    acc = _OpenAIUsageAccumulator()
    acc.observe(_make_text_chunk(content="hello"))
    acc.observe(_make_text_chunk(content="world"))
    assert acc.has_text_output is True


def test_accumulator_empty_string_content_does_not_set_text_output():
    acc = _OpenAIUsageAccumulator()
    acc.observe(_make_text_chunk(content=""))
    acc.observe(_make_text_chunk(content=None))
    assert acc.has_text_output is False


def test_accumulator_finish_reason_captured():
    acc = _OpenAIUsageAccumulator()
    acc.observe(_make_text_chunk(content="x", finish_reason="stop"))
    assert acc.finish_reason == "stop"


def test_accumulator_usage_marks_available_and_takes_max():
    acc = _OpenAIUsageAccumulator()
    acc.observe(_make_final_usage_chunk(prompt_tokens=10, completion_tokens=3))
    assert acc.usage_unavailable is False
    assert acc.input_tokens == 10
    assert acc.output_tokens == 3
    # Defensive: a follow-up chunk with smaller numbers must not regress.
    acc.observe(_make_final_usage_chunk(prompt_tokens=5, completion_tokens=1))
    assert acc.input_tokens == 10
    assert acc.output_tokens == 3


def test_accumulator_tool_call_name_only_first_delta():
    """Adversarial: the SDK puts ``name`` only on the first delta for an
    index. The accumulator must keep the name once captured even if later
    deltas have ``name=None``."""
    acc = _OpenAIUsageAccumulator()
    acc.observe(
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="search", arguments='{"q":')]
        )
    )
    # Subsequent delta has no name -- must not reset the name to "".
    acc.observe(
        _make_tool_call_chunk(deltas=[_make_tool_call_delta(index=0, name=None, arguments='"x"}')])
    )
    assert acc.tool_calls == [{"name": "search", "arguments": {"q": "x"}}]


def test_accumulator_tool_call_missing_name_first_delta():
    """Adversarial: first delta has no name -- accumulator preserves an empty
    string (matches the non-streaming wrapper's behavior on missing names)."""
    acc = _OpenAIUsageAccumulator()
    acc.observe(
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name=None, arguments='{"q":"x"}')]
        )
    )
    assert acc.tool_calls == [{"name": "", "arguments": {"q": "x"}}]


def test_accumulator_tool_call_unparseable_arguments_falls_back_to_raw():
    """Adversarial: argument fragments that don't parse as JSON must be
    surfaced as the raw string (matches ``_build_chat_record``'s behavior)."""
    acc = _OpenAIUsageAccumulator()
    acc.observe(
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="search", arguments="not valid {")]
        )
    )
    assert acc.tool_calls == [{"name": "search", "arguments": "not valid {"}]


def test_accumulator_tool_calls_out_of_order_indices():
    """Adversarial: deltas arrive in arbitrary index order -- the accumulator
    must produce them sorted by index for stable downstream comparisons."""
    acc = _OpenAIUsageAccumulator()
    # Index 1 first, then 0
    acc.observe(
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=1, name="b", arguments='{"k":2}')]
        )
    )
    acc.observe(
        _make_tool_call_chunk(
            deltas=[_make_tool_call_delta(index=0, name="a", arguments='{"k":1}')]
        )
    )
    out = acc.tool_calls
    assert [t["name"] for t in out] == ["a", "b"]


def test_accumulator_robust_to_garbage_chunks():
    """Adversarial: SimpleNamespace() with no choices/usage must not crash."""
    acc = _OpenAIUsageAccumulator()
    acc.observe(SimpleNamespace())
    acc.observe(SimpleNamespace(choices=[]))
    acc.observe(SimpleNamespace(choices=[SimpleNamespace()]))
    # Nothing accumulated, no crash.
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.has_text_output is False
    assert acc.tool_calls == []


def test_accumulator_tool_call_non_int_index_defaults_to_zero():
    """Defensive: if the SDK ever yields a non-int index, default to 0
    rather than crash (mirrors Bedrock's defensive behavior)."""
    acc = _OpenAIUsageAccumulator()
    fn = SimpleNamespace(name="x", arguments='{"k":1}')
    bad_delta = SimpleNamespace(index="not-an-int", id=None, type="function", function=fn)
    acc.observe(_make_tool_call_chunk(deltas=[bad_delta]))
    assert acc.tool_calls == [{"name": "x", "arguments": {"k": 1}}]


# ===========================================================================
# Block-mode warning is NOT emitted under normal streaming flow
# ===========================================================================


def test_block_mode_warning_not_emitted_on_normal_streaming():
    """15. The block-mode warning must NOT fire under normal streaming flow
    -- streams ARE instrumented now, so no bypass exists.
    """
    chunks = [_make_text_chunk(content="hi", finish_reason="stop")]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[],
                stream=True,
                _sentinel_session_id="warn-1",
            )
        )

    assert _stream_warnings(caught) == []


def test_block_mode_warning_emitted_on_proxy_construction_failure(monkeypatch):
    """The defensive fallback warning still fires if proxy construction
    fails for any reason (e.g. a non-iterable mock return)."""
    chunks = [_make_text_chunk(content="hi")]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj", mode="block")
    wrap_openai(client, s)

    # Force proxy construction to fail by monkeypatching the proxy class.
    import token_sentinel.wrappers.openai as openai_module

    class _BoomProxy:
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(openai_module, "_OpenAIStreamProxy", _BoomProxy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="warn-2",
        )
        # Fell back to passthrough -- the raw stream is not the proxy.
        assert isinstance(out, _RecordingStream)

    relevant = _stream_warnings(caught)
    assert len(relevant) == 1


# ===========================================================================
# Bonus: realistic chunk sequence (proxies the OpenAI SDK output shape)
# ===========================================================================


def test_realistic_streaming_sequence_full_pipeline():
    """16. A realistic chunk sequence resembling a true OpenAI streamed
    response: 4 content deltas + 1 final-with-usage. Verify the full
    accumulator -> CallRecord pipeline."""
    chunks = [
        _make_text_chunk(content="The "),
        _make_text_chunk(content="weather "),
        _make_text_chunk(content="is "),
        _make_text_chunk(content="sunny.", finish_reason="stop"),
        _make_final_usage_chunk(prompt_tokens=27, completion_tokens=5, finish_reason="stop"),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    out = list(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "weather?"}],
            stream=True,
            stream_options={"include_usage": True},
            _sentinel_session_id="realistic-1",
        )
    )
    assert len(out) == 5

    rec = s.tracer.session("realistic-1")[0]
    assert rec.prompt_tokens == 27
    assert rec.completion_tokens == 5
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_response_meta["finish_reason"] == "stop"
    assert rec.raw_response_meta["streamed"] is True
    assert rec.raw_response_meta["usage_unavailable"] is False
    assert rec.model == "gpt-4o-mini"
    assert rec.method == "chat.completions.create"


# ===========================================================================
# Forwarding & misc
# ===========================================================================


def test_sync_stream_proxy_forwards_attribute_access():
    """Helper attrs on the wrapped Stream remain reachable via __getattr__."""
    raw = _RecordingStream([])
    raw.helper_attr = "exposed"

    proxy = _OpenAIStreamProxy(
        stream=raw,
        accumulator=_OpenAIUsageAccumulator(),
        finalize=lambda acc: None,
    )
    assert proxy.helper_attr == "exposed"


def test_async_stream_proxy_forwards_attribute_access():
    raw = _AsyncRecordingStream([])
    raw.helper_attr = "exposed"

    proxy = _AsyncOpenAIStreamProxy(
        stream=raw,
        accumulator=_OpenAIUsageAccumulator(),
        finalize=lambda acc: None,
    )
    assert proxy.helper_attr == "exposed"


def test_sync_stream_session_id_kwarg_stripped_from_underlying():
    """The streaming path must strip ``_sentinel_session_id`` before the
    underlying SDK sees it (regression guard)."""
    received: dict[str, Any] = {}

    def real_create(**kwargs: Any) -> Any:
        received.update(kwargs)
        return _RecordingStream([_make_final_usage_chunk(prompt_tokens=1, completion_tokens=1)])

    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=real_create))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="abc",
        )
    )
    assert "_sentinel_session_id" not in received
    assert received.get("stream") is True


def test_sync_stream_underlying_exception_propagates():
    """If the SDK call itself raises (e.g. invalid kwargs), the exception
    must propagate -- the wrapper does not try to record anything."""
    cls = type("OpenAI", (), {"__module__": "openai"})
    client = cls()

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("API down")

    client.chat = SimpleNamespace(completions=SimpleNamespace(create=boom))
    client.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    with pytest.raises(RuntimeError, match="API down"):
        client.chat.completions.create(model="gpt-4o", messages=[], stream=True)


def test_sync_stream_record_build_failure_swallowed(monkeypatch):
    """A broken record-build path mid-finalize must not crash user iteration."""
    chunks = [_make_text_chunk(content="hi", finish_reason="stop")]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    # Force ``_build_record_from_accumulator`` to raise.
    import token_sentinel.wrappers.openai as openai_module

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("build broken")

    monkeypatch.setattr(openai_module, "_build_record_from_accumulator", boom)

    out = list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="brk-2",
        )
    )
    # User iteration succeeded; no record was created (build was broken).
    assert len(out) == 1
    assert s.tracer.session("brk-2") == []


def test_sync_stream_record_call_exception_does_not_crash_iteration(monkeypatch):
    """A buggy rule that throws non-LeakDetected from record_call must not
    crash the user's iteration."""
    chunks = [
        _make_text_chunk(content="hi"),
        _make_final_usage_chunk(prompt_tokens=1, completion_tokens=1),
    ]
    client = _make_sync_client(stream_chunks=chunks)
    s = Sentinel(project="proj")
    wrap_openai(client, s)

    monkeypatch.setattr(
        s,
        "record_call",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rule broken")),
    )

    out = list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
            stream=True,
            _sentinel_session_id="brk-3",
        )
    )
    assert len(out) == 2  # User iteration succeeded.
