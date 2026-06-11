"""Tests for ``token_sentinel.wrappers.gemini.wrap_gemini``.

NO real API calls. We construct mocks shaped like a ``google.genai.Client``
and verify the wrapper:
  - swaps in instrumented ``models.generate_content`` and
    ``models.generate_content_stream`` (plus their ``aio.*`` async twins)
  - delegates to the original methods unchanged
  - builds a ``CallRecord`` matching the response shape
  - finalizes the record on stream completion
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - propagates ``LeakDetected`` from block mode through the wrapper
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import LeakDetected, Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.wrappers.gemini import (
    _build_record_from_accumulator,
    _build_record_from_response,
    _extract_tool_calls_and_text,
    _request_hash,
    _UsageAccumulator,
    wrap_gemini,
)

# ---------------------------------------------------------------------------
# Mock google-genai client + response factories
# ---------------------------------------------------------------------------


def _make_text_part(text: str) -> SimpleNamespace:
    """A part with text only — function_call is explicitly None so getattr
    sees a real attribute (rather than walking through a missing one)."""
    return SimpleNamespace(text=text, function_call=None)


def _make_function_call_part(name: str, args: dict[str, Any]) -> SimpleNamespace:
    fn_call = SimpleNamespace(name=name, args=args)
    return SimpleNamespace(text=None, function_call=fn_call)


def _make_response(
    *,
    prompt_token_count: int = 100,
    candidates_token_count: int = 25,
    text_parts: list[str] | None = None,
    function_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "STOP",
) -> SimpleNamespace:
    """Build a google-genai-shaped non-streaming response."""
    parts: list[Any] = []
    if text_parts:
        parts.extend(_make_text_part(t) for t in text_parts)
    if function_calls:
        parts.extend(
            _make_function_call_part(fc["name"], fc.get("args", {})) for fc in function_calls
        )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_token_count,
            candidates_token_count=candidates_token_count,
        ),
    )


def _make_chunk(
    *,
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    text: str | None = None,
    function_call: dict[str, Any] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Build a single streaming chunk in google-genai shape."""
    parts: list[Any] = []
    if text is not None:
        parts.append(_make_text_part(text))
    if function_call is not None:
        parts.append(_make_function_call_part(function_call["name"], function_call.get("args", {})))
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class _RecordingCallable:
    """Real callable that records calls and returns a configurable value.

    Mirrors ``tests/conftest.py::_RecordingCreate``. We need a real function
    (not a MagicMock) because ``functools.wraps`` inside the wrapper copies
    ``__name__``/``__qualname__`` and chokes on auto-generated mock children.
    """

    __name__ = "generate_content"
    __qualname__ = "Models.generate_content"
    __module__ = "google.genai.models"
    __annotations__: dict = {}
    __doc__ = "mock generate_content"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.return_value: Any = None
        self.side_effect: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value


def _make_sync_client(
    generate_return: Any = None,
    stream_return: Any = None,
) -> SimpleNamespace:
    """Build a fake ``google.genai.Client`` with the sync ``models`` surface."""
    fake_class = type("Client", (), {"__module__": "google.genai.client"})
    client = fake_class()
    gc = _RecordingCallable()
    gc.__name__ = "generate_content"
    gc.return_value = generate_return

    gcs = _RecordingCallable()
    gcs.__name__ = "generate_content_stream"
    gcs.return_value = stream_return

    client.models = SimpleNamespace(generate_content=gc, generate_content_stream=gcs)
    # No aio surface — keep it simple for sync tests.
    return client


def _make_async_client(
    async_generate: Any = None,
    async_stream: Any = None,
) -> SimpleNamespace:
    """Build a fake client with the async ``aio.models`` surface populated."""
    fake_class = type("Client", (), {"__module__": "google.genai.client"})
    client = fake_class()
    # Sync surface kept minimal — many tests don't need it.
    client.models = SimpleNamespace(
        generate_content=lambda **kw: None,
        generate_content_stream=lambda **kw: iter([]),
    )
    client.aio = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=async_generate,
            generate_content_stream=async_stream,
        )
    )
    return client


# ---------------------------------------------------------------------------
# wrap_gemini: instrumentation
# ---------------------------------------------------------------------------


def test_wrap_replaces_generate_content_methods():
    client = _make_sync_client()
    s = Sentinel(project="proj")
    original_gc = client.models.generate_content
    original_gcs = client.models.generate_content_stream
    wrap_gemini(client, s)
    assert client.models.generate_content is not original_gc
    assert client.models.generate_content_stream is not original_gcs


def test_wrap_returns_same_client_instance():
    client = _make_sync_client()
    s = Sentinel(project="proj")
    out = wrap_gemini(client, s)
    assert out is client


def test_wrap_no_aio_surface_does_not_raise():
    """Many test clients omit ``client.aio`` — that must be fine."""
    client = _make_sync_client()
    assert not hasattr(client, "aio")
    s = Sentinel(project="proj")
    wrap_gemini(client, s)  # must not raise


def test_wrap_no_models_surface_does_not_raise():
    """Defensive: a client that exposes nothing should still be returned."""
    fake_class = type("Client", (), {"__module__": "google.genai.client"})
    client = fake_class()
    s = Sentinel(project="proj")
    out = wrap_gemini(client, s)
    assert out is client


# ---------------------------------------------------------------------------
# Sync non-streaming
# ---------------------------------------------------------------------------


def test_sync_generate_content_records_call():
    client = _make_sync_client(
        generate_return=_make_response(
            prompt_token_count=120,
            candidates_token_count=30,
            text_parts=["hi back"],
        )
    )
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="hi",
        _sentinel_session_id="s-1",
    )
    assert response is not None

    records = s.tracer.session("s-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "gemini"
    assert rec.method == "models.generate_content"
    assert rec.model == "gemini-2.5-pro"
    assert rec.prompt_tokens == 120
    assert rec.completion_tokens == 30
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_request["model"] == "gemini-2.5-pro"
    assert rec.raw_request["contents"] == "hi"


def test_sync_generate_content_session_id_stripped():
    """The wrapped function must strip ``_sentinel_session_id`` before the
    SDK sees it (the real google-genai SDK would error on the unknown kwarg)."""
    client = _make_sync_client(generate_return=_make_response(text_parts=["x"]))
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    client.models.generate_content(
        model="gemini-2.5-pro",
        contents="hi",
        _sentinel_session_id="abc",
    )
    received = client.models.generate_content.__wrapped__.calls[-1]  # type: ignore[union-attr]
    assert "_sentinel_session_id" not in received
    assert received["model"] == "gemini-2.5-pro"


def test_sync_generate_content_with_function_calls_not_user_facing():
    """A response with function_call parts is intermediate, not user-facing."""
    client = _make_sync_client(
        generate_return=_make_response(
            text_parts=["thinking out loud"],
            function_calls=[{"name": "search", "args": {"q": "kittens"}}],
        )
    )
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    client.models.generate_content(
        model="gemini-2.5-pro",
        contents="search please",
        _sentinel_session_id="s-1",
    )
    rec = s.tracer.session("s-1")[0]
    assert rec.user_facing_output is False
    assert rec.tool_calls == [{"name": "search", "arguments": {"q": "kittens"}}]


def test_sync_generate_content_underlying_exception_propagates():
    client = _make_sync_client()
    s = Sentinel(project="proj")
    client.models.generate_content.side_effect = RuntimeError("API down")
    wrap_gemini(client, s)
    with pytest.raises(RuntimeError, match="API down"):
        client.models.generate_content(model="gemini-2.5-pro", contents="hi")


def test_sync_record_call_exception_does_not_crash_user_call(monkeypatch):
    """If sentinel.record_call raises, the user's call must still return."""
    client = _make_sync_client(generate_return=_make_response(text_parts=["ok"]))
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    def boom(*a, **k):
        raise RuntimeError("tracer is broken")

    monkeypatch.setattr(s, "record_call", boom)

    out = client.models.generate_content(model="gemini-2.5-pro", contents="hi")
    # The wrapper swallowed the tracer error and returned the SDK response.
    assert out is not None


def test_sync_block_mode_propagates_leak_detected():
    """``record_call`` raising ``LeakDetected`` must propagate to the user.

    retry_storm with min_retries=2 fires on the second identical call —
    that's the cleanest deterministic trigger.
    """
    client = _make_sync_client(generate_return=_make_response(text_parts=["x"]))
    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_gemini(client, s)

    client.models.generate_content(
        model="gemini-2.5-pro",
        contents="hi",
        _sentinel_session_id="s-1",
    )
    with pytest.raises(LeakDetected) as exc:
        client.models.generate_content(
            model="gemini-2.5-pro",
            contents="hi",
            _sentinel_session_id="s-1",
        )
    assert exc.value.event.type == "retry_storm"


# ---------------------------------------------------------------------------
# Async non-streaming
# ---------------------------------------------------------------------------


def test_async_generate_content_records_call():
    captured: list[dict[str, Any]] = []

    async def agc(**kwargs):
        captured.append(kwargs)
        return _make_response(
            prompt_token_count=50, candidates_token_count=20, text_parts=["async ok"]
        )

    async def astream(**kwargs):
        return iter([])

    client = _make_async_client(async_generate=agc, async_stream=astream)
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    response = asyncio.run(
        client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents="hi",
            _sentinel_session_id="async-1",
        )
    )
    assert response is not None
    records = s.tracer.session("async-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "gemini"
    assert rec.method == "models.generate_content"
    assert rec.prompt_tokens == 50
    assert rec.completion_tokens == 20
    assert rec.user_facing_output is True
    # Session id kwarg must be stripped before the original is called.
    assert "_sentinel_session_id" not in captured[-1]


def test_async_generate_content_underlying_exception_propagates():
    async def agc(**kwargs):
        raise RuntimeError("network down")

    async def astream(**kwargs):
        return iter([])

    client = _make_async_client(async_generate=agc, async_stream=astream)
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    with pytest.raises(RuntimeError, match="network down"):
        asyncio.run(client.aio.models.generate_content(model="gemini-2.5-pro", contents="hi"))


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


def test_sync_stream_records_call_on_completion():
    """Sync streaming — the record is built when the iterator exhausts."""
    chunks = [
        _make_chunk(prompt_token_count=50, text="hello"),
        _make_chunk(
            prompt_token_count=50,
            candidates_token_count=10,
            text=" there",
        ),
        _make_chunk(
            prompt_token_count=50,
            candidates_token_count=20,
            finish_reason="STOP",
        ),
    ]
    client = _make_sync_client(stream_return=iter(chunks))
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    stream = client.models.generate_content_stream(
        model="gemini-2.5-pro",
        contents="hi",
        _sentinel_session_id="stream-1",
    )
    observed = list(stream)
    assert len(observed) == 3

    records = s.tracer.session("stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "models.generate_content_stream"
    assert rec.prompt_tokens == 50
    assert rec.completion_tokens == 20  # max() over chunks
    assert rec.user_facing_output is True
    assert rec.raw_response_meta.get("streamed") is True
    assert rec.raw_response_meta.get("finish_reason") == "STOP"


def test_sync_stream_records_function_calls():
    """Function-call chunks in a stream must populate ``tool_calls`` and
    suppress ``user_facing_output``."""
    chunks = [
        _make_chunk(prompt_token_count=10, text="I will search"),
        _make_chunk(
            prompt_token_count=10,
            candidates_token_count=5,
            function_call={"name": "search", "args": {"q": "stuff"}},
            finish_reason="STOP",
        ),
    ]
    client = _make_sync_client(stream_return=iter(chunks))
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    list(
        client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents="search",
            _sentinel_session_id="s-1",
        )
    )

    rec = s.tracer.session("s-1")[0]
    assert rec.tool_calls == [{"name": "search", "arguments": {"q": "stuff"}}]
    assert rec.user_facing_output is False


def test_sync_stream_proxy_forwards_attribute_access():
    """Helper attributes on the underlying iterator stay reachable through
    the proxy via ``__getattr__``."""

    class _CustomIter:
        def __init__(self):
            self._chunks = iter([_make_chunk(text="x", finish_reason="STOP")])
            self.helper_attr = "exposed"

        def __iter__(self):
            return self._chunks

    client = _make_sync_client(stream_return=_CustomIter())
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    proxy = client.models.generate_content_stream(model="gemini-2.5-pro", contents="hi")
    assert proxy.helper_attr == "exposed"  # forwarded through __getattr__


def test_sync_stream_record_build_failure_does_not_crash_iteration():
    """A broken response shape mid-stream must not break user iteration."""
    # garbage chunks (no .candidates) — observe() must swallow.
    chunks = [SimpleNamespace(), SimpleNamespace()]
    client = _make_sync_client(stream_return=iter(chunks))
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    out = list(
        client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents="hi",
            _sentinel_session_id="s-1",
        )
    )
    assert len(out) == 2
    # A record is still built (with zero usage / no content).
    records = s.tracer.session("s-1")
    assert len(records) == 1
    assert records[0].prompt_tokens == 0


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


def _make_async_iter(chunks: list[Any]) -> Any:
    """Wrap a list of chunks as a real async iterator."""

    async def aiter_impl():
        for c in chunks:
            yield c

    return aiter_impl()


def test_async_stream_records_call_on_completion():
    chunks = [
        _make_chunk(prompt_token_count=42, text="hello "),
        _make_chunk(
            prompt_token_count=42,
            candidates_token_count=8,
            text="world",
            finish_reason="STOP",
        ),
    ]

    async def astream(**kwargs):
        return _make_async_iter(chunks)

    async def agc(**kwargs):
        return _make_response(text_parts=["x"])

    client = _make_async_client(async_generate=agc, async_stream=astream)
    s = Sentinel(project="proj")
    wrap_gemini(client, s)

    async def run():
        proxy = await client.aio.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents="hi",
            _sentinel_session_id="async-stream-1",
        )
        out = []
        async for chunk in proxy:
            out.append(chunk)
        return out

    out = asyncio.run(run())
    assert len(out) == 2

    records = s.tracer.session("async-stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "models.generate_content_stream"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 8
    assert rec.user_facing_output is True
    assert rec.raw_response_meta.get("streamed") is True


def test_async_stream_block_mode_propagates_leak():
    """Block mode + retry_storm via async streaming — second identical call
    fires LeakDetected, which must propagate out of the async iterator's
    finalizer."""
    chunks = [_make_chunk(text="x", finish_reason="STOP")]

    async def astream(**kwargs):
        return _make_async_iter(list(chunks))

    async def agc(**kwargs):
        return _make_response(text_parts=["x"])

    client = _make_async_client(async_generate=agc, async_stream=astream)
    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_gemini(client, s)

    async def consume():
        proxy = await client.aio.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents="hi",
            _sentinel_session_id="s-1",
        )
        async for _ in proxy:
            pass

    # First stream: records but doesn't fire (need 2 to trigger).
    asyncio.run(consume())
    # Second identical stream: fires retry_storm in block mode.
    with pytest.raises(LeakDetected) as exc:
        asyncio.run(consume())
    assert exc.value.event.type == "retry_storm"


# ---------------------------------------------------------------------------
# _UsageAccumulator unit tests
# ---------------------------------------------------------------------------


def test_usage_accumulator_observes_chunks_max_token_counts():
    acc = _UsageAccumulator()
    acc.observe(_make_chunk(prompt_token_count=10, candidates_token_count=5))
    acc.observe(_make_chunk(prompt_token_count=10, candidates_token_count=12))
    # Buggy SDK regression — must not regress under max().
    acc.observe(_make_chunk(prompt_token_count=10, candidates_token_count=3))
    assert acc.prompt_tokens == 10
    assert acc.completion_tokens == 12


def test_usage_accumulator_collects_text():
    acc = _UsageAccumulator()
    acc.observe(_make_chunk(text="hello "))
    acc.observe(_make_chunk(text="world"))
    assert acc.has_text_output is True


def test_usage_accumulator_collects_function_calls():
    acc = _UsageAccumulator()
    acc.observe(_make_chunk(function_call={"name": "search", "args": {"q": "x"}}))
    assert acc.tool_calls == [{"name": "search", "arguments": {"q": "x"}}]


def test_usage_accumulator_dedupes_repeated_function_calls():
    """Defensive: if the SDK ever emits the same function_call across multiple
    chunks, we must not double-count it."""
    acc = _UsageAccumulator()
    acc.observe(_make_chunk(function_call={"name": "s", "args": {"q": "x"}}))
    acc.observe(_make_chunk(function_call={"name": "s", "args": {"q": "x"}}))
    assert acc.tool_calls == [{"name": "s", "arguments": {"q": "x"}}]


def test_usage_accumulator_tracks_finish_reason():
    acc = _UsageAccumulator()
    acc.observe(_make_chunk(text="hi"))
    acc.observe(_make_chunk(finish_reason="STOP"))
    assert acc.finish_reason == "STOP"


def test_usage_accumulator_robust_to_garbage_chunks():
    acc = _UsageAccumulator()
    acc.observe(SimpleNamespace())  # nothing
    acc.observe(SimpleNamespace(candidates=[]))  # empty candidates
    acc.observe(SimpleNamespace(candidates=[SimpleNamespace()]))  # no .content
    # No crashes; nothing accumulated.
    assert acc.prompt_tokens == 0
    assert acc.completion_tokens == 0
    assert acc.has_text_output is False
    assert acc.tool_calls == []


# ---------------------------------------------------------------------------
# _build_record_from_response / _build_record_from_accumulator
# ---------------------------------------------------------------------------


def test_build_record_from_response_basic():
    response = _make_response(
        prompt_token_count=42,
        candidates_token_count=7,
        text_parts=["hi"],
    )
    rec = _build_record_from_response(
        session_id="s1",
        kwargs={
            "model": "gemini-2.5-pro",
            "contents": "hi",
            "tools": [],
        },
        response=response,
        latency_ms=12.5,
        method="models.generate_content",
    )
    assert isinstance(rec, CallRecord)
    assert rec.provider == "gemini"
    assert rec.model == "gemini-2.5-pro"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 7
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_request["contents"] == "hi"
    assert rec.raw_response_meta["finish_reason"] == "STOP"


def test_build_record_from_response_function_calls():
    response = _make_response(
        text_parts=None,
        function_calls=[
            {"name": "search", "args": {"q": "kittens"}},
            {"name": "search", "args": {"q": "puppies"}},
        ],
    )
    rec = _build_record_from_response(
        session_id="s1",
        kwargs={"model": "gemini-2.5-pro", "contents": "search"},
        response=response,
        latency_ms=10.0,
        method="models.generate_content",
    )
    assert len(rec.tool_calls) == 2
    assert rec.tool_calls[0] == {"name": "search", "arguments": {"q": "kittens"}}
    assert rec.user_facing_output is False


def test_build_record_from_response_no_candidates():
    response = SimpleNamespace(
        candidates=[],
        usage_metadata=SimpleNamespace(prompt_token_count=0, candidates_token_count=0),
    )
    rec = _build_record_from_response(
        session_id="s1",
        kwargs={"model": "gemini-2.5-pro", "contents": ""},
        response=response,
        latency_ms=1.0,
        method="models.generate_content",
    )
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    assert rec.raw_response_meta["finish_reason"] is None


def test_build_record_from_response_missing_usage():
    response = SimpleNamespace(candidates=[], usage_metadata=None)
    rec = _build_record_from_response(
        session_id="s1",
        kwargs={"model": "gemini-2.5-pro", "contents": ""},
        response=response,
        latency_ms=1.0,
        method="models.generate_content",
    )
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


def test_build_record_from_response_default_model():
    response = SimpleNamespace(candidates=[], usage_metadata=None)
    rec = _build_record_from_response(
        session_id="s1",
        kwargs={"contents": ""},
        response=response,
        latency_ms=1.0,
        method="models.generate_content",
    )
    assert rec.model == "unknown"


def test_build_record_from_accumulator_basic():
    acc = _UsageAccumulator()
    acc.prompt_tokens = 100
    acc.completion_tokens = 25
    acc.finish_reason = "STOP"
    acc.has_text_output = True
    rec = _build_record_from_accumulator(
        session_id="s1",
        kwargs={
            "model": "gemini-2.5-pro",
            "contents": "hi",
            "tools": [],
        },
        accumulator=acc,
        latency_ms=200.0,
        method="models.generate_content_stream",
    )
    assert rec.method == "models.generate_content_stream"
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 25
    assert rec.user_facing_output is True
    assert rec.raw_response_meta["streamed"] is True
    assert rec.raw_response_meta["finish_reason"] == "STOP"


def test_build_record_from_accumulator_text_plus_tool_not_user_facing():
    acc = _UsageAccumulator()
    acc.has_text_output = True
    acc.tool_calls = [{"name": "search", "arguments": {"q": "x"}}]
    rec = _build_record_from_accumulator(
        session_id="s1",
        kwargs={"model": "gemini-2.5-pro", "contents": "hi"},
        accumulator=acc,
        latency_ms=10.0,
        method="models.generate_content_stream",
    )
    assert rec.user_facing_output is False


# ---------------------------------------------------------------------------
# _request_hash + _extract_tool_calls_and_text
# ---------------------------------------------------------------------------


def test_request_hash_stable_over_kwarg_order():
    a = _request_hash({"model": "m", "contents": "hi", "tools": [], "generation_config": None})
    b = _request_hash({"generation_config": None, "tools": [], "contents": "hi", "model": "m"})
    assert a == b


def test_request_hash_changes_with_model():
    a = _request_hash({"model": "gemini-2.5-pro", "contents": "hi"})
    b = _request_hash({"model": "gemini-2.5-flash", "contents": "hi"})
    assert a != b


def test_request_hash_changes_with_contents():
    a = _request_hash({"model": "m", "contents": "hi"})
    b = _request_hash({"model": "m", "contents": "bye"})
    assert a != b


def test_request_hash_handles_missing_keys():
    h = _request_hash({})
    assert isinstance(h, str)
    assert len(h) == 64


def test_extract_tool_calls_and_text_handles_dictlike_args():
    """The SDK may surface ``args`` as a Struct or proto-like object instead
    of a dict. The extractor must coerce sensibly."""

    class _ProtoLike:
        def __init__(self, data):
            self._data = data

        def keys(self):
            return self._data.keys()

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, key):
            return self._data[key]

    fn_call = SimpleNamespace(name="search", args=_ProtoLike({"q": "x"}))
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=None, function_call=fn_call)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=None,
    )
    tool_calls, has_text = _extract_tool_calls_and_text(response)
    assert tool_calls == [{"name": "search", "arguments": {"q": "x"}}]
    assert has_text is False


def test_extract_tool_calls_and_text_args_none():
    fn_call = SimpleNamespace(name="noop", args=None)
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=None, function_call=fn_call)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=None,
    )
    tool_calls, has_text = _extract_tool_calls_and_text(response)
    assert tool_calls == [{"name": "noop", "arguments": {}}]
    assert has_text is False


def test_extract_tool_calls_and_text_empty_text_not_counted():
    """Empty-string text parts must not flip the user_facing flag."""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="", function_call=None)]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=None,
    )
    _, has_text = _extract_tool_calls_and_text(response)
    assert has_text is False
