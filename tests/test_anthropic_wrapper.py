"""Tests for ``token_sentinel.wrappers.anthropic.wrap_anthropic``.

NO real API calls. We construct mocks shaped like an ``anthropic.Anthropic``
client and verify the wrapper:
  - swaps in an instrumented ``messages.create``
  - delegates to the original create unchanged
  - builds a ``CallRecord`` matching the response
  - never crashes the user's call when the tracer or Sentinel misbehaves

The V0 wrapper also handles async clients and streaming context managers; we
exercise both paths here.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.wrappers.anthropic import (
    _build_record,
    _build_record_from_accumulator,
    _build_record_from_message,
    _is_async_client,
    _request_hash,
    _UsageAccumulator,
    wrap_anthropic,
)

# ---------------------------------------------------------------------------
# wrap_anthropic: instrumentation
# ---------------------------------------------------------------------------


def test_wrap_replaces_messages_create(mock_anthropic_client):
    s = Sentinel(project="proj")
    original = mock_anthropic_client.messages.create
    wrap_anthropic(mock_anthropic_client, s)
    assert mock_anthropic_client.messages.create is not original
    # The instrumented function is a regular Python function, not a MagicMock.
    assert callable(mock_anthropic_client.messages.create)


def test_wrap_returns_same_client_instance(mock_anthropic_client):
    s = Sentinel(project="proj")
    out = wrap_anthropic(mock_anthropic_client, s)
    assert out is mock_anthropic_client


def test_instrumented_create_delegates_to_original(mock_anthropic_client, mock_anthropic_response):
    """The wrapper must call the original .create and return its response."""
    response = mock_anthropic_response(text_blocks=["hello"])
    mock_anthropic_client.messages.create.return_value = response
    original = mock_anthropic_client.messages.create

    s = Sentinel(project="proj")
    wrap_anthropic(mock_anthropic_client, s)

    out = mock_anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out is response
    original.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )


def test_instrumented_records_call(mock_anthropic_client, mock_anthropic_response):
    response = mock_anthropic_response(input_tokens=120, output_tokens=30, text_blocks=["hi back"])
    mock_anthropic_client.messages.create.return_value = response

    s = Sentinel(project="proj")
    wrap_anthropic(mock_anthropic_client, s)

    mock_anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    # Tracer should have one record under the auto-generated session id.
    sessions = list(s.tracer.all_sessions())
    assert len(sessions) == 1
    records = s.tracer.session(sessions[0])
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "anthropic"
    assert rec.method == "messages.create"
    assert rec.model == "claude-sonnet-4-6"
    assert rec.prompt_tokens == 120
    assert rec.completion_tokens == 30


def test_explicit_session_id_threaded_through(mock_anthropic_client, mock_anthropic_response):
    """Passing ``_sentinel_session_id`` keeps records in the same session bucket."""
    mock_anthropic_client.messages.create.return_value = mock_anthropic_response(text_blocks=["x"])
    s = Sentinel(project="proj")
    wrap_anthropic(mock_anthropic_client, s)

    for _ in range(3):
        mock_anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id="my-session",
        )
    # All three records are bucketed in the same session — proves the kwarg
    # flowed through the wrapper. (The kwarg-strip behavior is verified
    # separately in ``test_session_id_kwarg_stripped_before_original``.)
    assert len(s.tracer.session("my-session")) == 3


def test_session_id_kwarg_stripped_before_original(
    mock_anthropic_response,
):
    """The original .create must not receive ``_sentinel_session_id``."""
    # Build a custom client where we can inspect inner calls precisely.
    received = {}

    def real_create(**kwargs):
        received.update(kwargs)
        return mock_anthropic_response(text_blocks=["x"])

    fake_module_class = type("Anthropic", (), {"__module__": "anthropic"})
    client = fake_module_class()
    client.messages = SimpleNamespace(create=real_create)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        _sentinel_session_id="abc",
    )

    assert "_sentinel_session_id" not in received
    assert received["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Wrapper resilience: tracer/sentinel exceptions don't crash user
# ---------------------------------------------------------------------------


def test_tracer_raise_does_not_crash_user_call(
    mock_anthropic_client, mock_anthropic_response, monkeypatch
):
    """If sentinel.record_call throws, the user's API call should still succeed."""
    response = mock_anthropic_response(text_blocks=["ok"])
    mock_anthropic_client.messages.create.return_value = response

    s = Sentinel(project="proj")
    wrap_anthropic(mock_anthropic_client, s)

    # Make the tracer raise on every record.
    def boom(*a, **k):
        raise RuntimeError("tracer is broken")

    monkeypatch.setattr(s, "record_call", boom)

    out = mock_anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out is response


def test_underlying_create_exception_propagates(mock_anthropic_client):
    """If the real Anthropic call fails, the wrapper must re-raise (don't swallow)."""
    s = Sentinel(project="proj")
    mock_anthropic_client.messages.create.side_effect = RuntimeError("API down")
    wrap_anthropic(mock_anthropic_client, s)
    with pytest.raises(RuntimeError, match="API down"):
        mock_anthropic_client.messages.create(model="claude-sonnet-4-6", messages=[])


def test_block_mode_propagates_leak_detected_through_wrapper(
    mock_anthropic_response,
):
    """Block mode: the wrapper must propagate ``LeakDetected`` from record_call
    so the user's call site actually halts on detection.

    Earlier V0 had a bug where the wrapper's broad ``except Exception: pass``
    around ``sentinel.record_call`` swallowed ``LeakDetected``. Fixed by
    splitting the safety boundary: record-building exceptions are still
    suppressed, but ``record_call`` is called outside that try/except so
    block-mode raises propagate to the caller.
    """
    received_response = mock_anthropic_response(text_blocks=["x"])

    def real_create(**kwargs):
        return received_response

    fake_module_class = type("Anthropic", (), {"__module__": "anthropic"})
    client = fake_module_class()
    client.messages = SimpleNamespace(create=real_create)

    # retry_storm with min_retries=2 fires on the second identical call.
    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_anthropic(client, s)

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        _sentinel_session_id="s1",
    )
    # Second identical call fires retry_storm at confidence 0.9 — block mode
    # must raise LeakDetected through the wrapper.
    import pytest as _pytest

    from token_sentinel import LeakDetected

    with _pytest.raises(LeakDetected) as exc:
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id="s1",
        )
    assert exc.value.event.type == "retry_storm"


# ---------------------------------------------------------------------------
# _build_record — the heart of the wrapper
# ---------------------------------------------------------------------------


def test_build_record_basic_fields(mock_anthropic_response):
    response = mock_anthropic_response(input_tokens=42, output_tokens=7, text_blocks=["hi"])
    rec = _build_record(
        session_id="s1",
        kwargs={
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
        response=response,
        latency_ms=12.5,
    )
    assert isinstance(rec, CallRecord)
    assert rec.session_id == "s1"
    assert rec.provider == "anthropic"
    assert rec.method == "messages.create"
    assert rec.model == "claude-sonnet-4-6"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 7
    assert rec.latency_ms == 12.5
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    # raw_request preserves messages/tools/max_tokens
    assert rec.raw_request["max_tokens"] == 100
    assert rec.raw_request["messages"] == [{"role": "user", "content": "hi"}]
    assert rec.raw_request["tools"] == []
    # raw_response_meta carries stop_reason
    assert rec.raw_response_meta["stop_reason"] == "end_turn"


def test_build_record_with_tool_uses(mock_anthropic_response):
    response = mock_anthropic_response(
        text_blocks=None,
        tool_uses=[
            {"name": "search", "input": {"q": "kittens"}},
            {"name": "search", "input": {"q": "puppies"}},
        ],
    )
    rec = _build_record(
        session_id="s1",
        kwargs={"model": "claude-sonnet-4-6", "messages": [], "tools": []},
        response=response,
        latency_ms=10.0,
    )
    assert len(rec.tool_calls) == 2
    assert rec.tool_calls[0] == {"name": "search", "arguments": {"q": "kittens"}}
    assert rec.tool_calls[1] == {"name": "search", "arguments": {"q": "puppies"}}
    # Tool calls present → not user_facing_output even if text exists.
    assert rec.user_facing_output is False


def test_build_record_text_plus_tool_use_not_user_facing(mock_anthropic_response):
    """If both text and tool_use blocks exist, user_facing_output is False.

    Per spec: a user-facing output is a non-tool-call response. Mixed-block
    responses (text + tool_use) are intermediate, not terminal.
    """
    response = mock_anthropic_response(
        text_blocks=["I'll search now"],
        tool_uses=[{"name": "search", "input": {"q": "x"}}],
    )
    rec = _build_record(
        session_id="s1",
        kwargs={"model": "claude-sonnet-4-6", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.user_facing_output is False
    assert len(rec.tool_calls) == 1


def test_build_record_text_only_is_user_facing(mock_anthropic_response):
    response = mock_anthropic_response(text_blocks=["Final answer."])
    rec = _build_record(
        session_id="s1",
        kwargs={"model": "claude-sonnet-4-6", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.user_facing_output is True


def test_build_record_no_content_attr():
    """If the response has no ``content``, default to no tool_calls / no text."""
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    rec = _build_record(
        session_id="s1",
        kwargs={"model": "claude-sonnet-4-6", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    # missing usage → 0 tokens
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


def test_build_record_missing_usage():
    response = SimpleNamespace(content=[])
    rec = _build_record(
        session_id="s1",
        kwargs={"model": "claude-sonnet-4-6", "messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


def test_build_record_request_hash_stable():
    """Same args → same hash. Different model → different hash."""
    response = SimpleNamespace(content=[], usage=None)
    a = _build_record(
        session_id="s1",
        kwargs={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        latency_ms=10.0,
    )
    b = _build_record(
        session_id="s2",
        kwargs={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        latency_ms=99.0,
    )
    c = _build_record(
        session_id="s3",
        kwargs={"model": "y", "messages": [{"role": "user", "content": "hi"}]},
        response=response,
        latency_ms=10.0,
    )
    assert a.request_hash == b.request_hash  # latency/session don't matter
    assert a.request_hash != c.request_hash


def test_build_record_unknown_model_default():
    response = SimpleNamespace(content=[], usage=None)
    rec = _build_record(
        session_id="s1",
        kwargs={"messages": []},
        response=response,
        latency_ms=10.0,
    )
    assert rec.model == "unknown"


def test_build_record_includes_tools_in_request_hash():
    response = SimpleNamespace(content=[], usage=None)
    a = _build_record(
        session_id="s1",
        kwargs={"model": "x", "messages": [], "tools": [{"name": "t1"}]},
        response=response,
        latency_ms=10.0,
    )
    b = _build_record(
        session_id="s2",
        kwargs={"model": "x", "messages": [], "tools": [{"name": "t2"}]},
        response=response,
        latency_ms=10.0,
    )
    assert a.request_hash != b.request_hash


# ---------------------------------------------------------------------------
# _request_hash helper
# ---------------------------------------------------------------------------


def test_request_hash_stable_over_keys_order():
    """Hash is computed with sort_keys=True — kwarg-arrival order shouldn't matter."""
    a = _request_hash({"model": "m", "messages": [], "tools": [], "max_tokens": 5})
    b = _request_hash({"max_tokens": 5, "tools": [], "messages": [], "model": "m"})
    assert a == b


def test_request_hash_handles_missing_keys():
    """Defaults: model='unknown', messages=[], tools=[], max_tokens=0."""
    h = _request_hash({})
    assert isinstance(h, str)
    assert len(h) == 64


def test_request_hash_max_tokens_changes_output():
    a = _request_hash({"model": "m", "max_tokens": 10})
    b = _request_hash({"model": "m", "max_tokens": 20})
    assert a != b


# ---------------------------------------------------------------------------
# _is_async_client detection
# ---------------------------------------------------------------------------


def test_is_async_client_sync_client():
    """Plain sync client → False."""
    cls = type("Anthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=lambda **kw: None)
    assert _is_async_client(client) is False


def test_is_async_client_by_class_name():
    """A client whose class is named ``AsyncAnthropic`` → True even if create is sync."""
    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=lambda **kw: None)
    assert _is_async_client(client) is True


def test_is_async_client_by_coroutine_function():
    """If create is a coroutine function, treat client as async."""
    cls = type("Anthropic", (), {"__module__": "anthropic"})
    client = cls()

    async def acreate(**kwargs):
        return None

    client.messages = SimpleNamespace(create=acreate)
    assert _is_async_client(client) is True


def test_is_async_client_robust_to_missing_messages():
    """If client.messages doesn't exist, falls through to False (not raise)."""
    cls = type("Anthropic", (), {"__module__": "anthropic"})
    client = cls()
    # No messages attribute at all.
    assert _is_async_client(client) is False


# ---------------------------------------------------------------------------
# Async client: end-to-end through the wrapper
# ---------------------------------------------------------------------------


def test_async_wrapper_records_call(mock_anthropic_response):
    """Async client + async create → wrapper builds record on completion."""
    captured: list[dict[str, Any]] = []

    async def acreate(**kwargs):
        captured.append(kwargs)
        return mock_anthropic_response(text_blocks=["ok"])

    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=acreate)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    response = asyncio.run(
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id="async-1",
        )
    )
    assert response is not None
    assert len(s.tracer.session("async-1")) == 1
    # The original received kwargs without the sentinel session id.
    assert "_sentinel_session_id" not in captured[-1]


def test_async_wrapper_underlying_exception_propagates(mock_anthropic_response):
    async def acreate(**kwargs):
        raise RuntimeError("network down")

    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=acreate)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    with pytest.raises(RuntimeError, match="network down"):
        asyncio.run(client.messages.create(model="m", messages=[]))


# ---------------------------------------------------------------------------
# Streaming: sync messages.stream
# ---------------------------------------------------------------------------


class _FakeMessageStream:
    """A bare-bones MessageStream stand-in iterable over canned events."""

    def __init__(self, events: list[Any], final_message: Any = None):
        self._events = events
        self._final_message = final_message

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final_message

    @property
    def text_stream(self):
        return iter("")


class _FakeStreamCM:
    """Sync context manager that yields a fake MessageStream."""

    def __init__(self, stream: _FakeMessageStream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        return False


def test_sync_stream_records_call_on_exit(mock_anthropic_response):
    """Sync streaming context manager — record built on __exit__."""
    final_msg = mock_anthropic_response(
        input_tokens=50, output_tokens=20, text_blocks=["streamed answer"]
    )
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=50, output_tokens=0)),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=50, output_tokens=20),
            delta=SimpleNamespace(stop_reason="end_turn"),
        ),
        SimpleNamespace(type="message_stop", message=final_msg),
    ]
    stream = _FakeMessageStream(events, final_message=final_msg)

    def fake_stream(**kwargs):
        return _FakeStreamCM(stream)

    cls = type("Anthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=lambda **kw: None, stream=fake_stream)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    with client.messages.stream(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        _sentinel_session_id="stream-1",
    ) as ms:
        # Iterate to drive the accumulator.
        observed_events = list(ms)
        assert len(observed_events) == 3

    records = s.tracer.session("stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "messages.stream"
    assert rec.prompt_tokens == 50
    assert rec.completion_tokens == 20
    assert rec.user_facing_output is True
    assert rec.raw_response_meta.get("streamed") is True
    assert rec.raw_response_meta.get("stop_reason") == "end_turn"


def test_sync_stream_proxy_forwards_attribute_access(mock_anthropic_response):
    """Helper attrs on the underlying stream (e.g. ``get_final_message``) still work
    through the proxy."""
    final_msg = mock_anthropic_response(text_blocks=["x"])
    stream = _FakeMessageStream([], final_message=final_msg)

    def fake_stream(**kwargs):
        return _FakeStreamCM(stream)

    cls = type("Anthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=lambda **kw: None, stream=fake_stream)
    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    with client.messages.stream(model="m") as ms:
        # Forwarded via __getattr__ on _StreamProxy
        assert ms.get_final_message() is final_msg


# ---------------------------------------------------------------------------
# _UsageAccumulator unit tests
# ---------------------------------------------------------------------------


def test_usage_accumulator_message_start_seeds_input():
    acc = _UsageAccumulator()
    acc.observe(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=42, output_tokens=0)),
        )
    )
    assert acc.input_tokens == 42
    assert acc.output_tokens == 0


def test_usage_accumulator_takes_max_on_delta():
    """Output_tokens monotonically grows — max() guards regressions."""
    acc = _UsageAccumulator()
    acc.observe(
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=0, output_tokens=10),
            delta=SimpleNamespace(stop_reason=None),
        )
    )
    acc.observe(
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=0, output_tokens=25),
            delta=SimpleNamespace(stop_reason="end_turn"),
        )
    )
    # Even if a buggy SDK sends a smaller value:
    acc.observe(
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=0, output_tokens=5),
            delta=SimpleNamespace(stop_reason=None),
        )
    )
    assert acc.output_tokens == 25
    assert acc.stop_reason == "end_turn"


def test_usage_accumulator_finalize_pulls_content():
    acc = _UsageAccumulator()
    final = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=70, output_tokens=30),
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="text", text="answer"),
            SimpleNamespace(type="tool_use", name="t", input={"a": 1}),
        ],
    )
    acc.finalize_from_message(final)
    assert acc.input_tokens == 70
    assert acc.output_tokens == 30
    assert acc.has_text_output is True
    assert acc.tool_calls == [{"name": "t", "arguments": {"a": 1}}]


def test_usage_accumulator_robust_to_garbage():
    """Malformed events must not crash observe()."""
    acc = _UsageAccumulator()
    acc.observe(SimpleNamespace())  # no type
    acc.observe(SimpleNamespace(type="message_start"))  # no .message
    acc.observe(SimpleNamespace(type="message_stop"))  # no .message
    acc.finalize_from_message(None)  # no _final_message either
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0


def test_build_record_from_accumulator_basic():
    acc = _UsageAccumulator()
    acc.input_tokens = 100
    acc.output_tokens = 25
    acc.stop_reason = "end_turn"
    acc.has_text_output = True
    rec = _build_record_from_accumulator(
        session_id="s1",
        kwargs={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "max_tokens": 100,
        },
        accumulator=acc,
        latency_ms=200.0,
        method="messages.stream",
    )
    assert rec.method == "messages.stream"
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 25
    assert rec.user_facing_output is True
    assert rec.raw_response_meta["streamed"] is True


def test_build_record_from_accumulator_with_tool_calls():
    acc = _UsageAccumulator()
    acc.has_text_output = True
    acc.tool_calls = [{"name": "search", "arguments": {"q": "x"}}]
    rec = _build_record_from_accumulator(
        session_id="s1",
        kwargs={"model": "m", "messages": []},
        accumulator=acc,
        latency_ms=10.0,
        method="messages.stream",
    )
    # text + tool_calls → not user-facing
    assert rec.user_facing_output is False


# ---------------------------------------------------------------------------
# Backward-compat _build_record shim
# ---------------------------------------------------------------------------


def test_legacy_build_record_alias_still_works(mock_anthropic_response):
    """``_build_record`` is preserved for backward compatibility — it should
    delegate to ``_build_record_from_message`` with method='messages.create'."""
    response = mock_anthropic_response(text_blocks=["x"])
    rec_legacy = _build_record(
        session_id="s1",
        kwargs={"model": "m", "messages": [], "tools": []},
        response=response,
        latency_ms=5.0,
    )
    rec_new = _build_record_from_message(
        session_id="s1",
        kwargs={"model": "m", "messages": [], "tools": []},
        response=response,
        latency_ms=5.0,
        method="messages.create",
    )
    assert rec_legacy.method == "messages.create"
    assert rec_legacy.method == rec_new.method
    assert rec_legacy.request_hash == rec_new.request_hash
