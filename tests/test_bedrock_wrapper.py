"""Tests for ``token_sentinel.wrappers.bedrock.wrap_bedrock``.

NO real AWS calls. We construct mocks shaped like a boto3 ``bedrock-runtime``
client (``SimpleNamespace`` + recording callables) and verify the wrapper:
  - swaps in instrumented ``converse`` and ``converse_stream``
  - delegates to the originals unchanged (return value, exception propagation)
  - builds a ``CallRecord`` matching the response shape
  - extracts tool_use blocks correctly (both non-stream and stream)
  - aggregates usage from the stream's ``metadata`` event
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - propagates ``LeakDetected`` in block mode (the whole point of block mode)
  - is dispatched correctly by ``Sentinel.wrap`` (boto3 client recognized)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord, LeakDetected
from token_sentinel.wrappers.bedrock import (
    _build_record_from_accumulator,
    _build_record_from_converse,
    _EventStreamProxy,
    _request_hash,
    _StreamUsageAccumulator,
    wrap_bedrock,
)

# ---------------------------------------------------------------------------
# Test helpers — mock boto3 client surface
# ---------------------------------------------------------------------------


class _RecordingCallable:
    """Real callable that records calls and returns a configurable response.

    Mirrors ``conftest._RecordingCreate`` but with names that fit boto3.
    Required because ``functools.wraps`` (used inside ``wrap_bedrock``) tries
    to copy ``__name__`` / ``__qualname__`` from the wrapped object — a plain
    ``MagicMock`` exposes those as auto-generated child mocks (not strings),
    which trips Python's function-attribute type check.
    """

    __name__ = "converse"
    __qualname__ = "BedrockRuntime.converse"
    __module__ = "botocore.client"
    __annotations__: dict = {}
    __doc__ = "mock converse"

    def __init__(self, name: str = "converse"):
        self.__name__ = name
        self.__qualname__ = f"BedrockRuntime.{name}"
        self.calls: list[tuple[tuple, dict]] = []
        self.return_value: Any = None
        self.side_effect: Any = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value


def _make_bedrock_client(converse_return: Any = None, converse_stream_return: Any = None) -> Any:
    """Build a mock object shaped like a boto3 ``bedrock-runtime`` client.

    We construct a class whose ``__module__`` reports ``botocore.client`` and
    whose name contains "Bedrock" so ``Sentinel.wrap``'s dispatch logic
    (``cls_name.lower()`` containing "bedrock") routes us correctly. We also
    mount a ``meta.service_model.service_name`` attribute equal to
    ``bedrock-runtime`` so the service-name leg of the dispatch also passes.
    """
    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()
    client.converse = _RecordingCallable("converse")
    client.converse.return_value = converse_return
    client.converse_stream = _RecordingCallable("converse_stream")
    client.converse_stream.return_value = converse_stream_return
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))
    return client


def _make_converse_response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 25,
    stop_reason: str = "end_turn",
    text_blocks: list[str] | None = None,
    tool_uses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a dict shaped like a real ``converse`` response."""
    content: list[Any] = []
    if text_blocks:
        for t in text_blocks:
            content.append({"text": t})
    if tool_uses:
        for t in tool_uses:
            content.append(
                {
                    "toolUse": {
                        "toolUseId": t.get("toolUseId", "tu_1"),
                        "name": t["name"],
                        "input": t.get("input", {}),
                    }
                }
            )
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "metrics": {"latencyMs": 200},
    }


def _make_stream_events(
    *,
    input_tokens: int = 50,
    output_tokens: int = 12,
    stop_reason: str = "end_turn",
    text_chunks: list[str] | None = None,
    tool_use: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a sequence of EventStream-style dicts emitted by converse_stream.

    Mimics the real boto3 wire format: ``messageStart`` -> ``contentBlockStart``
    (-> ``contentBlockDelta`` -> ``contentBlockStop`` repeated) ->
    ``messageStop`` -> ``metadata``.
    """
    events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    block_idx = 0
    if text_chunks:
        events.append({"contentBlockStart": {"start": {}, "contentBlockIndex": block_idx}})
        for chunk in text_chunks:
            events.append(
                {
                    "contentBlockDelta": {
                        "delta": {"text": chunk},
                        "contentBlockIndex": block_idx,
                    }
                }
            )
        events.append({"contentBlockStop": {"contentBlockIndex": block_idx}})
        block_idx += 1
    if tool_use:
        events.append(
            {
                "contentBlockStart": {
                    "start": {
                        "toolUse": {
                            "toolUseId": tool_use.get("toolUseId", "tu_1"),
                            "name": tool_use["name"],
                        }
                    },
                    "contentBlockIndex": block_idx,
                }
            }
        )
        # JSON streamed in chunks — matches real Bedrock behaviour.
        import json as _json

        full_input = _json.dumps(tool_use.get("input", {}))
        # Split mid-string to exercise the accumulator's chunk-stitching.
        mid = max(1, len(full_input) // 2)
        for chunk in (full_input[:mid], full_input[mid:]):
            if not chunk:
                continue
            events.append(
                {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": chunk}},
                        "contentBlockIndex": block_idx,
                    }
                }
            )
        events.append({"contentBlockStop": {"contentBlockIndex": block_idx}})
        block_idx += 1
    events.append({"messageStop": {"stopReason": stop_reason}})
    events.append(
        {
            "metadata": {
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                "metrics": {"latencyMs": 100},
            }
        }
    )
    return events


# ---------------------------------------------------------------------------
# wrap_bedrock: instrumentation
# ---------------------------------------------------------------------------


def test_wrap_replaces_converse():
    client = _make_bedrock_client()
    original = client.converse
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)
    assert client.converse is not original
    assert callable(client.converse)


def test_wrap_replaces_converse_stream():
    client = _make_bedrock_client()
    original = client.converse_stream
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)
    assert client.converse_stream is not original
    assert callable(client.converse_stream)


def test_wrap_returns_same_client_instance():
    client = _make_bedrock_client()
    s = Sentinel(project="proj")
    out = wrap_bedrock(client, s)
    assert out is client


def test_wrap_skips_missing_methods():
    """If a client lacks ``converse`` (e.g. older botocore), don't crash."""
    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()
    # No converse, no converse_stream.
    s = Sentinel(project="proj")
    out = wrap_bedrock(client, s)  # must not raise
    assert out is client


# ---------------------------------------------------------------------------
# converse: end-to-end through the wrapper
# ---------------------------------------------------------------------------


def test_converse_delegates_and_returns_response():
    response = _make_converse_response(text_blocks=["hello"])
    client = _make_bedrock_client(converse_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    out = client.converse(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 100},
    )
    assert out is response


def test_converse_records_call_with_correct_fields():
    response = _make_converse_response(input_tokens=120, output_tokens=30, text_blocks=["hi back"])
    client = _make_bedrock_client(converse_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    client.converse(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 100},
        _sentinel_session_id="bedrock-1",
    )

    records = s.tracer.session("bedrock-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "bedrock"
    assert rec.method == "converse"
    assert rec.model == "anthropic.claude-sonnet-4-5-v2:0"
    assert rec.prompt_tokens == 120
    assert rec.completion_tokens == 30
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_response_meta["stopReason"] == "end_turn"


def test_converse_session_id_kwarg_stripped_before_original():
    """The original ``converse`` must not receive ``_sentinel_session_id``."""
    received: dict[str, Any] = {}

    def real_converse(**kwargs):
        received.update(kwargs)
        return _make_converse_response(text_blocks=["x"])

    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()
    client.converse = real_converse
    client.converse_stream = lambda **kw: None
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))

    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    client.converse(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        _sentinel_session_id="abc",
    )
    assert "_sentinel_session_id" not in received
    assert received["modelId"] == "anthropic.claude-sonnet-4-5-v2:0"


def test_converse_extracts_tool_uses():
    """Tool-use blocks in ``output.message.content`` must surface as
    ``CallRecord.tool_calls`` so the tool_loop rule can score them."""
    response = _make_converse_response(
        text_blocks=None,
        tool_uses=[
            {"name": "search", "input": {"q": "kittens"}},
            {"name": "search", "input": {"q": "puppies"}},
        ],
    )
    client = _make_bedrock_client(converse_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    client.converse(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[],
        toolConfig={"tools": []},
        _sentinel_session_id="t1",
    )
    rec = s.tracer.session("t1")[0]
    assert len(rec.tool_calls) == 2
    assert rec.tool_calls[0] == {"name": "search", "arguments": {"q": "kittens"}}
    assert rec.tool_calls[1] == {"name": "search", "arguments": {"q": "puppies"}}
    # Mixed/tool-use response → not user-facing.
    assert rec.user_facing_output is False


def test_converse_underlying_exception_propagates():
    """If the boto3 call raises, the wrapper must re-raise (don't swallow)."""
    client = _make_bedrock_client()
    client.converse.side_effect = RuntimeError("AWS down")
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)
    with pytest.raises(RuntimeError, match="AWS down"):
        client.converse(modelId="m", messages=[])


def test_converse_record_call_exception_swallowed(monkeypatch):
    """If sentinel.record_call throws non-LeakDetected, the user call still succeeds."""
    response = _make_converse_response(text_blocks=["ok"])
    client = _make_bedrock_client(converse_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    def boom(*a, **k):
        raise RuntimeError("tracer broken")

    monkeypatch.setattr(s, "record_call", boom)
    out = client.converse(modelId="m", messages=[])
    assert out is response


def test_converse_record_building_failure_isolated():
    """If record-building blows up, return the response and don't crash.

    Driven by handing the wrapper an unexpected response shape (not a dict
    and not the right SimpleNamespace fields). The wrapper's two-level safety
    boundary should swallow the building exception.
    """

    # Response that breaks the dict-path code (not a dict, no usage attr).
    class WeirdResponse:
        def __getattr__(self, name):
            raise AttributeError("boom")

    weird = WeirdResponse()
    client = _make_bedrock_client(converse_return=weird)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    out = client.converse(modelId="m", messages=[])
    assert out is weird  # user got their response


# ---------------------------------------------------------------------------
# Block-mode propagation
# ---------------------------------------------------------------------------


def test_block_mode_propagates_leak_detected_through_wrapper():
    """The wrapper must propagate ``LeakDetected`` from record_call so the user's
    call site actually halts on detection.

    Mirrors the same safety check applied to the Anthropic and OpenAI wrappers
    — the broad ``except Exception`` around record_call must not swallow
    ``LeakDetected`` (the entire point of mode='block').
    """
    response = _make_converse_response(text_blocks=["x"])

    def real_converse(**kwargs):
        return response

    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()
    client.converse = real_converse
    client.converse_stream = lambda **kw: None
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))

    # retry_storm with min_retries=2 fires on the second identical call.
    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_bedrock(client, s)

    client.converse(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 100},
        _sentinel_session_id="s1",
    )
    with pytest.raises(LeakDetected) as exc:
        client.converse(
            modelId="anthropic.claude-sonnet-4-5-v2:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 100},
            _sentinel_session_id="s1",
        )
    assert exc.value.event.type == "retry_storm"


# ---------------------------------------------------------------------------
# converse_stream: end-to-end
# ---------------------------------------------------------------------------


def test_converse_stream_records_call_after_iteration():
    """Iterating the proxy stream end-to-end builds a record on completion."""
    events = _make_stream_events(input_tokens=50, output_tokens=12, text_chunks=["he", "llo"])
    response = {"ResponseMetadata": {}, "stream": iter(events)}
    client = _make_bedrock_client(converse_stream_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    result = client.converse_stream(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 50},
        _sentinel_session_id="stream-1",
    )

    # Drive the proxy.
    observed = list(result["stream"])
    assert len(observed) == len(events)

    records = s.tracer.session("stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "converse_stream"
    assert rec.provider == "bedrock"
    assert rec.prompt_tokens == 50
    assert rec.completion_tokens == 12
    assert rec.user_facing_output is True
    assert rec.raw_response_meta.get("streamed") is True
    assert rec.raw_response_meta.get("stopReason") == "end_turn"


def test_converse_stream_extracts_tool_use_from_deltas():
    """Streamed tool_use blocks (name in contentBlockStart, JSON in deltas)
    must be stitched back together and surface as ``tool_calls``."""
    events = _make_stream_events(
        text_chunks=None,
        tool_use={"name": "search", "input": {"q": "kittens"}},
    )
    response = {"ResponseMetadata": {}, "stream": iter(events)}
    client = _make_bedrock_client(converse_stream_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    result = client.converse_stream(
        modelId="anthropic.claude-sonnet-4-5-v2:0",
        messages=[],
        toolConfig={"tools": []},
        _sentinel_session_id="stream-2",
    )
    list(result["stream"])

    rec = s.tracer.session("stream-2")[0]
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0] == {"name": "search", "arguments": {"q": "kittens"}}
    # Tool-use only → not user-facing.
    assert rec.user_facing_output is False


def test_converse_stream_passes_through_when_no_stream_key():
    """If the underlying response lacks a 'stream' key, return untouched."""
    response = {"ResponseMetadata": {}}  # no stream
    client = _make_bedrock_client(converse_stream_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)
    out = client.converse_stream(modelId="m", messages=[])
    assert out is response


# ---------------------------------------------------------------------------
# _StreamUsageAccumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_extracts_usage_from_metadata():
    acc = _StreamUsageAccumulator()
    acc.observe({"messageStart": {"role": "assistant"}})
    acc.observe({"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}})
    acc.observe({"messageStop": {"stopReason": "end_turn"}})
    acc.observe(
        {
            "metadata": {
                "usage": {
                    "inputTokens": 200,
                    "outputTokens": 50,
                    "totalTokens": 250,
                }
            }
        }
    )
    assert acc.input_tokens == 200
    assert acc.output_tokens == 50
    assert acc.has_text_output is True
    assert acc.stop_reason == "end_turn"
    assert acc.metadata_seen is True


def test_accumulator_stitches_tool_use_chunks():
    """``contentBlockDelta`` for toolUse arrives with chunked JSON; the
    accumulator must concatenate and parse on stop."""
    acc = _StreamUsageAccumulator()
    acc.observe(
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu_1", "name": "search"}},
                "contentBlockIndex": 0,
            }
        }
    )
    acc.observe(
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"q":'}},
                "contentBlockIndex": 0,
            }
        }
    )
    acc.observe(
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '"hello"}'}},
                "contentBlockIndex": 0,
            }
        }
    )
    acc.observe({"contentBlockStop": {"contentBlockIndex": 0}})
    assert acc.tool_calls == [{"name": "search", "arguments": {"q": "hello"}}]


def test_accumulator_handles_invalid_tool_input_json():
    """Tool-use input that isn't valid JSON should fall back to the raw string."""
    acc = _StreamUsageAccumulator()
    acc.observe(
        {
            "contentBlockStart": {
                "start": {"toolUse": {"name": "t"}},
                "contentBlockIndex": 0,
            }
        }
    )
    acc.observe(
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": "not json"}},
                "contentBlockIndex": 0,
            }
        }
    )
    acc.observe({"contentBlockStop": {"contentBlockIndex": 0}})
    assert acc.tool_calls == [{"name": "t", "arguments": "not json"}]


def test_accumulator_robust_to_garbage():
    """Malformed events must not crash observe()."""
    acc = _StreamUsageAccumulator()
    acc.observe({})  # no recognised key
    acc.observe(None)  # not a dict
    acc.observe({"contentBlockDelta": None})
    acc.observe({"metadata": None})
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0


# ---------------------------------------------------------------------------
# _build_record_from_converse — direct unit tests
# ---------------------------------------------------------------------------


def test_build_record_basic_fields():
    response = _make_converse_response(input_tokens=42, output_tokens=7, text_blocks=["hi"])
    rec = _build_record_from_converse(
        session_id="s1",
        kwargs={
            "modelId": "anthropic.claude-sonnet-4-5-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 100},
        },
        response=response,
        latency_ms=12.5,
        method="converse",
    )
    assert isinstance(rec, CallRecord)
    assert rec.provider == "bedrock"
    assert rec.method == "converse"
    assert rec.model == "anthropic.claude-sonnet-4-5-v2:0"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 7
    assert rec.latency_ms == 12.5
    assert rec.user_facing_output is True
    assert rec.tool_calls == []
    assert rec.raw_response_meta["stopReason"] == "end_turn"


def test_build_record_text_plus_tool_use_not_user_facing():
    """If both text and tool_use blocks exist, user_facing_output is False."""
    response = _make_converse_response(
        text_blocks=["I'll search now"],
        tool_uses=[{"name": "search", "input": {"q": "x"}}],
    )
    rec = _build_record_from_converse(
        session_id="s1",
        kwargs={"modelId": "m", "messages": []},
        response=response,
        latency_ms=10.0,
        method="converse",
    )
    assert rec.user_facing_output is False
    assert len(rec.tool_calls) == 1


def test_build_record_no_usage_field():
    """Missing usage → 0 tokens, no crash."""
    response = {"output": {"message": {"content": []}}}
    rec = _build_record_from_converse(
        session_id="s1",
        kwargs={"modelId": "m", "messages": []},
        response=response,
        latency_ms=10.0,
        method="converse",
    )
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0


def test_build_record_unknown_model_default():
    response = {"output": {"message": {"content": []}}, "usage": {}}
    rec = _build_record_from_converse(
        session_id="s1",
        kwargs={"messages": []},
        response=response,
        latency_ms=10.0,
        method="converse",
    )
    assert rec.model == "unknown"


def test_build_record_from_accumulator_basic():
    acc = _StreamUsageAccumulator()
    acc.input_tokens = 100
    acc.output_tokens = 25
    acc.stop_reason = "end_turn"
    acc.has_text_output = True
    rec = _build_record_from_accumulator(
        session_id="s1",
        kwargs={
            "modelId": "anthropic.claude-sonnet-4-5-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "toolConfig": {},
        },
        accumulator=acc,
        latency_ms=200.0,
        method="converse_stream",
    )
    assert rec.method == "converse_stream"
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 25
    assert rec.user_facing_output is True
    assert rec.raw_response_meta["streamed"] is True
    assert rec.provider == "bedrock"


# ---------------------------------------------------------------------------
# _request_hash
# ---------------------------------------------------------------------------


def test_request_hash_stable_over_key_order():
    """Hash is computed with sort_keys=True — kwarg-arrival order shouldn't matter."""
    a = _request_hash(
        {
            "modelId": "m",
            "messages": [],
            "toolConfig": {},
            "inferenceConfig": {"maxTokens": 5},
        }
    )
    b = _request_hash(
        {
            "inferenceConfig": {"maxTokens": 5},
            "toolConfig": {},
            "messages": [],
            "modelId": "m",
        }
    )
    assert a == b


def test_request_hash_changes_on_model():
    a = _request_hash({"modelId": "x", "messages": []})
    b = _request_hash({"modelId": "y", "messages": []})
    assert a != b


def test_request_hash_handles_missing_keys():
    h = _request_hash({})
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Sentinel.wrap dispatch detection
# ---------------------------------------------------------------------------


def test_sentinel_wrap_dispatches_to_bedrock_by_class_name():
    """A boto3 client whose class name contains 'Bedrock' must be routed to
    ``wrap_bedrock`` by ``Sentinel.wrap``."""
    client = _make_bedrock_client(converse_return=_make_converse_response(text_blocks=["ok"]))
    s = Sentinel(project="proj")
    out = s.wrap(client)
    assert out is client
    # Verify by calling — the wrapper must record.
    out.converse(modelId="m", messages=[], _sentinel_session_id="d1")
    assert len(s.tracer.session("d1")) == 1


def test_sentinel_wrap_dispatches_by_service_name():
    """A class without 'bedrock' in its name but whose ``meta.service_model``
    reports a service starting with ``bedrock`` must still be routed."""
    cls = type("ClientShim", (), {"__module__": "botocore.client"})
    client = cls()
    client.converse = _RecordingCallable("converse")
    client.converse.return_value = _make_converse_response(text_blocks=["ok"])
    client.converse_stream = _RecordingCallable("converse_stream")
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))

    s = Sentinel(project="proj")
    out = s.wrap(client)
    assert out is client
    out.converse(modelId="m", messages=[], _sentinel_session_id="d2")
    assert len(s.tracer.session("d2")) == 1


def test_sentinel_wrap_rejects_non_bedrock_botocore():
    """A botocore client for a non-Bedrock service must NOT be routed to
    ``wrap_bedrock`` — it should hit the unsupported-client error."""
    cls = type("S3", (), {"__module__": "botocore.client"})
    client = cls()
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="s3"))
    s = Sentinel(project="proj")
    with pytest.raises(TypeError, match="Unsupported client type"):
        s.wrap(client)


# ---------------------------------------------------------------------------
# EventStreamProxy: close / double-finalize protection
# ---------------------------------------------------------------------------


def test_event_stream_proxy_finalizes_only_once():
    """Iterating to completion AND calling close() must produce ONE record,
    not two — the ``_flushed`` guard prevents double-counting."""
    events = _make_stream_events(input_tokens=10, output_tokens=5, text_chunks=["x"])
    response = {"stream": iter(events)}
    client = _make_bedrock_client(converse_stream_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    result = client.converse_stream(modelId="m", messages=[], _sentinel_session_id="once-1")
    list(result["stream"])  # first finalize
    # Calling close after exhaustion should not double-record.
    closer = getattr(result["stream"], "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass

    assert len(s.tracer.session("once-1")) == 1


def test_event_stream_proxy_finalizes_on_close_without_iteration():
    """If a user obtains the stream but never iterates, calling ``close()``
    still flushes a CallRecord (best-effort — usage is 0 but the record
    exists, which is the right semantic for retry_storm/zombie detection)."""
    events = _make_stream_events(input_tokens=10, output_tokens=5, text_chunks=["x"])

    class _Closeable:
        def __init__(self, evs):
            self._evs = evs
            self.closed = False

        def __iter__(self):
            return iter(self._evs)

        def close(self):
            self.closed = True

    raw_stream = _Closeable(events)
    response = {"stream": raw_stream}
    client = _make_bedrock_client(converse_stream_return=response)
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    result = client.converse_stream(modelId="m", messages=[], _sentinel_session_id="close-1")
    proxy = result["stream"]
    proxy.close()
    assert raw_stream.closed is True
    # Closed without iterating → record exists, but tokens are 0.
    records = s.tracer.session("close-1")
    assert len(records) == 1
    assert records[0].prompt_tokens == 0


def test_event_stream_proxy_forwards_attribute_access():
    """Helper attrs on the underlying stream still work through the proxy."""
    raw_stream = SimpleNamespace(custom_attr="hello", __iter__=lambda self: iter([]))

    proxy = _EventStreamProxy(
        stream=raw_stream,
        accumulator=_StreamUsageAccumulator(),
        sentinel=Sentinel(project="p"),
        kwargs={"modelId": "m", "messages": []},
        session_id="x",
        start=0.0,
    )
    # Forwarded via __getattr__
    assert proxy.custom_attr == "hello"
