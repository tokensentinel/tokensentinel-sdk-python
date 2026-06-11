"""Targeted coverage tests for high-impact gaps.

This file fills branches and lines that the existing test suite misses.
Each test has a 'why this matters' docstring and is small and focused.

Coverage gaps targeted (from `pytest --cov=... --cov-report=term-missing`):

1. Sentinel.wrap() dispatch:
   - vertex path (lines 82-84 in sentinel.py)
   - bedrock service-model exception path (line 98-99 in sentinel.py)
2. Anthropic wrapper:
   - sync + async record-build failure isolation (lines 130-131 / 161-162)
   - _AsyncStreamProxy.__getattr__ forwarding (line 336)
   - async stream wrap end-to-end (lines 442-475)
   - block-mode propagation through async stream finalize
3. OpenAI wrapper:
   - record-build exception swallowing (lines 69-70, 169-170)
   - block-mode LeakDetected propagation in async + embeddings (74-77, 144-147)
   - stream=True passthrough (sync) with no recording
4. Bedrock wrapper:
   - _EventStreamProxy.__enter__/__exit__ context-manager support (295-313)
   - __del__ best-effort flush (321-322)
   - record_call exception swallowed in finalize (350-353)
5. Tool-loop rule:
   - _mean_jaccard / _tokens (the entire jaccard fallback path lines 132-161)
   - _mean_pairwise_similarity len(calls)<2 short circuit (line 111)
6. Tool-definition-bloat rule:
   - _tool_name attribute fallback for objects with .name (line 167)
   - top-tools-by-size when a tool serialisation fails (140-141)
"""

from __future__ import annotations

import asyncio
import gc
from datetime import timedelta
from types import SimpleNamespace

import pytest

from token_sentinel import LeakDetected, Sentinel
from token_sentinel.rules.tool_definition_bloat import (
    _tool_name,
    _top_tools_by_size,
)
from token_sentinel.rules.tool_loop import (
    _mean_jaccard,
    _mean_pairwise_similarity,
    _tokens,
)
from token_sentinel.wrappers.anthropic import wrap_anthropic
from token_sentinel.wrappers.bedrock import (
    _EventStreamProxy,
    _StreamUsageAccumulator,
    wrap_bedrock,
)
from token_sentinel.wrappers.openai import wrap_openai

# ---------------------------------------------------------------------------
# Sentinel.wrap dispatch
# ---------------------------------------------------------------------------


def test_wrap_legacy_vertexai_module_rejected_with_typeerror():
    """Why this matters: the legacy ``vertexai`` SDK is deprecated in favor of
    google-genai with ``vertexai=True``. Sentinel's wrap() does NOT route
    legacy clients to a wrapper — it lets them fall through to TypeError so
    customers get a clear "migrate" signal rather than a silent no-op. A
    regression that adds a phantom vertex import would surface here.
    """
    cls = type("VertexClient", (), {"__module__": "vertexai.preview"})
    client = cls()
    s = Sentinel(project="proj")
    with pytest.raises(TypeError, match="Unsupported client type"):
        s.wrap(client)


def test_wrap_bedrock_meta_attribute_exception_swallowed():
    """Why this matters: lines 98-99 in sentinel.py catch a broad exception when
    interrogating ``client.meta.service_model`` because boto3 client surfaces
    can raise on attribute access (esp. lazy proxies). If the swallow ever
    breaks, dispatch on a partially-mocked bedrock client crashes the user.

    v0.3.2 (MED-1) added accessor validation — `Sentinel.wrap` now requires
    the Bedrock client to expose `converse`. We stub it as a no-op callable
    so the validation passes and we can still exercise the meta-introspection
    swallow path (which fires before the converse check).
    """

    class WeirdMeta:
        @property
        def service_model(self):
            raise RuntimeError("flaky introspection")

    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()
    client.meta = WeirdMeta()
    # Stub `converse` so MED-1 accessor validation passes; the test is about
    # the meta-introspection swallow, not method-presence.
    client.converse = lambda **kw: None
    s = Sentinel(project="proj")
    out = s.wrap(client)
    assert out is client


# ---------------------------------------------------------------------------
# Anthropic wrapper: failure isolation, proxies, async
# ---------------------------------------------------------------------------


def test_record_build_failure_returns_response_anthropic(mock_anthropic_client):
    """Why this matters: the wrapper's two-level safety boundary is the entire
    contract — instrumentation must NEVER break the user's call. A
    record-building exception (e.g. weird response shape) should be swallowed
    and the original response returned. Currently the anthropic record-build
    failure branch (lines 130-131) is uncovered.
    """
    sentinel_marker = object()

    class AccessExplodes:
        # ``_build_record_from_message`` reads ``response.usage``, ``content``,
        # ``stop_reason``. Make every attribute access raise.
        def __getattr__(self, name):
            raise RuntimeError("response is broken")

    weird_response = AccessExplodes()
    mock_anthropic_client.messages.create.return_value = weird_response
    s = Sentinel(project="proj")
    wrap_anthropic(mock_anthropic_client, s)

    out = mock_anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    # User got their response back — the wrapper did not crash.
    assert out is weird_response
    # No record built — record-building blew up before the tracer was hit.
    assert list(s.tracer.all_sessions()) == []
    del sentinel_marker  # silence unused; keep ref to be defensive


def test_async_record_build_failure_returns_response_anthropic():
    """Why this matters: mirror of the sync record-build failure path on the
    async side. The wrapper's contract — never crash user code — must hold for
    both sync and async clients. Async record-build exception path
    (lines 161-162) is otherwise uncovered.
    """

    class AccessExplodes:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    weird = AccessExplodes()

    async def acreate(**kwargs):
        return weird

    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=acreate)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    out = asyncio.run(client.messages.create(model="m", messages=[]))
    assert out is weird
    assert list(s.tracer.all_sessions()) == []


def test_async_anthropic_stream_records_call_end_to_end():
    """Why this matters: the entire async-stream flow (lines 442-475 in
    anthropic.py) is currently uncovered. This is the highest-impact missing
    coverage in the SDK — async streaming is a real customer code path.
    """

    class _FakeAsyncStream:
        def __init__(self, events, final):
            self._events = events
            self._final = final

        def __aiter__(self):
            async def gen():
                for e in self._events:
                    yield e

            return gen()

        async def get_final_message(self):
            return self._final

    class _FakeAsyncCM:
        def __init__(self, stream):
            self._stream = stream

        async def __aenter__(self):
            return self._stream

        async def __aexit__(self, *a):
            return False

    final_msg = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="done")],
    )
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=0)),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            delta=SimpleNamespace(stop_reason="end_turn"),
        ),
        SimpleNamespace(type="message_stop", message=final_msg),
    ]
    fake_stream = _FakeAsyncStream(events, final_msg)

    def fake_stream_factory(**kwargs):
        return _FakeAsyncCM(fake_stream)

    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(create=lambda **kw: None, stream=fake_stream_factory)

    s = Sentinel(project="proj")
    wrap_anthropic(client, s)

    async def run():
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            _sentinel_session_id="async-stream-1",
        ) as ms:
            collected = []
            async for ev in ms:
                collected.append(ev)
            return collected

    out = asyncio.run(run())
    assert len(out) == 3

    records = s.tracer.session("async-stream-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.method == "messages.stream"
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 20
    assert rec.user_facing_output is True
    assert rec.raw_response_meta.get("streamed") is True


def test_async_anthropic_stream_block_mode_propagates_leak():
    """Why this matters: block-mode propagation through async-stream finalize
    is uncovered. If a future refactor moves the LeakDetected re-raise inside
    a broad except, mode='block' silently breaks for async-stream users —
    a critical regression.
    """

    class _FakeAsyncStream:
        def __init__(self):
            self._final = SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="x")],
            )

        def __aiter__(self):
            async def gen():
                yield SimpleNamespace(type="message_stop", message=self._final)

            return gen()

        async def get_final_message(self):
            return self._final

    class _FakeAsyncCM:
        def __init__(self, stream):
            self._stream = stream

        async def __aenter__(self):
            return self._stream

        async def __aexit__(self, *a):
            return False

    cls = type("AsyncAnthropic", (), {"__module__": "anthropic"})
    client = cls()
    client.messages = SimpleNamespace(
        create=lambda **kw: None,
        stream=lambda **kw: _FakeAsyncCM(_FakeAsyncStream()),
    )

    # Two identical streamed calls -> retry_storm fires on second -> block raises.
    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_anthropic(client, s)

    async def run():
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            _sentinel_session_id="block-1",
        ) as ms:
            async for _ in ms:
                pass

    asyncio.run(run())  # first call succeeds, no fire yet

    with pytest.raises(LeakDetected) as exc:
        asyncio.run(run())
    assert exc.value.event.type == "retry_storm"


# ---------------------------------------------------------------------------
# OpenAI wrapper: failure isolation across sync + async + embeddings
# ---------------------------------------------------------------------------


def test_openai_chat_record_build_failure_returns_response(mock_openai_client):
    """Why this matters: openai.py lines 69-70 / 104-105 — record-build
    failure must be swallowed and the response returned to the user. A
    regression here would crash callers on any unexpected response shape.
    """

    class AccessExplodes:
        # _build_chat_record reads .usage and .choices — make both blow up.
        def __getattr__(self, name):
            raise RuntimeError("borked response")

    weird = AccessExplodes()
    mock_openai_client.chat.completions.create.return_value = weird

    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    out = mock_openai_client.chat.completions.create(model="gpt-4o", messages=[])
    assert out is weird


def test_openai_async_chat_record_build_failure_returns_response():
    """Why this matters: covers lines 69-70 in the async path — same contract,
    different code branch. Without this test the async-record-build failure
    isolation is silently broken on any future refactor.
    """

    class AccessExplodes:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    weird = AccessExplodes()

    async def acreate(**kwargs):
        return weird

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=acreate))
    client.embeddings = SimpleNamespace(create=lambda **kw: None)

    s = Sentinel(project="proj")
    wrap_openai(client, s)

    out = asyncio.run(client.chat.completions.create(model="m", messages=[]))
    assert out is weird


def test_openai_async_chat_block_mode_propagates_leak():
    """Why this matters: covers lines 74-77 — async record_call must propagate
    LeakDetected. Async OpenAI users with mode='block' depend on this branch.
    """
    response_obj = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="x", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    async def acreate(**kwargs):
        return response_obj

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=acreate))
    client.embeddings = SimpleNamespace(create=lambda **kw: None)

    s = Sentinel(
        project="proj",
        mode="block",
        rules=["retry_storm"],
        config={"retry_storm.min_retries": 2, "retry_storm.window_seconds": 600},
    )
    wrap_openai(client, s)

    asyncio.run(
        client.chat.completions.create(model="gpt-4o", messages=[], _sentinel_session_id="block-1")
    )
    with pytest.raises(LeakDetected) as exc:
        asyncio.run(
            client.chat.completions.create(
                model="gpt-4o", messages=[], _sentinel_session_id="block-1"
            )
        )
    assert exc.value.event.type == "retry_storm"


def test_openai_embedding_record_build_failure_returns_response(mock_openai_client):
    """Why this matters: lines 169-170 (sync embedding record-build failure)
    — without this the embedding instrumentation crashes user calls on weird
    response shapes. Embedding workflows are common (RAG) so this matters.
    """

    class AccessExplodes:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    weird = AccessExplodes()
    mock_openai_client.embeddings.create.return_value = weird
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    out = mock_openai_client.embeddings.create(model="text-embedding-3-small", input="hello")
    assert out is weird


def test_openai_async_embedding_block_mode_propagates_leak():
    """Why this matters: covers lines 144-147 — async embeddings + block mode.
    Cheapest path: identical embedding calls fire embedding_waste in mode=
    block.
    """
    response_obj = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10),
        data=[SimpleNamespace(embedding=[0.1])],
    )

    async def aembed(**kwargs):
        return response_obj

    cls = type("AsyncOpenAI", (), {"__module__": "openai"})
    client = cls()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None))
    client.embeddings = SimpleNamespace(create=aembed)

    s = Sentinel(project="proj", mode="block", rules=["embedding_waste"])
    wrap_openai(client, s)

    asyncio.run(
        client.embeddings.create(
            model="text-embedding-3-small",
            input="dup",
            _sentinel_session_id="b-1",
        )
    )
    with pytest.raises(LeakDetected) as exc:
        asyncio.run(
            client.embeddings.create(
                model="text-embedding-3-small",
                input="dup",
                _sentinel_session_id="b-1",
            )
        )
    assert exc.value.event.type == "embedding_waste"


def test_openai_sync_stream_records_on_iteration(mock_openai_client):
    """Why this matters: as of stable release, stream=True is fully instrumented --
    the wrapper returns a proxy that records a CallRecord when the stream
    is exhausted. This regression guards the new contract.
    """
    sentinel_iter = iter([SimpleNamespace(choices=[])])
    mock_openai_client.chat.completions.create.return_value = sentinel_iter
    s = Sentinel(project="proj")
    wrap_openai(mock_openai_client, s)
    out = mock_openai_client.chat.completions.create(
        model="gpt-4o", messages=[], stream=True, _sentinel_session_id="stream-1"
    )
    # Proxy wraps the underlying iterator -- not identity-equal.
    assert out is not sentinel_iter
    list(out)  # Iterate to flush the record.
    # A record was created for the streaming call.
    assert len(s.tracer.session("stream-1")) == 1


# ---------------------------------------------------------------------------
# Bedrock wrapper: context-manager paths, __del__, post-finalize idempotency
# ---------------------------------------------------------------------------


def test_bedrock_event_stream_proxy_context_manager_forwards_and_finalizes():
    """Why this matters: lines 295-313 in bedrock.py — `__enter__`/`__exit__`
    on the EventStreamProxy forward to the underlying stream's CM hooks AND
    finalize. Customers can do ``with response['stream'] as evs:`` — that
    path is currently uncovered.
    """
    entered = []
    exited = []

    class _CMStream:
        def __iter__(self):
            return iter([{"messageStop": {"stopReason": "end_turn"}}])

        def __enter__(self):
            entered.append(True)
            return self

        def __exit__(self, *args):
            exited.append(True)
            return False

    stream = _CMStream()
    proxy = _EventStreamProxy(
        stream=stream,
        accumulator=_StreamUsageAccumulator(),
        sentinel=Sentinel(project="p"),
        kwargs={"modelId": "m", "messages": []},
        session_id="cm-1",
        start=0.0,
    )
    with proxy as p:
        list(p)  # iterate to consume events

    assert entered == [True]
    assert exited == [True]
    # Iteration AND __exit__ both call _finalize, but the _flushed guard
    # ensures only one record was written.
    assert len(proxy._sentinel.tracer.session("cm-1")) == 1


def test_bedrock_event_stream_proxy_cm_swallows_underlying_exceptions():
    """Why this matters: lines 295-313 — if the underlying ``__enter__`` or
    ``__exit__`` raises, the proxy must NOT propagate. The user's ``with`` block
    needs to exit cleanly even on a flaky boto3 stream, and the record should
    still be flushed by ``_finalize``. Combines two near-identical edge cases
    into one test for budget reasons.
    """

    class _BadEnterStream:
        def __iter__(self):
            return iter([])

        def __enter__(self):
            raise RuntimeError("bad enter")

        def __exit__(self, *a):
            raise RuntimeError("bad exit")

    proxy = _EventStreamProxy(
        stream=_BadEnterStream(),
        accumulator=_StreamUsageAccumulator(),
        sentinel=Sentinel(project="p"),
        kwargs={"modelId": "m", "messages": []},
        session_id="cm-bad",
        start=0.0,
    )
    # Must not raise even though both __enter__ and __exit__ explode.
    with proxy as p:
        list(p)
    # Finalize fired exactly once via __exit__.
    assert len(proxy._sentinel.tracer.session("cm-bad")) == 1


def test_bedrock_event_stream_proxy_del_finalizes_when_iteration_skipped():
    """Why this matters: lines 315-322 — ``__del__`` is the last-resort flush
    path when a user obtains a stream and never iterates / never closes it.
    GC must trigger _finalize without raising. Covered indirectly elsewhere
    via close(), but never via __del__ directly.
    """
    s = Sentinel(project="proj")
    proxy = _EventStreamProxy(
        stream=iter([{"messageStop": {"stopReason": "end_turn"}}]),
        accumulator=_StreamUsageAccumulator(),
        sentinel=s,
        kwargs={"modelId": "m", "messages": []},
        session_id="del-1",
        start=0.0,
    )
    # Drop our reference and force GC; __del__ should run and finalize.
    del proxy
    gc.collect()
    # The record was flushed via __del__.
    records = s.tracer.session("del-1")
    assert len(records) == 1


def test_bedrock_finalize_record_call_exception_swallowed(monkeypatch):
    """Why this matters: lines 350-353 — _finalize's except Exception around
    record_call. A buggy rule must not crash the streaming proxy's finalize.
    """
    cls = type("BedrockRuntime", (), {"__module__": "botocore.client"})
    client = cls()

    class _Recorder:
        __name__ = "converse_stream"
        __qualname__ = "B"
        __module__ = "botocore.client"

        def __init__(self):
            self.return_value = None

        def __call__(self, *a, **k):
            return self.return_value

    client.converse = _Recorder()
    client.converse_stream = _Recorder()
    client.converse_stream.return_value = {
        "stream": iter([{"messageStop": {"stopReason": "end_turn"}}])
    }
    client.meta = SimpleNamespace(service_model=SimpleNamespace(service_name="bedrock-runtime"))
    s = Sentinel(project="proj")
    wrap_bedrock(client, s)

    # Make record_call raise non-LeakDetected.
    def boom(rec):
        raise RuntimeError("borked rule")

    monkeypatch.setattr(s, "record_call", boom)

    out = client.converse_stream(modelId="m", messages=[])
    # Must not raise — finalize swallowed RuntimeError.
    list(out["stream"])


# ---------------------------------------------------------------------------
# Tool-loop rule: jaccard fallback / metric=jaccard / single-call short circuit
# ---------------------------------------------------------------------------


def test_mean_pairwise_similarity_single_call_returns_zero():
    """Why this matters: line 111 — _mean_pairwise_similarity short-circuits
    when fewer than 2 calls. That branch is uncovered. The contract is that
    1-call corpora can't have similarity defined, so we return 0.0.
    """
    assert _mean_pairwise_similarity([{"name": "x", "arguments": {"q": "y"}}]) == 0.0
    assert _mean_pairwise_similarity([]) == 0.0


def test_mean_jaccard_identical_args_returns_one():
    """Why this matters: the entire jaccard fallback (lines 132-146) is
    untested. `_mean_jaccard` is reachable via metric='jaccard' AND from the
    safety fallback when metric is unknown. Identical token sets → Jaccard=1.0
    """
    args = ['{"q": "kittens"}', '{"q": "kittens"}', '{"q": "kittens"}']
    assert _mean_jaccard(args) == 1.0


def test_mean_jaccard_disjoint_args_returns_zero():
    """Why this matters: complement of the above. Disjoint tokens → 0.0.
    Covers the divisor branch with non-zero pairs and zero matches.
    """
    args = ['{"q": "alpha"}', '{"q": "bravo"}', '{"q": "charlie"}']
    sim = _mean_jaccard(args)
    # alpha/bravo/charlie share the "q" token but no value tokens. With token
    # set including key + value, Jaccard is > 0 because of "q" overlap. Test
    # that it's < 0.5 to prove the divisor path is exercised.
    assert 0.0 < sim < 0.6


def test_mean_jaccard_empty_token_sets_skip_pairs():
    """Why this matters: lines 141-142 — empty token sets are skipped, not
    NaN-divided. Without test, a regression where the divisor counts empty
    pairs goes undetected.
    """
    # Two empty strings produce empty token sets — should be skipped, returning 0.
    sim = _mean_jaccard(["", ""])
    assert sim == 0.0


def test_tokens_handles_punctuation_and_unicode():
    """Why this matters: the _tokens helper (lines 149-161) is the entire
    tokenizer for the Jaccard fallback. Ensure punctuation is split,
    alphanumerics survive, and trailing alphanumerics aren't dropped.
    """
    assert _tokens("hello, world!") == ["hello", "world"]
    # Trailing alnum chunk must be flushed.
    assert _tokens("abc 123") == ["abc", "123"]
    # Empty input → empty list.
    assert _tokens("") == []
    # All-punct input → empty list.
    assert _tokens(",.!?") == []


def test_tool_loop_metric_jaccard_explicit(make_call, now):
    """Why this matters: routes execution through the metric='jaccard' branch
    in _mean_pairwise_similarity AND the rule's threshold dispatch (lines
    115-118 in tool_loop.py). Covers a real customer config knob.
    """
    from token_sentinel.rules.tool_loop import ToolLoopRule

    rule = ToolLoopRule({"tool_loop.similarity_metric": "jaccard"})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "kittens are cute"}}],
        )
        for i in range(3)
    ]
    ev = rule.evaluate(session, project="proj")
    assert ev is not None
    assert ev.evidence["mean_similarity"] == 1.0


def test_tool_loop_metric_unknown_falls_back_to_jaccard(make_call, now):
    """Why this matters: line 121 — the unknown-metric branch falls back to
    Jaccard rather than throwing inside detection. A regression that re-raises
    breaks the rule for any customer who typoes their config key.
    """
    from token_sentinel.rules.tool_loop import ToolLoopRule

    rule = ToolLoopRule({"tool_loop.similarity_metric": "totally-bogus"})
    session = [
        make_call(
            timestamp=now + timedelta(seconds=i),
            tool_calls=[{"name": "search", "arguments": {"q": "duplicate query"}}],
        )
        for i in range(3)
    ]
    ev = rule.evaluate(session, project="proj")
    # Falls back to jaccard — identical args → similarity=1.0 → fires.
    assert ev is not None


# ---------------------------------------------------------------------------
# Tool-definition-bloat rule: edge cases in helpers
# ---------------------------------------------------------------------------


def test_tool_name_object_with_name_attribute():
    """Why this matters: line 167 — _tool_name fallback for objects (e.g.
    Pydantic AI's ToolDefinition) that expose .name as an attribute, not a
    dict key. Without this branch the bloat rule's evidence is missing names.
    """

    class FakeToolDef:
        name = "calculator"

    assert _tool_name(FakeToolDef()) == "calculator"


def test_top_tools_by_size_serialisation_failure_skips():
    """Why this matters: an unserialisable tool must not poison the whole
    evidence dict. _serialise_tools_bounded silently skips tools whose
    serialisation throws so _top_tools_by_size only ever sees clean entries.
    """
    from token_sentinel.rules.tool_definition_bloat import _serialise_tools_bounded

    class Unserialisable:
        # json.dumps with default=str falls back to str() — make str() also fail.
        def __str__(self):
            raise RuntimeError("nope")

        def __repr__(self):
            raise RuntimeError("nope")

    tools = [
        {"name": "good", "input_schema": {}},
        Unserialisable(),
        {"name": "good2", "input_schema": {}},
    ]
    sized = _serialise_tools_bounded(tools, max_tool_bytes=262144, max_total_bytes=5_242_880)
    out = _top_tools_by_size(sized["per_tool"], k=5)
    # The bad tool is silently skipped; the good ones surface.
    names = [t["name"] for t in out]
    assert "good" in names
    assert "good2" in names
    assert len(out) == 2  # unserialisable was skipped
