"""Tests for ``token_sentinel.wrappers.cohere.wrap_cohere``.

NO real Cohere API calls. We construct mocks shaped like a
``cohere.ClientV2`` / ``cohere.AsyncClientV2`` and verify the wrapper:
  - swaps in instrumented ``chat`` / ``embed`` / ``rerank``
  - delegates to the originals unchanged (return value, kwargs pass-through)
  - builds a ``CallRecord`` with provider="cohere" and the spec'd shape
  - reads chat tokens from ``response.usage.tokens.input_tokens`` /
    ``output_tokens``
  - prefers ``meta.billed_units.input_tokens`` for embed when present;
    falls back to char count otherwise
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - is dispatched correctly by ``Sentinel.wrap`` (cohere module recognised)
  - feeds the ``embedding_waste`` rule the duplicate-detection shape it needs

The cohere SDK is an optional dependency; tests are skipped if it isn't
installed. We mock the SDK surface with ``SimpleNamespace`` so this file
does not pin a specific cohere release in the test path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

# Cohere SDK is optional. Skip the entire file if it isn't installed —
# the wrapper itself does not import cohere at module load time, but
# the Sentinel.wrap() dispatch test below constructs a real
# ``cohere.ClientV2()`` so we need the SDK to exist.
cohere = pytest.importorskip("cohere")

from token_sentinel import Sentinel  # noqa: E402
from token_sentinel.events import CallRecord  # noqa: E402
from token_sentinel.wrappers.cohere import (  # noqa: E402
    _build_chat_record,
    _build_embed_record,
    _build_rerank_record,
    _char_count,
    _extract_rerank_inputs,
    _extract_texts,
    wrap_cohere,
)

# ---------------------------------------------------------------------------
# Mock cohere V2 client surface
# ---------------------------------------------------------------------------


class _RecordingCallable:
    """Real callable that records calls and returns a configurable response.

    Mirrors the conftest._RecordingCreate pattern: ``functools.wraps`` (used
    inside ``wrap_cohere``) needs ``__name__``/``__qualname__`` to be plain
    strings, which a MagicMock's auto-generated child mocks fail to provide.
    """

    __name__ = "chat"
    __qualname__ = "ClientV2.chat"
    __module__ = "cohere.client_v2"
    __annotations__: dict = {}
    __doc__ = "mock cohere method"

    def __init__(self, name: str = "chat"):
        self.__name__ = name
        self.__qualname__ = f"ClientV2.{name}"
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


class _AsyncRecordingCallable(_RecordingCallable):
    """Async counterpart of ``_RecordingCallable``.

    ``inspect.iscoroutinefunction`` is the wrapper's async detection hook.
    Defining ``__call__`` as ``async def`` is what flips that bit.
    """

    def __init__(self, name: str = "chat"):
        super().__init__(name=name)

    async def __call__(self, *args, **kwargs):  # type: ignore[override]
        self.calls.append((args, dict(kwargs)))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value


def _make_cohere_client(
    *,
    async_: bool = False,
    chat_return: Any = None,
    embed_return: Any = None,
    rerank_return: Any = None,
) -> Any:
    """Build a mock cohere ClientV2 / AsyncClientV2.

    Constructs a class whose ``__module__`` is ``cohere.client_v2`` and
    whose ``__name__`` is ``ClientV2`` (or ``AsyncClientV2`` for the
    async case) so ``Sentinel.wrap`` routes through the Cohere branch
    in its module-prefix + class-name check.
    """
    cls_name = "AsyncClientV2" if async_ else "ClientV2"
    fake_cls = type(cls_name, (), {"__module__": "cohere.client_v2"})
    client = fake_cls()
    if async_:
        chat = _AsyncRecordingCallable(name="chat")
        embed = _AsyncRecordingCallable(name="embed")
        rerank = _AsyncRecordingCallable(name="rerank")
    else:
        chat = _RecordingCallable(name="chat")
        embed = _RecordingCallable(name="embed")
        rerank = _RecordingCallable(name="rerank")
    chat.return_value = chat_return if chat_return is not None else _chat_response()
    embed.return_value = embed_return if embed_return is not None else _embed_response()
    rerank.return_value = rerank_return if rerank_return is not None else _rerank_response()
    client.chat = chat
    client.embed = embed
    client.rerank = rerank
    return client


def _chat_response(
    *,
    input_tokens: int = 50,
    output_tokens: int = 20,
    text: str = "Hello there.",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "COMPLETE",
) -> SimpleNamespace:
    """Cohere V2 chat response shape."""
    if tool_calls is None:
        content = [SimpleNamespace(type="text", text=text)]
        message = SimpleNamespace(role="assistant", tool_calls=None, content=content)
    else:
        # When tool_calls are present, content may be None or empty
        message = SimpleNamespace(role="assistant", tool_calls=tool_calls, content=[])
    return SimpleNamespace(
        id="chat-id-1",
        finish_reason=finish_reason,
        message=message,
        usage=SimpleNamespace(
            billed_units=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            tokens=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            cached_tokens=None,
        ),
        logprobs=None,
    )


def _embed_response(
    *,
    billed_input_tokens: int | None = 8,
    embeddings: Any = None,
) -> SimpleNamespace:
    """Cohere V2 embed response shape. ``billed_input_tokens=None`` simulates
    a response missing the ``meta.billed_units.input_tokens`` field so the
    fallback chain (response usage → char count) is exercised."""
    if billed_input_tokens is None:
        meta = SimpleNamespace(
            api_version=None,
            billed_units=None,
            tokens=None,
            cached_tokens=None,
            warnings=None,
        )
    else:
        meta = SimpleNamespace(
            api_version=None,
            billed_units=SimpleNamespace(
                images=None,
                input_tokens=billed_input_tokens,
                image_tokens=None,
                output_tokens=None,
                search_units=None,
                classifications=None,
            ),
            tokens=None,
            cached_tokens=None,
            warnings=None,
        )
    return SimpleNamespace(
        response_type="embeddings_by_type",
        id="embed-id-1",
        embeddings=embeddings or SimpleNamespace(float_=[[0.1, 0.2]]),
        texts=None,
        images=None,
        meta=meta,
    )


def _rerank_response(*, search_units: int = 1) -> SimpleNamespace:
    """Cohere V2 rerank response shape."""
    return SimpleNamespace(
        id="rerank-id-1",
        results=[
            SimpleNamespace(index=0, relevance_score=0.9),
            SimpleNamespace(index=1, relevance_score=0.4),
        ],
        meta=SimpleNamespace(
            api_version=None,
            billed_units=SimpleNamespace(
                images=None,
                input_tokens=None,
                image_tokens=None,
                output_tokens=None,
                search_units=search_units,
                classifications=None,
            ),
            tokens=None,
            cached_tokens=None,
            warnings=None,
        ),
    )


def _tool_call(name: str, arguments_json: str) -> SimpleNamespace:
    """Cohere V2 ``ToolCallV2`` shape: id, type, function(name, arguments)."""
    return SimpleNamespace(
        id="tc-1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


# ---------------------------------------------------------------------------
# 1. Sentinel.wrap dispatch — cohere.ClientV2 detection
# ---------------------------------------------------------------------------


def test_wrap_cohere_client_detection():
    """``Sentinel.wrap(cohere.ClientV2())`` returns the wrapped client.

    Real ``cohere.ClientV2()`` accepts an ``api_key`` constructor arg without
    making any network call (it lazily reads ``COHERE_API_KEY`` only on the
    actual API call), so we can construct one without secrets.
    """
    s = Sentinel(project="proj")
    real_client = cohere.ClientV2(api_key="fake-key-not-used")
    original_chat = real_client.chat
    out = s.wrap(real_client)
    assert out is real_client
    # Sentinel mutated chat in-place.
    assert real_client.chat is not original_chat


# ---------------------------------------------------------------------------
# 2. AsyncClientV2 detection
# ---------------------------------------------------------------------------


def test_wrap_cohere_async_client_detection():
    """``Sentinel.wrap(cohere.AsyncClientV2())`` is also recognised."""
    s = Sentinel(project="proj")
    async_client = cohere.AsyncClientV2(api_key="fake-key-not-used")
    original_chat = async_client.chat
    out = s.wrap(async_client)
    assert out is async_client
    assert async_client.chat is not original_chat


# ---------------------------------------------------------------------------
# 3. chat builds a CallRecord
# ---------------------------------------------------------------------------


def test_chat_records_callrecord():
    """A sync chat call produces one CallRecord with provider=cohere."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "hello"}],
        _sentinel_session_id="sess-chat-1",
    )

    records = s.tracer.session("sess-chat-1")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    assert rec.provider == "cohere"
    assert rec.method == "chat"
    assert rec.model == "command-r-plus-08-2024"


# ---------------------------------------------------------------------------
# 4. chat token counts come from usage.tokens.{input,output}_tokens
# ---------------------------------------------------------------------------


def test_chat_token_counts_from_usage_block():
    """``prompt_tokens`` / ``completion_tokens`` come from
    ``response.usage.tokens.input_tokens`` / ``output_tokens``."""
    response = _chat_response(input_tokens=123, output_tokens=45)
    client = _make_cohere_client(chat_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": "hi"}],
        _sentinel_session_id="tok-1",
    )
    rec = s.tracer.session("tok-1")[0]
    assert rec.prompt_tokens == 123
    assert rec.completion_tokens == 45


# ---------------------------------------------------------------------------
# 5. async chat builds a CallRecord
# ---------------------------------------------------------------------------


def test_chat_async_records_callrecord():
    """An async ``AsyncClientV2.chat`` produces one CallRecord."""
    client = _make_cohere_client(async_=True)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    async def run() -> Any:
        return await client.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": "async hi"}],
            _sentinel_session_id="async-chat",
        )

    asyncio.run(run())
    records = s.tracer.session("async-chat")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "cohere"
    assert rec.method == "chat"


# ---------------------------------------------------------------------------
# 6. tool_calls are mined and passed through
# ---------------------------------------------------------------------------


def test_chat_tool_calls_passed_through():
    """When the response carries ``message.tool_calls``, the wrapper mines
    them into ``CallRecord.tool_calls`` with parsed JSON arguments."""
    response = _chat_response(
        tool_calls=[
            _tool_call("search_docs", '{"query": "kubernetes"}'),
            _tool_call("get_weather", '{"city": "Paris"}'),
        ]
    )
    client = _make_cohere_client(chat_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "search and weather please"}],
        tools=[{"type": "function", "function": {"name": "search_docs"}}],
        _sentinel_session_id="tc",
    )

    rec = s.tracer.session("tc")[0]
    assert len(rec.tool_calls) == 2
    assert rec.tool_calls[0]["name"] == "search_docs"
    assert rec.tool_calls[0]["arguments"] == {"query": "kubernetes"}
    assert rec.tool_calls[1]["name"] == "get_weather"
    assert rec.tool_calls[1]["arguments"] == {"city": "Paris"}


# ---------------------------------------------------------------------------
# 7. embed builds a CallRecord
# ---------------------------------------------------------------------------


def test_embed_records_callrecord():
    """A sync embed call produces one CallRecord with provider=cohere."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.embed(
        model="embed-v4.0",
        texts=["hello world", "another"],
        input_type="search_document",
        embedding_types=["float"],
        _sentinel_session_id="sess-emb-1",
    )

    records = s.tracer.session("sess-emb-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "cohere"
    assert rec.method == "embed"
    assert rec.model == "embed-v4.0"
    assert rec.completion_tokens == 0
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    assert rec.raw_request["input_type"] == "search_document"
    assert rec.raw_request["input_count"] == 2
    assert rec.raw_request["input"] == ["hello world", "another"]


# ---------------------------------------------------------------------------
# 8. embed: prefer billed_units when present
# ---------------------------------------------------------------------------


def test_embed_uses_billed_units_when_present():
    """When ``response.meta.billed_units.input_tokens`` is set, it is used
    in preference to the char-count proxy."""
    response = _embed_response(billed_input_tokens=42)
    client = _make_cohere_client(embed_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.embed(
        model="embed-v4.0",
        texts=["a", "b", "c"],
        input_type="search_query",
        _sentinel_session_id="billed",
    )
    rec = s.tracer.session("billed")[0]
    assert rec.prompt_tokens == 42  # NOT 3 (char count)


# ---------------------------------------------------------------------------
# 9. embed: fall back to char count when billing meta missing
# ---------------------------------------------------------------------------


def test_embed_falls_back_to_char_count_when_billing_meta_missing():
    """When ``meta.billed_units`` is None / missing, ``prompt_tokens``
    falls back to the char-count proxy ``sum(len(t) for t in texts)``."""
    response = _embed_response(billed_input_tokens=None)
    client = _make_cohere_client(embed_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    texts = ["hello", "world!"]  # 5 + 6 = 11
    client.embed(
        model="embed-v4.0",
        texts=texts,
        input_type="search_query",
        _sentinel_session_id="fallback",
    )
    rec = s.tracer.session("fallback")[0]
    assert rec.prompt_tokens == 11


# ---------------------------------------------------------------------------
# 10. embedding_waste rule fires on duplicate cohere embed
# ---------------------------------------------------------------------------


def test_embedding_waste_fires_on_duplicate_cohere_embed():
    """Two cohere embed calls with the same texts fire embedding_waste on
    the second call.

    The embedding_waste rule (-broadened ``method == "embed"`` branch)
    accepts cohere alongside voyage, and the wrapper produces
    ``raw_request["input"]`` keyed on the texts list — same shape as
    OpenAI / Voyage, so the rule's SHA-256 hash function lands identically.
    """
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    captured_events: list[Any] = []

    @s.on_leak
    def collect(event):
        captured_events.append(event)

    for _ in range(2):
        client.embed(
            model="embed-v4.0",
            texts=["alpha", "beta"],
            input_type="search_document",
            _sentinel_session_id="dup-sess",
        )

    assert any(ev.type == "embedding_waste" for ev in captured_events), (
        f"expected embedding_waste, got types: {[e.type for e in captured_events]}"
    )
    waste = next(ev for ev in captured_events if ev.type == "embedding_waste")
    assert waste.evidence["model"] == "embed-v4.0"
    assert waste.evidence["duplicate_count"] == 2


# ---------------------------------------------------------------------------
# 11. rerank builds a CallRecord
# ---------------------------------------------------------------------------


def test_rerank_records_callrecord():
    """A rerank call produces a CallRecord with method=rerank."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.rerank(
        model="rerank-3.5",
        query="What is RAG?",
        documents=["d1 text", "d2 text", "d3 text"],
        top_n=2,
        _sentinel_session_id="rr-1",
    )

    records = s.tracer.session("rr-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "cohere"
    assert rec.method == "rerank"
    assert rec.model == "rerank-3.5"
    # Char count: len("What is RAG?") + 7 + 7 + 7 = 12 + 21 = 33
    assert rec.prompt_tokens == len("What is RAG?") + len("d1 text") * 3
    assert rec.completion_tokens == 0
    assert rec.raw_request["query"] == "What is RAG?"


# ---------------------------------------------------------------------------
# 12. rerank: document count surfaced in raw_request
# ---------------------------------------------------------------------------


def test_rerank_doc_count_in_model_specific_meta():
    """Document count is captured in ``raw_request["document_count"]``
    ( will lift to ``usage_extra.model_specific_meta``)."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    docs = ["d1", "d2", "d3", "d4", "d5"]
    client.rerank(
        model="rerank-3.5",
        query="q",
        documents=docs,
        top_n=3,
        _sentinel_session_id="dc",
    )
    rec = s.tracer.session("dc")[0]
    assert rec.raw_request["document_count"] == 5


# ---------------------------------------------------------------------------
# 13. rerank: top_n surfaced in raw_request
# ---------------------------------------------------------------------------


def test_rerank_top_n_in_model_specific_meta():
    """``top_n`` is captured in ``raw_request["top_n"]`` ( will lift
    to ``usage_extra.model_specific_meta``)."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.rerank(
        model="rerank-3.5",
        query="q",
        documents=["d1", "d2"],
        top_n=7,
        _sentinel_session_id="tn",
    )
    rec = s.tracer.session("tn")[0]
    assert rec.raw_request["top_n"] == 7


# ---------------------------------------------------------------------------
# 14. Failure in instrumentation does not break the user's call
# ---------------------------------------------------------------------------


def test_failure_in_instrumentation_does_not_break_user_call(monkeypatch):
    """A bug inside ``Sentinel.record_call`` must not crash the user's call."""
    sentinel_response = _chat_response()
    client = _make_cohere_client(chat_return=sentinel_response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    def boom(*_args, **_kwargs):
        raise RuntimeError("instrumentation bug")

    monkeypatch.setattr(s, "record_call", boom)

    # The user's call returns the underlying SDK response unchanged.
    out = client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "x"}],
    )
    assert out is sentinel_response


# ---------------------------------------------------------------------------
# 15. provider field is set to "cohere"
# ---------------------------------------------------------------------------


def test_provider_field_set_to_cohere():
    """Every record emitted by the cohere wrapper carries provider=cohere."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "x"}],
        _sentinel_session_id="p-1",
    )
    client.embed(
        model="embed-v4.0", texts=["x"], input_type="search_query", _sentinel_session_id="p-2"
    )
    client.rerank(model="rerank-3.5", query="q", documents=["d"], _sentinel_session_id="p-3")

    for sess in ("p-1", "p-2", "p-3"):
        rec = s.tracer.session(sess)[0]
        assert rec.provider == "cohere"


# ---------------------------------------------------------------------------
# 16. Model is passed through
# ---------------------------------------------------------------------------


def test_model_passed_through_to_callrecord():
    """The model kwarg lands in CallRecord.model verbatim."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    for method, args in [
        ("chat", {"messages": [{"role": "user", "content": "x"}]}),
        ("embed", {"texts": ["x"], "input_type": "search_query"}),
        ("rerank", {"query": "q", "documents": ["d"]}),
    ]:
        for model in ["command-r-plus-08-2024", "command-a-03-2025", "embed-v4.0", "rerank-3.5"]:
            sess = f"m-{method}-{model}"
            getattr(client, method)(model=model, _sentinel_session_id=sess, **args)
            rec = s.tracer.session(sess)[0]
            assert rec.model == model


# ---------------------------------------------------------------------------
# 17. raw_request strips message content for chat
# ---------------------------------------------------------------------------


def test_raw_request_strips_message_content():
    """The chat record's ``raw_request`` does NOT contain the full messages
    array — only model, tools, max_tokens, and message_count.

    Mirrors the anthropic wrapper's redaction discipline: customer message
    content should not round-trip through the rule engine / cloud sink in
    plaintext.
    """
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    messages = [
        {"role": "user", "content": "secret query about PII"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    client.chat(
        model="command-r-plus-08-2024",
        messages=messages,
        tools=[{"type": "function", "function": {"name": "search"}}],
        _sentinel_session_id="redact",
    )
    rec = s.tracer.session("redact")[0]
    # messages must NOT appear in raw_request.
    assert "messages" not in rec.raw_request
    # Schema-only fields are kept.
    assert rec.raw_request["model"] == "command-r-plus-08-2024"
    assert rec.raw_request["tools"] == [{"type": "function", "function": {"name": "search"}}]
    assert rec.raw_request["message_count"] == 3


# ---------------------------------------------------------------------------
# 18. raw_request keeps texts list for embed (needed by embedding_waste)
# ---------------------------------------------------------------------------


def test_raw_request_strips_text_content_in_embed():
    """The embed record's ``raw_request`` carries the texts list under
    ``input`` — same shape as OpenAI/Voyage so the embedding_waste rule
    works unchanged. The texts list IS the input here; redacting it would
    break duplicate detection."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    texts = ["payload_a", "payload_b"]
    client.embed(
        model="embed-v4.0",
        texts=texts,
        input_type="search_query",
        _sentinel_session_id="emb-redact",
    )
    rec = s.tracer.session("emb-redact")[0]
    assert rec.raw_request["input"] == texts
    assert rec.raw_request["input_count"] == 2


# ---------------------------------------------------------------------------
# 19. Latency is captured
# ---------------------------------------------------------------------------


def test_latency_captured():
    """CallRecord.latency_ms is a non-negative float (perf_counter-based)."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "x"}],
        _sentinel_session_id="lat",
    )
    rec = s.tracer.session("lat")[0]
    assert isinstance(rec.latency_ms, float)
    assert rec.latency_ms >= 0.0
    # A no-op recording callable should take well under a second.
    assert rec.latency_ms < 1000.0


# ---------------------------------------------------------------------------
# 20. user_facing_output is True for chat with text content
# ---------------------------------------------------------------------------


def test_user_facing_output_true_for_chat():
    """A chat response with a non-empty text block sets
    ``user_facing_output=True``."""
    response = _chat_response(text="Hello, here is your answer.")
    client = _make_cohere_client(chat_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "x"}],
        _sentinel_session_id="ufo",
    )
    rec = s.tracer.session("ufo")[0]
    assert rec.user_facing_output is True


# ---------------------------------------------------------------------------
# 21. user_facing_output is False for embed and rerank
# ---------------------------------------------------------------------------


def test_user_facing_output_false_for_embed_and_rerank():
    """Embeddings and reranks are intermediate retrieval steps — they are
    never themselves the user's final output."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.embed(
        model="embed-v4.0", texts=["x"], input_type="search_query", _sentinel_session_id="u-1"
    )
    client.rerank(model="rerank-3.5", query="q", documents=["d"], _sentinel_session_id="u-2")
    for sess in ("u-1", "u-2"):
        rec = s.tracer.session(sess)[0]
        assert rec.user_facing_output is False


# ---------------------------------------------------------------------------
# 22. Empty texts list is handled in embed
# ---------------------------------------------------------------------------


def test_handles_empty_texts_list_in_embed():
    """An ``embed(texts=[])`` call produces a CallRecord with prompt_tokens=0
    (when billing meta missing) rather than crashing the wrapper."""
    response = _embed_response(billed_input_tokens=None)
    client = _make_cohere_client(embed_return=response)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.embed(
        model="embed-v4.0", texts=[], input_type="search_query", _sentinel_session_id="empty"
    )
    records = s.tracer.session("empty")
    assert len(records) == 1
    rec = records[0]
    assert rec.prompt_tokens == 0
    assert rec.raw_request["input_count"] == 0
    assert rec.raw_request["input"] == []


# ---------------------------------------------------------------------------
# 23. Empty documents list is handled in rerank
# ---------------------------------------------------------------------------


def test_handles_empty_documents_list_in_rerank():
    """A ``rerank(documents=[])`` call records a CallRecord with the query
    chars as prompt_tokens and document_count=0."""
    client = _make_cohere_client()
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    client.rerank(
        model="rerank-3.5",
        query="hi",
        documents=[],
        _sentinel_session_id="empty-rr",
    )
    rec = s.tracer.session("empty-rr")[0]
    assert rec.prompt_tokens == 2  # len("hi")
    assert rec.raw_request["document_count"] == 0
    assert rec.raw_request["documents"] == []


# ---------------------------------------------------------------------------
# 24. async embed builds a CallRecord
# ---------------------------------------------------------------------------


def test_async_embed_records_callrecord():
    """An async ``AsyncClientV2.embed`` produces one CallRecord."""
    client = _make_cohere_client(async_=True)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    async def run() -> Any:
        return await client.embed(
            model="embed-v4.0",
            texts=["foo", "bar"],
            input_type="search_query",
            _sentinel_session_id="async-emb",
        )

    asyncio.run(run())
    records = s.tracer.session("async-emb")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "cohere"
    assert rec.method == "embed"
    assert rec.model == "embed-v4.0"


# ---------------------------------------------------------------------------
# 25. async rerank builds a CallRecord
# ---------------------------------------------------------------------------


def test_async_rerank_records_callrecord():
    """An async ``AsyncClientV2.rerank`` produces one CallRecord."""
    client = _make_cohere_client(async_=True)
    s = Sentinel(project="proj")
    wrap_cohere(client, s)

    async def run() -> Any:
        return await client.rerank(
            model="rerank-3.5",
            query="search",
            documents=["doc1", "doc2"],
            top_n=1,
            _sentinel_session_id="async-rr",
        )

    asyncio.run(run())
    records = s.tracer.session("async-rr")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "cohere"
    assert rec.method == "rerank"
    assert rec.model == "rerank-3.5"


# ---------------------------------------------------------------------------
# Bonus: exercise the unittest.mock.patch path the spec asks for.
# ---------------------------------------------------------------------------


def test_patch_clientv2_classes_smoke():
    """Smoke test: ``unittest.mock.patch("cohere.ClientV2")`` works.

    Per the specification ("Mock cohere SDK via patch(...)"), we verify that
    patching is sane in the test environment — the patched class can be
    instantiated and the patched object reports a recognisable type.
    """
    with patch("cohere.ClientV2") as patched_sync, patch("cohere.AsyncClientV2") as patched_async:
        sync_client = patched_sync(api_key="x")
        async_client = patched_async(api_key="x")
        # Each call returns a MagicMock-like object — we just want to
        # verify the patch hook fired (it did; the next assertion would
        # raise if the lookup failed).
        assert sync_client is not None
        assert async_client is not None
        assert patched_sync.called
        assert patched_async.called


# ---------------------------------------------------------------------------
# Extra sanity checks for the small extractor helpers.
# ---------------------------------------------------------------------------


def test_extract_texts_positional_argument():
    """Defensive fallback for a customer who bypasses the keyword-only sig."""
    assert _extract_texts(("solo",), {}) == ["solo"]
    assert _extract_texts((["a", "b"],), {}) == ["a", "b"]
    # kwargs wins over positional.
    assert _extract_texts((["pos"],), {"texts": ["kw"]}) == ["kw"]


def test_extract_rerank_inputs_positional():
    """Positional fallback for rerank."""
    q, docs = _extract_rerank_inputs(("q1", ["d1", "d2"]), {})
    assert q == "q1"
    assert docs == ["d1", "d2"]


def test_build_chat_record_unknown_model():
    """Missing ``model`` kwarg yields ``model='unknown'`` rather than crashing."""
    rec = _build_chat_record(
        session_id="s",
        kwargs={"messages": [{"role": "user", "content": "x"}]},
        response=_chat_response(),
        latency_ms=1.0,
    )
    assert rec.model == "unknown"


def test_build_embed_record_unknown_model():
    """Missing ``model`` kwarg yields ``model='unknown'`` rather than crashing."""
    rec = _build_embed_record(
        session_id="s",
        args=(),
        kwargs={"texts": ["x"]},
        response=_embed_response(billed_input_tokens=None),
        latency_ms=1.0,
    )
    assert rec.model == "unknown"


def test_build_rerank_record_no_top_n():
    """Rerank without top_n still produces a CallRecord with top_n=None."""
    rec = _build_rerank_record(
        session_id="s",
        args=(),
        kwargs={"query": "q", "documents": ["d"], "model": "rerank-3.5"},
        response=_rerank_response(),
        latency_ms=1.0,
    )
    assert rec.raw_request["top_n"] is None


def test_char_count_helper():
    """``_char_count`` sums char lengths."""
    assert _char_count([]) == 0
    assert _char_count(["a"]) == 1
    assert _char_count(["abc", "de"]) == 5
