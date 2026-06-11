"""Tests for ``token_sentinel.wrappers.voyage.wrap_voyage``.

NO real Voyage AI calls. We construct mocks shaped like a
``voyageai.Client`` / ``voyageai.AsyncClient`` and verify the wrapper:
  - swaps in instrumented ``embed`` / ``rerank``
  - delegates to the originals unchanged (return value, kwargs pass-through)
  - builds a ``CallRecord`` with provider="voyage" and the spec'd shape
  - uses char count as the ``prompt_tokens`` proxy for ``embed`` and
    doc count for ``rerank`` (Voyage's response only exposes an aggregate
    ``total_tokens`` field — there's no per-input breakdown)
  - never crashes the user's call when the tracer or Sentinel misbehaves
  - is dispatched correctly by ``Sentinel.wrap`` (voyageai module recognised)
  - feeds the ``embedding_waste`` rule the duplicate-detection shape it needs

The voyageai SDK is an optional dependency; tests are skipped if it isn't
installed. We mock the SDK surface with ``SimpleNamespace`` so this file
does not pin a specific voyageai release in the test path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

# Voyage SDK is optional. Skip the entire file if it isn't installed —
# the wrapper itself does not import voyageai at module load time, but
# the Sentinel.wrap() dispatch test below constructs a real
# ``voyageai.Client()`` so we need the SDK to exist.
voyageai = pytest.importorskip("voyageai")

from token_sentinel import Sentinel  # noqa: E402
from token_sentinel.events import CallRecord  # noqa: E402
from token_sentinel.wrappers.voyage import (  # noqa: E402
    _build_embed_record,
    _build_rerank_record,
    _char_count,
    _extract_rerank_inputs,
    _extract_texts,
    wrap_voyage,
)

# ---------------------------------------------------------------------------
# Mock voyageai client surface
# ---------------------------------------------------------------------------


class _RecordingCallable:
    """Real callable that records calls and returns a configurable response.

    Mirrors the conftest._RecordingCreate pattern: ``functools.wraps`` (used
    inside ``wrap_voyage``) needs ``__name__``/``__qualname__`` to be plain
    strings, which a MagicMock's auto-generated child mocks fail to provide.
    """

    __name__ = "embed"
    __qualname__ = "Client.embed"
    __module__ = "voyageai.client"
    __annotations__: dict = {}
    __doc__ = "mock voyage method"

    def __init__(self, name: str = "embed"):
        self.__name__ = name
        self.__qualname__ = f"Client.{name}"
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

    def __init__(self, name: str = "embed"):
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


def _make_voyage_client(
    *, async_: bool = False, embed_return: Any = None, rerank_return: Any = None
) -> Any:
    """Build a mock voyageai client.

    Constructs a class whose ``__module__`` is ``voyageai.client`` and
    whose ``__name__`` is ``Client`` (or ``AsyncClient`` for the async
    case) so ``Sentinel.wrap`` routes through the Voyage branch in its
    module-prefix + class-name check.
    """
    cls_name = "AsyncClient" if async_ else "Client"
    fake_cls = type(cls_name, (), {"__module__": "voyageai.client"})
    client = fake_cls()
    if async_:
        embed = _AsyncRecordingCallable(name="embed")
        rerank = _AsyncRecordingCallable(name="rerank")
    else:
        embed = _RecordingCallable(name="embed")
        rerank = _RecordingCallable(name="rerank")
    embed.return_value = embed_return if embed_return is not None else _embedding_response()
    rerank.return_value = rerank_return if rerank_return is not None else _rerank_response()
    client.embed = embed
    client.rerank = rerank
    return client


def _embedding_response(*, embeddings: list[list[float]] | None = None, total_tokens: int = 12):
    """Voyage's ``EmbeddingsObject`` shape: ``embeddings`` + ``total_tokens``."""
    return SimpleNamespace(
        embeddings=embeddings or [[0.1, 0.2, 0.3]],
        total_tokens=total_tokens,
    )


def _rerank_response(*, results: list[Any] | None = None, total_tokens: int = 20):
    """Voyage's ``RerankingObject`` shape: ``results`` + ``total_tokens``."""
    return SimpleNamespace(
        results=results
        or [
            SimpleNamespace(index=0, document="a", relevance_score=0.9),
            SimpleNamespace(index=1, document="b", relevance_score=0.5),
        ],
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# 1. Sentinel.wrap dispatch — voyageai.Client detection
# ---------------------------------------------------------------------------


def test_wrap_voyage_client_detection():
    """``Sentinel.wrap(voyageai.Client())`` returns the wrapped client.

    The real ``voyageai.Client()`` constructor doesn't require an API key
    (it lazily reads ``VOYAGE_API_KEY`` only on the actual API call), so
    we can construct one without secrets.
    """
    s = Sentinel(project="proj")
    real_client = voyageai.Client(api_key="fake-key-not-used")
    original_embed = real_client.embed
    out = s.wrap(real_client)
    assert out is real_client
    # Sentinel mutated embed in-place.
    assert real_client.embed is not original_embed

    # AsyncClient is also recognised.
    async_client = voyageai.AsyncClient(api_key="fake-key-not-used")
    out2 = s.wrap(async_client)
    assert out2 is async_client


# ---------------------------------------------------------------------------
# 2. embed builds a CallRecord
# ---------------------------------------------------------------------------


def test_embed_records_callrecord():
    """A sync embed call produces one CallRecord with provider=voyage."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(
        texts=["hello world", "another"],
        model="voyage-3",
        input_type="query",
        _sentinel_session_id="sess-1",
    )

    records = s.tracer.session("sess-1")
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, CallRecord)
    assert rec.provider == "voyage"
    assert rec.method == "embed"
    assert rec.model == "voyage-3"
    # Char-count proxy: 11 + 7 = 18.
    assert rec.prompt_tokens == 18
    assert rec.completion_tokens == 0
    assert rec.tool_calls == []
    assert rec.user_facing_output is False
    # input_type is captured in raw_request for any future rule that wants it.
    assert rec.raw_request["input_type"] == "query"
    assert rec.raw_request["input_count"] == 2
    # raw_request["input"] mirrors the OpenAI embedding wrapper key so the
    # embedding_waste rule's hash function lands on the same shape.
    assert rec.raw_request["input"] == ["hello world", "another"]


# ---------------------------------------------------------------------------
# 3. AsyncClient.embed builds a CallRecord
# ---------------------------------------------------------------------------


def test_embed_async_records_callrecord():
    """An async ``AsyncClient.embed`` produces one CallRecord."""
    client = _make_voyage_client(async_=True)
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    async def run() -> Any:
        return await client.embed(
            texts=["foo", "bar baz"],
            model="voyage-3-large",
            _sentinel_session_id="async-sess",
        )

    asyncio.run(run())

    records = s.tracer.session("async-sess")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "voyage"
    assert rec.method == "embed"
    assert rec.model == "voyage-3-large"
    # 3 + 7 = 10.
    assert rec.prompt_tokens == 10


# ---------------------------------------------------------------------------
# 4. rerank builds a CallRecord
# ---------------------------------------------------------------------------


def test_rerank_records_callrecord():
    """A rerank call produces a CallRecord with method=rerank."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.rerank(
        query="What is RAG?",
        documents=["doc 1 text", "doc 2 text", "doc 3 text"],
        model="rerank-2",
        top_k=2,
        _sentinel_session_id="rr-1",
    )

    records = s.tracer.session("rr-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "voyage"
    assert rec.method == "rerank"
    assert rec.model == "rerank-2"
    # Doc-count proxy: 3 documents.
    assert rec.prompt_tokens == 3
    assert rec.completion_tokens == 0
    assert rec.raw_request["query"] == "What is RAG?"
    assert rec.raw_request["documents"] == ["doc 1 text", "doc 2 text", "doc 3 text"]
    assert rec.raw_request["top_k"] == 2


# ---------------------------------------------------------------------------
# 5. embedding_waste fires on duplicate Voyage embed
# ---------------------------------------------------------------------------


def test_embedding_waste_fires_on_duplicate_voyage_embed():
    """Two Voyage embed calls with the same texts fire embedding_waste on
    the second call.

    The embedding_waste rule (``rules/embedding_waste.py``) accepts both
    ``method.endswith("embeddings.create")`` (OpenAI) and ``method == "embed"``
    (Voyage) — see the rule's docstring for the full set. The wrapper
    produces ``raw_request["input"]`` keyed on the texts list, which is the
    shape the rule's hash function expects.
    """
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    captured_events: list[Any] = []

    @s.on_leak
    def collect(event):
        captured_events.append(event)

    # Same texts twice in the same session.
    for _ in range(2):
        client.embed(
            texts=["alpha", "beta"],
            model="voyage-3",
            _sentinel_session_id="dup-sess",
        )

    assert any(ev.type == "embedding_waste" for ev in captured_events), (
        f"expected embedding_waste, got types: {[e.type for e in captured_events]}"
    )
    waste = next(ev for ev in captured_events if ev.type == "embedding_waste")
    assert waste.evidence["model"] == "voyage-3"
    assert waste.evidence["duplicate_count"] == 2


# ---------------------------------------------------------------------------
# 6. Failure in instrumentation does not break the user's call
# ---------------------------------------------------------------------------


def test_failure_in_instrumentation_does_not_break_user_call(monkeypatch):
    """A bug inside ``Sentinel.record_call`` must not crash the user's call."""
    # Configure the recording callable BEFORE wrapping so the return value
    # is captured by the original (wrapped) closure -- once wrap_voyage runs,
    # ``client.embed`` is the instrumented function, not the original
    # ``_RecordingCallable`` (so writing ``.return_value`` after the wrap
    # would land on the wrong object).
    sentinel_response = _embedding_response()
    client = _make_voyage_client(embed_return=sentinel_response)
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    # Replace record_call with a function that raises a generic Exception.
    def boom(*_args, **_kwargs):
        raise RuntimeError("instrumentation bug")

    monkeypatch.setattr(s, "record_call", boom)

    # The user's call returns the underlying SDK response unchanged.
    out = client.embed(texts=["x"], model="voyage-3")
    assert out is sentinel_response


# ---------------------------------------------------------------------------
# 7. CallRecord carries provider="voyage"
# ---------------------------------------------------------------------------


def test_voyage_emit_includes_provider_field():
    """Every record emitted by the Voyage wrapper carries provider=voyage."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(texts=["x"], model="voyage-3", _sentinel_session_id="p-1")
    client.rerank(
        query="q",
        documents=["d"],
        model="rerank-2",
        _sentinel_session_id="p-2",
    )
    embed_rec = s.tracer.session("p-1")[0]
    rerank_rec = s.tracer.session("p-2")[0]
    assert embed_rec.provider == "voyage"
    assert rerank_rec.provider == "voyage"


# ---------------------------------------------------------------------------
# 8. Model is passed through
# ---------------------------------------------------------------------------


def test_voyage_model_passed_through():
    """The model kwarg lands in CallRecord.model verbatim."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    for model in ["voyage-3", "voyage-3-large", "voyage-code-2", "rerank-2"]:
        client.embed(texts=["x"], model=model, _sentinel_session_id=f"m-{model}")
        rec = s.tracer.session(f"m-{model}")[0]
        assert rec.model == model


# ---------------------------------------------------------------------------
# 9. embed: char count is the prompt_tokens proxy
# ---------------------------------------------------------------------------


def test_voyage_char_count_is_prompt_tokens_proxy():
    """``prompt_tokens`` on embed records equals total chars across inputs.

    Voyage's response carries ``total_tokens`` as an aggregate (no per-input
    breakdown), so we proxy by char count for stability — see the wrapper's
    module docstring.
    """
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    texts = ["short", "a longer string", "x"]
    expected = sum(len(t) for t in texts)
    client.embed(texts=texts, model="voyage-3", _sentinel_session_id="cc")
    rec = s.tracer.session("cc")[0]
    assert rec.prompt_tokens == expected
    # Helper exposes the same number for direct rule authoring.
    assert _char_count(texts) == expected


# ---------------------------------------------------------------------------
# 10. rerank: doc count is the prompt_tokens proxy
# ---------------------------------------------------------------------------


def test_voyage_rerank_doc_count_is_prompt_tokens_proxy():
    """``prompt_tokens`` on rerank records equals the number of documents."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    docs = ["d1", "d2", "d3", "d4", "d5"]
    client.rerank(
        query="q",
        documents=docs,
        model="rerank-2",
        _sentinel_session_id="dc",
    )
    rec = s.tracer.session("dc")[0]
    assert rec.prompt_tokens == len(docs)


# ---------------------------------------------------------------------------
# 11. Latency is captured
# ---------------------------------------------------------------------------


def test_voyage_latency_captured():
    """CallRecord.latency_ms is a positive float (perf_counter-based)."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(texts=["x"], model="voyage-3", _sentinel_session_id="lat")
    rec = s.tracer.session("lat")[0]
    assert isinstance(rec.latency_ms, float)
    assert rec.latency_ms >= 0.0
    # A no-op recording callable should take well under a second.
    assert rec.latency_ms < 1000.0


# ---------------------------------------------------------------------------
# 12. Empty texts list is handled
# ---------------------------------------------------------------------------


def test_voyage_handles_empty_texts_list():
    """An ``embed(texts=[])`` call produces a CallRecord with prompt_tokens=0
    rather than crashing the wrapper."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(texts=[], model="voyage-3", _sentinel_session_id="empty")
    records = s.tracer.session("empty")
    assert len(records) == 1
    rec = records[0]
    assert rec.prompt_tokens == 0
    assert rec.raw_request["input_count"] == 0
    assert rec.raw_request["input"] == []


# ---------------------------------------------------------------------------
# 13. Single text input is handled
# ---------------------------------------------------------------------------


def test_voyage_handles_single_text():
    """A single-text embed call records correctly."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(
        texts=["just one"],
        model="voyage-3",
        _sentinel_session_id="single",
    )
    rec = s.tracer.session("single")[0]
    assert rec.prompt_tokens == len("just one")
    assert rec.raw_request["input_count"] == 1
    assert rec.raw_request["input"] == ["just one"]


# ---------------------------------------------------------------------------
# 14. Batch of 100 inputs is handled
# ---------------------------------------------------------------------------


def test_voyage_handles_batch_of_100():
    """A 100-text batch records correctly — exercises the loop in
    ``_char_count`` / ``_extract_texts``."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    texts = [f"item-{i}" for i in range(100)]
    expected = sum(len(t) for t in texts)
    client.embed(
        texts=texts,
        model="voyage-3",
        _sentinel_session_id="batch100",
    )
    rec = s.tracer.session("batch100")[0]
    assert rec.prompt_tokens == expected
    assert rec.raw_request["input_count"] == 100


# ---------------------------------------------------------------------------
# 15. user_facing_output is always False on Voyage records
# ---------------------------------------------------------------------------


def test_voyage_user_facing_output_false():
    """Embeddings and reranks are intermediate retrieval steps — they are
    never themselves the user's final output."""
    client = _make_voyage_client()
    s = Sentinel(project="proj")
    wrap_voyage(client, s)

    client.embed(texts=["x"], model="voyage-3", _sentinel_session_id="u-1")
    client.rerank(query="q", documents=["d"], model="rerank-2", _sentinel_session_id="u-2")
    for sess in ("u-1", "u-2"):
        rec = s.tracer.session(sess)[0]
        assert rec.user_facing_output is False


# ---------------------------------------------------------------------------
# Extra sanity checks for the small extractor helpers — not counted in the
# 15 but cheap insurance for the kwargs-vs-positional code path the wrapper
# uses to read ``texts`` / ``query`` / ``documents``.
# ---------------------------------------------------------------------------


def test_extract_texts_positional_argument():
    """``embed("solo", model=...)`` — positional texts must still be picked up."""
    # _extract_texts pulls texts from args[0] when kwargs.get("texts") is None.
    assert _extract_texts(("solo",), {}) == ["solo"]
    # A list passed positionally.
    assert _extract_texts((["a", "b"],), {}) == ["a", "b"]
    # kwargs wins over positional.
    assert _extract_texts((["pos"],), {"texts": ["kw"]}) == ["kw"]


def test_extract_rerank_inputs_positional():
    """``rerank("query", ["d1", "d2"], model="...")`` — positional fallback."""
    q, docs = _extract_rerank_inputs(("q1", ["d1", "d2"]), {})
    assert q == "q1"
    assert docs == ["d1", "d2"]


def test_build_embed_record_unknown_model():
    """Missing ``model`` kwarg yields ``model='unknown'`` rather than crashing."""
    rec = _build_embed_record(
        session_id="s",
        args=(),
        kwargs={"texts": ["x"]},
        latency_ms=1.0,
    )
    assert rec.model == "unknown"


def test_build_rerank_record_no_top_k():
    """rerank without top_k still produces a CallRecord with top_k=None."""
    rec = _build_rerank_record(
        session_id="s",
        args=(),
        kwargs={"query": "q", "documents": ["d"], "model": "rerank-2"},
        latency_ms=1.0,
    )
    assert rec.raw_request["top_k"] is None
