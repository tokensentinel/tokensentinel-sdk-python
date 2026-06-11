"""Wrap a Voyage AI client to capture call records.

Pattern: in-place mutation of ``client.embed`` and ``client.rerank``. The
original is captured in a closure so the instrumented version can delegate.
This preserves all type hints in IDEs because we mutate the live instance,
not a subclass.

Reference pattern: see ``wrappers/anthropic.py`` for the two-level safety
boundary (record-build errors swallowed; ``record_call`` errors swallowed
EXCEPT ``LeakDetected``, which must propagate so block mode works).

Four cases handled by ``wrap_voyage``:
  1. Sync embed:  ``voyageai.Client().embed(texts=..., model=...)``
  2. Sync rerank: ``voyageai.Client().rerank(query=..., documents=..., model=...)``
  3. Async embed:  ``voyageai.AsyncClient().embed(texts=..., model=...)``
  4. Async rerank: ``voyageai.AsyncClient().rerank(query=..., documents=..., model=...)``

Note on method names: voyageai's ``AsyncClient`` keeps the same method names
as ``Client`` (``embed`` / ``rerank``) — it does NOT use the ``a``-prefix
convention (``aembed`` / ``arerank``) some other async SDKs adopt. We detect
async via ``inspect.iscoroutinefunction`` on the original method, mirroring
how ``wrappers/openai.py`` distinguishes sync vs async.

Token accounting — char-count proxy:
    Voyage's responses expose a ``total_tokens`` field on the
    ``EmbeddingsObject`` / ``RerankingObject``, but only as a single
    aggregate — NOT per-input. The embedding_waste rule and any other
    rule that wants to compare token counts at the call-record level
    benefits from a stable per-call number that's computable BEFORE the
    response arrives (so failure paths still produce a sensible record).
    Per the specification, we use ``sum(len(t) for t in texts)`` (for embed)
    or ``len(query) + sum(len(d) for d in documents)`` (for rerank) as
    a char-count proxy for ``prompt_tokens``. The ratio of chars to
    tokens varies by model and language; the rule engine treats this
    proxy as an ordinal signal (duplicate hash match, growth over time)
    rather than an absolute, so under-counting by ~4× compared to the
    real token figure does not break any existing rule.

    ``completion_tokens`` is always 0 — embeddings and reranking are
    "input only" from a token-billing standpoint; there is no generation.

Method label and the ``embedding_waste`` rule:
    The CallRecord ``method`` is set to ``"embed"`` (or ``"rerank"``) per
    spec, matching Voyage's actual API surface. The ``embedding_waste``
    rule (``rules/embedding_waste.py``) accepts BOTH
    ``method.endswith("embeddings.create")`` (OpenAI) and ``method == "embed"``
    (Voyage). The wrapper's ``raw_request["input"]`` carries the texts
    list, which is the shape the rule's SHA-256-based duplicate-detection
    hash function expects — identical to the OpenAI embedding wrapper's
    shape, so the rule needs no provider-specific code path.

Failure isolation: matches the pattern in ``wrappers/anthropic.py`` — a
bare ``try / except Exception: pass`` around the record-build, then
``try / except LeakDetected: raise / except Exception: pass`` around the
``record_call`` itself. Instrumentation MUST NOT break the user's call.
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
    from token_sentinel.sentinel import Sentinel


def wrap_voyage(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a Voyage AI client. Mutates ``embed`` and ``rerank`` in place.

    Supports both ``voyageai.Client`` (sync) and ``voyageai.AsyncClient``.
    Detects async via ``inspect.iscoroutinefunction`` on the original method
    — voyageai's AsyncClient keeps the bare names (``embed`` / ``rerank``)
    rather than using an ``a``-prefix, so we cannot dispatch on name alone.

    Returns the same client object with instrumented methods. Missing
    methods (e.g., a mock that only exposes ``embed``) are tolerated
    silently — we patch what's reachable.
    """
    original_embed = getattr(client, "embed", None)
    if original_embed is not None and callable(original_embed):
        if inspect.iscoroutinefunction(original_embed):
            client.embed = _make_async_embed(original_embed, sentinel)
        else:
            client.embed = _make_sync_embed(original_embed, sentinel)

    original_rerank = getattr(client, "rerank", None)
    if original_rerank is not None and callable(original_rerank):
        if inspect.iscoroutinefunction(original_rerank):
            client.rerank = _make_async_rerank(original_rerank, sentinel)
        else:
            client.rerank = _make_sync_rerank(original_rerank, sentinel)

    return client


# ---------------------------------------------------------------------------
# embed — sync and async
# ---------------------------------------------------------------------------


def _make_sync_embed(original_embed: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_embed)
    def instrumented_embed(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_embed(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary (mirrors anthropic.py):
        # - Record-building errors are swallowed (instrumentation must
        #   never break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected,
        #   which is the entire point of mode='block' and must propagate.
        try:
            record = _build_embed_record(
                session_id=session_id,
                args=args,
                kwargs=kwargs,
                latency_ms=elapsed_ms,
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

    return instrumented_embed


def _make_async_embed(original_embed: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_embed)
    async def instrumented_embed(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await original_embed(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_embed_record(
                session_id=session_id,
                args=args,
                kwargs=kwargs,
                latency_ms=elapsed_ms,
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

    return instrumented_embed


# ---------------------------------------------------------------------------
# rerank — sync and async
# ---------------------------------------------------------------------------


def _make_sync_rerank(original_rerank: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_rerank)
    def instrumented_rerank(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_rerank(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_rerank_record(
                session_id=session_id,
                args=args,
                kwargs=kwargs,
                latency_ms=elapsed_ms,
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

    return instrumented_rerank


def _make_async_rerank(original_rerank: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_rerank)
    async def instrumented_rerank(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await original_rerank(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_rerank_record(
                session_id=session_id,
                args=args,
                kwargs=kwargs,
                latency_ms=elapsed_ms,
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

    return instrumented_rerank


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _extract_texts(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Pull the ``texts`` argument off ``embed(texts=..., ...)``.

    voyageai's embed signature is ``embed(self, texts: List[str], model=...,
    ...)`` so a positional call lands at index 0. We coerce any non-string
    element to ``str()`` defensively — a customer could pass numbers, bytes,
    etc., and we'd rather under-count than crash the wrapper.
    """
    texts = kwargs.get("texts")
    if texts is None and args:
        texts = args[0]
    if texts is None:
        return []
    # Single string is also valid input — treat as a one-item list.
    if isinstance(texts, str):
        return [texts]
    try:
        return [t if isinstance(t, str) else str(t) for t in texts]
    except TypeError:
        return []


def _extract_rerank_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, list[str]]:
    """Pull ``query`` and ``documents`` off ``rerank(query=..., documents=..., ...)``."""
    query = kwargs.get("query")
    documents = kwargs.get("documents")
    # Positional fallback: rerank(self, query, documents, model, ...).
    if query is None and len(args) >= 1:
        query = args[0]
    if documents is None and len(args) >= 2:
        documents = args[1]
    query_str = query if isinstance(query, str) else (str(query) if query is not None else "")
    if documents is None:
        documents_list: list[str] = []
    elif isinstance(documents, str):
        documents_list = [documents]
    else:
        try:
            documents_list = [d if isinstance(d, str) else str(d) for d in documents]
        except TypeError:
            documents_list = []
    return query_str, documents_list


def _char_count(strings: list[str]) -> int:
    """Sum the character count across all inputs. Used as the
    ``prompt_tokens`` proxy — see module docstring for rationale."""
    return sum(len(s) for s in strings)


def _request_hash_for_embed(model: str, texts: list[str]) -> str:
    """Stable SHA-256 hash of (model, texts) for the request_hash field.

    Identical-input embeds in the same session produce identical hashes,
    which is what duplicate-detection rules key on.
    """
    return hashlib.sha256(
        json.dumps({"model": model, "input": texts}, sort_keys=True, default=str).encode()
    ).hexdigest()


def _request_hash_for_rerank(model: str, query: str, documents: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"model": model, "query": query, "documents": documents},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _build_embed_record(
    *,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a Voyage ``embed`` call.

    ``prompt_tokens`` is the total char count across inputs (char-count
    proxy — see module docstring). ``completion_tokens`` is 0 because
    embeddings have no generated output. ``raw_request`` carries ``input``
    so the embedding_waste rule (which keys on ``raw_request["input"]``)
    can detect duplicate-input embeddings — even though the rule's
    current ``method`` suffix check (``endswith("embeddings.create")``)
    does not match our ``"embed"`` label, the request-shape compatibility
    means a future rule fix needs nothing on the wrapper side.
    """
    texts = _extract_texts(args, kwargs)
    model = kwargs.get("model") or "unknown"
    if not isinstance(model, str):
        model = str(model)
    input_type = kwargs.get("input_type")
    prompt_tokens = _char_count(texts)
    request_hash = _request_hash_for_embed(model, texts)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="voyage",
        model=model,
        method="embed",
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=False,
        # ``input`` mirrors the OpenAI embedding wrapper's key so any
        # rule that already keys on raw_request["input"] (e.g.,
        # embedding_waste) sees the same shape.
        raw_request={
            "input": texts,
            "model": model,
            "input_type": input_type,
            "input_count": len(texts),
        },
        raw_response_meta={},
    )


def _build_rerank_record(
    *,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a Voyage ``rerank`` call.

    For rerank, the spec asks ``prompt_tokens`` to be the doc count as a
    proxy. We follow the spec — doc count keeps rerank-specific rules
    (e.g., "rerank N >> retrieve K" sanity check) cheap to write while
    still being meaningful at the per-call level.
    """
    query, documents = _extract_rerank_inputs(args, kwargs)
    model = kwargs.get("model") or "unknown"
    if not isinstance(model, str):
        model = str(model)
    top_k = kwargs.get("top_k")
    # Doc count proxy: see method docstring above.
    prompt_tokens = len(documents)
    request_hash = _request_hash_for_rerank(model, query, documents)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider="voyage",
        model=model,
        method="rerank",
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=False,
        raw_request={
            "query": query,
            "documents": documents,
            "model": model,
            "top_k": top_k,
            "input_count": len(documents),
        },
        raw_response_meta={},
    )
