"""Wrap a Cohere ``cohere>=5`` client to capture call records.

Pattern: in-place mutation of ``client.chat``, ``client.embed``, and
``client.rerank``. The original is captured in a closure so the
instrumented version can delegate. This preserves all type hints in IDEs
because we mutate the live instance, not a subclass.

Reference patterns:
  - ``wrappers/anthropic.py`` for chat scaffolding (token extraction,
    tool-call mining, redaction of message content from ``raw_request``).
  - ``wrappers/voyage.py`` for embed + rerank scaffolding (char-count
    proxy for ``prompt_tokens``, ``method="embed"`` / ``method="rerank"``,
    ``user_facing_output=False``).

client choice: we wrap ``cohere.ClientV2`` and ``cohere.AsyncClientV2``
exclusively. The legacy ``cohere.Client`` is deprecated as of cohere>=5
and internal research
explicitly calls V2 "the current top-level class" for new builds; targeting
only V2 keeps the surface tight and matches what new customers will land on.
Detection in ``Sentinel.wrap`` is by class name (``ClientV2`` /
``AsyncClientV2``) plus module prefix (``cohere``), so a legacy ``cohere.Client``
falls through to the "Unsupported" branch with a clear error.

Six cases handled by ``wrap_cohere``:
  1. Sync chat:    ``ClientV2.chat(model=..., messages=[...])``
  2. Sync embed:   ``ClientV2.embed(model=..., texts=[...], input_type=...)``
  3. Sync rerank:  ``ClientV2.rerank(model=..., query=..., documents=[...])``
  4. Async chat:   ``AsyncClientV2.chat(...)`` (coroutine)
  5. Async embed:  ``AsyncClientV2.embed(...)`` (coroutine)
  6. Async rerank: ``AsyncClientV2.rerank(...)`` (coroutine)

Detection of async-vs-sync mirrors ``wrappers/voyage.py``: the AsyncClientV2
keeps the same bare method names as ClientV2 (``chat`` / ``embed`` /
``rerank``) rather than using an ``a``-prefix, so we use
``inspect.iscoroutinefunction`` on the original method.

Token accounting:

  * **chat**: Cohere's ``V2ChatResponse.usage`` exposes both
    ``tokens.input_tokens`` / ``tokens.output_tokens`` (raw token counts)
    AND ``billed_units.input_tokens`` / ``billed_units.output_tokens``
    (the actual billing meter, which can differ when Cohere's
    server-side caching applies a discount). Per the specification we read
    from ``usage.tokens.*`` for the chat record. If a future cohere
    release changes the layout, we fall back to zeros — record-building
    never crashes.

  * **embed**: prompt_tokens fallback chain:
        1. ``response.meta.billed_units.input_tokens`` (Cohere's actual
           billable token count — preferred when present because it
           matches what the customer will see on their invoice).
        2. ``response.usage.tokens.input_tokens`` (some Cohere responses
           include both ``meta`` AND ``usage`` blocks; ``meta`` wins
           because that's the billing source of truth, but ``usage`` is
           a clean fallback for SDK shape variations).
        3. ``sum(len(t) for t in texts)`` (char-count proxy — see
           ``wrappers/voyage.py`` module docstring for the rationale).
           This is the lowest-resolution fallback but ensures
           ``prompt_tokens`` is always populated with something stable
           and growth-monotonic, which is all the embedding_waste rule
           needs (it keys on duplicate input hashes, not absolute
           token counts).

  * **rerank**: Cohere does NOT return a per-token count for rerank — the
    response carries ``meta.billed_units.search_units`` (the billing unit),
    not token counts. We use ``len(query) + sum(len(d) for d in documents)``
    as the char-count proxy for ``prompt_tokens``. ``completion_tokens=0``
    (no generation). The ``search_units`` value is preserved in
    ``raw_request["search_units"]`` for future use; planned
    ``usage_extra`` integration will surface it to the cost estimator.

rerank_thrash rule (deferred):
    The research called out a candidate rule "rerank_thrash" that
    would fire when the same ``(query, doc_set)`` is reranked multiple
    times in a session — the rerank-analog of the existing
    ``embedding_waste`` rule. The wrapper captures the necessary
    telemetry (``raw_request["query"]`` + ``raw_request["documents"]``
    + ``request_hash``) so a future rule can be written without any
    wrapper changes. The rule itself is scope, NOT shipped here.

Redaction:
    ``raw_request`` for chat strips the full ``messages`` content array
    and keeps only the model name and tool definitions (which are
    schema, not customer data). Mirrors the anthropic wrapper's
    redaction discipline. For embed/rerank, the texts list IS the
    input — we keep it under ``raw_request["input"]`` for the
    embedding_waste hash to work, the same way ``wrappers/voyage.py``
    keeps it. Customers who consider their embedding inputs sensitive
    can configure global redaction at the rule layer.

Failure isolation: standard two-level safety boundary mirrors
``wrappers/voyage.py`` and ``wrappers/anthropic.py``. Record-building
errors are swallowed; ``record_call`` exceptions are caught EXCEPT
``LeakDetected``, which must propagate so block mode works.
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


PROVIDER = "cohere"


def wrap_cohere(client: Any, sentinel: Sentinel) -> Any:
    """Wrap a Cohere V2 client. Mutates ``chat`` / ``embed`` / ``rerank``
    in place.

    Supports both ``cohere.ClientV2`` (sync) and ``cohere.AsyncClientV2``.
    Detects async via ``inspect.iscoroutinefunction`` on each method
    independently — although the official SDK keeps the methods coherent
    (all sync on ClientV2, all async on AsyncClientV2), a customer's
    subclass might mix them, so we check per-method rather than once
    per client. Missing methods (e.g., a mock that only exposes
    ``embed``) are tolerated silently — we patch what's reachable.

    Returns the same client object with instrumented methods.
    """
    for method_name, sync_factory, async_factory in (
        ("chat", _make_sync_chat, _make_async_chat),
        ("embed", _make_sync_embed, _make_async_embed),
        ("rerank", _make_sync_rerank, _make_async_rerank),
    ):
        original = getattr(client, method_name, None)
        if original is None or not callable(original):
            continue
        if inspect.iscoroutinefunction(original):
            setattr(client, method_name, async_factory(original, sentinel))
        else:
            setattr(client, method_name, sync_factory(original, sentinel))
    return client


# ---------------------------------------------------------------------------
# chat — sync and async
# ---------------------------------------------------------------------------


def _make_sync_chat(original_chat: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_chat)
    def instrumented_chat(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = original_chat(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Two-level safety boundary (mirrors anthropic.py / voyage.py):
        # - Record-building errors are swallowed (instrumentation must
        #   never break the user's call).
        # - record_call exceptions are caught EXCEPT for LeakDetected,
        #   which is the entire point of mode='block' and must propagate.
        try:
            record = _build_chat_record(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
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

    return instrumented_chat


def _make_async_chat(original_chat: Any, sentinel: Sentinel) -> Any:
    @functools.wraps(original_chat)
    async def instrumented_chat(*args: Any, **kwargs: Any) -> Any:
        session_id: str = kwargs.pop("_sentinel_session_id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await original_chat(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

        try:
            record = _build_chat_record(
                session_id=session_id,
                kwargs=kwargs,
                response=response,
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

    return instrumented_chat


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

        try:
            record = _build_embed_record(
                session_id=session_id,
                args=args,
                kwargs=kwargs,
                response=response,
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
                response=response,
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
                response=response,
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
                response=response,
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
# Argument extraction helpers
# ---------------------------------------------------------------------------


def _extract_texts(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Pull the ``texts`` argument off ``embed(texts=..., ...)``.

    Cohere V2's ``embed`` signature is keyword-only (``embed(*, model: str,
    input_type: ..., texts: Optional[Sequence[str]] = ...)``), so we read
    from kwargs first. We fall back to args[0] for defensive parity with
    older or forked SDKs that may permit positional. Non-string elements
    are coerced via ``str()`` — we'd rather under-count than crash the
    wrapper on a customer's odd input.
    """
    texts = kwargs.get("texts")
    if texts is None and args:
        # Defensive fallback: if a customer reflectively bypasses the
        # keyword-only signature, the texts may show up positionally.
        texts = args[0]
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    try:
        return [t if isinstance(t, str) else str(t) for t in texts]
    except TypeError:
        return []


def _extract_rerank_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, list[str]]:
    """Pull ``query`` and ``documents`` off ``rerank(query=..., documents=..., ...)``.

    Like ``_extract_texts``, the rerank signature is keyword-only on
    ClientV2, but we read from kwargs first and fall back positionally
    for resilience to forked SDKs.
    """
    query = kwargs.get("query")
    documents = kwargs.get("documents")
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
    """Sum the character count across all inputs.

    Used as the fallback ``prompt_tokens`` proxy when the response's
    ``meta.billed_units.input_tokens`` is missing. See module docstring.
    """
    return sum(len(s) for s in strings)


# ---------------------------------------------------------------------------
# Response field extraction
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    """Return ``value`` as an int if it can be cleanly coerced, else None.

    Cohere SDK fields are typed ``Optional[int]`` by Pydantic; in practice
    they arrive as ``int`` or ``None`` but a wire-format quirk could
    surface a float. We accept any numeric and coerce defensively rather
    than crash record building.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is a subclass of int; reject it explicitly so a stray True
        # doesn't become ``1`` in a token count column.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_chat_usage(response: Any) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from a chat response.

    Per the specification, we read from ``response.usage.tokens.input_tokens``
    and ``response.usage.tokens.output_tokens`` (Cohere V2's
    ``UsageTokens`` block). Any missing intermediate attribute yields 0
    rather than crashing — the wrapper must always produce a record.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        tokens = getattr(usage, "tokens", None)
        if tokens is None:
            return 0, 0
        input_tokens = _safe_int(getattr(tokens, "input_tokens", None)) or 0
        output_tokens = _safe_int(getattr(tokens, "output_tokens", None)) or 0
        return input_tokens, output_tokens
    except Exception:
        return 0, 0


def _extract_embed_billed_tokens(response: Any) -> int | None:
    """Return Cohere's billed ``input_tokens`` from an embed response, or None.

    First priority is ``response.meta.billed_units.input_tokens``
    (the specification's preferred path — matches what the customer is
    invoiced for). Second priority is ``response.usage.tokens.input_tokens``
    in case a future SDK version exposes a ``usage`` block on embed
    responses. Returns None when neither is available, signalling to
    the caller that the char-count proxy fallback should kick in.
    """
    try:
        meta = getattr(response, "meta", None)
        if meta is not None:
            billed = getattr(meta, "billed_units", None)
            if billed is not None:
                value = _safe_int(getattr(billed, "input_tokens", None))
                if value is not None:
                    return value
    except Exception:
        pass
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "tokens", None)
            if tokens is not None:
                value = _safe_int(getattr(tokens, "input_tokens", None))
                if value is not None:
                    return value
    except Exception:
        pass
    return None


def _extract_rerank_search_units(response: Any) -> int | None:
    """Return Cohere's ``meta.billed_units.search_units`` if present, else None.

    Captured into ``raw_request["search_units"]`` for planned
    ``usage_extra`` surfacing. Not used for ``prompt_tokens`` because
    the spec asks for char count there (search_units is a fundamentally
    different billing unit — 1 per [query + ≤100 docs] window).
    """
    try:
        meta = getattr(response, "meta", None)
        if meta is None:
            return None
        billed = getattr(meta, "billed_units", None)
        if billed is None:
            return None
        return _safe_int(getattr(billed, "search_units", None))
    except Exception:
        return None


def _extract_tool_calls(response: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(tool_calls, has_text_output)`` from a chat response.

    Cohere V2's ``message.tool_calls`` is ``Optional[List[ToolCallV2]]``
    where each ``ToolCallV2`` has ``function.name`` and
    ``function.arguments`` (a JSON-encoded string). We parse arguments
    to a dict for parity with the anthropic wrapper's tool_calls shape
    (``{"name": ..., "arguments": {...}}``), falling back to the raw
    string when JSON parsing fails (mirrors the OpenAI wrapper's
    streamed-tool-call handling for malformed argument JSON).

    ``has_text_output`` is True iff ``message.content`` contains at
    least one non-empty text block — used to compute
    ``user_facing_output`` in the same way the anthropic wrapper does.
    """
    tool_calls: list[dict[str, Any]] = []
    has_text_output = False
    try:
        message = getattr(response, "message", None)
        if message is None:
            return tool_calls, has_text_output
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            try:
                func = getattr(tc, "function", None)
                name = getattr(func, "name", "") if func is not None else ""
                raw_args = getattr(func, "arguments", "") if func is not None else ""
                if isinstance(raw_args, dict):
                    parsed_args: Any = raw_args
                elif isinstance(raw_args, str) and raw_args:
                    try:
                        parsed_args = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        # Preserve the raw string so a downstream rule
                        # can still inspect the failed payload.
                        parsed_args = raw_args
                else:
                    parsed_args = {}
                tool_calls.append({"name": name or "", "arguments": parsed_args})
            except Exception:
                # One malformed tool_call shouldn't drop the others.
                continue

        content = getattr(message, "content", None) or []
        # Cohere's content can be a list of typed blocks (``{type: "text",
        # text: "..."}``) OR a plain string in some SDK forks. Handle both.
        if isinstance(content, str):
            has_text_output = bool(content.strip())
        else:
            try:
                for block in content:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        text = getattr(block, "text", "") or ""
                        if text.strip():
                            has_text_output = True
                            break
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "") or ""
                        if text.strip():
                            has_text_output = True
                            break
            except TypeError:
                # Non-iterable content; treat as no text output.
                pass
    except Exception:
        pass
    return tool_calls, has_text_output


# ---------------------------------------------------------------------------
# Request hashing
# ---------------------------------------------------------------------------


def _request_hash_for_chat(model: str, messages: Any, tools: Any) -> str:
    """Stable SHA-256 hash for chat — keyed on (model, messages, tools).

    Mirrors the anthropic wrapper's hash inputs. messages is hashed in
    full (not redacted to ``raw_request``) because the hash IS the
    duplicate-detection key for the retry_storm rule; redaction at the
    raw_request layer doesn't affect the hash.
    """
    return hashlib.sha256(
        json.dumps(
            {"model": model, "messages": messages, "tools": tools},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def _request_hash_for_embed(model: str, texts: list[str]) -> str:
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


# ---------------------------------------------------------------------------
# CallRecord builders
# ---------------------------------------------------------------------------


def _build_chat_record(
    *,
    session_id: str,
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a Cohere V2 chat call.

    Token counts come from ``response.usage.tokens.input_tokens`` /
    ``output_tokens`` (Cohere's raw token meter on V2 chat responses).
    Tool calls are mined from ``response.message.tool_calls``; their
    ``function.arguments`` JSON string is parsed to a dict for parity
    with anthropic's tool_calls shape.

    Redaction: ``raw_request`` strips the full ``messages`` content array
    and keeps only the model name, tool definitions (schema, not customer
    data), and structural metadata (max_tokens etc.). This mirrors the
    anthropic wrapper. The message contents are still hashed into
    ``request_hash`` via ``_request_hash_for_chat`` — that hash is used
    by the retry_storm rule to detect repeated identical prompts.
    """
    model = kwargs.get("model") or "unknown"
    if not isinstance(model, str):
        model = str(model)
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    max_tokens = kwargs.get("max_tokens", 0)

    prompt_tokens, completion_tokens = _extract_chat_usage(response)
    tool_calls, has_text_output = _extract_tool_calls(response)
    user_facing_output = bool(has_text_output)
    finish_reason = getattr(response, "finish_reason", None)

    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider=PROVIDER,
        model=model,
        method="chat",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=_request_hash_for_chat(model, messages, tools),
        tool_calls=tool_calls,
        user_facing_output=user_facing_output,
        # Redacted: ``messages`` content stripped, schema-only fields kept.
        # The hash above already captured the messages for dedup rules.
        raw_request={
            "model": model,
            "tools": tools,
            "max_tokens": max_tokens,
            "message_count": len(messages) if isinstance(messages, list) else 0,
        },
        raw_response_meta={"finish_reason": finish_reason},
    )


def _build_embed_record(
    *,
    session_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    response: Any,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a Cohere V2 embed call.

    ``prompt_tokens`` follows the three-tier fallback chain documented
    in the module docstring: (1) ``meta.billed_units.input_tokens``,
    (2) ``usage.tokens.input_tokens``, (3) char count.

    ``raw_request["input"]`` mirrors the OpenAI / Voyage embedding
    wrapper key so the ``embedding_waste`` rule (which keys on
    ``raw_request["input"]``) sees a familiar shape — its
    SHA-256-based duplicate-detection runs unchanged across providers.
    """
    texts = _extract_texts(args, kwargs)
    model = kwargs.get("model") or "unknown"
    if not isinstance(model, str):
        model = str(model)
    input_type = kwargs.get("input_type")

    # Three-tier prompt_tokens fallback chain. Char count is the
    # always-available baseline; we prefer the response's billed token
    # count when present because that matches the customer's invoice.
    billed = _extract_embed_billed_tokens(response)
    prompt_tokens = billed if billed is not None else _char_count(texts)

    request_hash = _request_hash_for_embed(model, texts)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider=PROVIDER,
        model=model,
        method="embed",
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=[],
        user_facing_output=False,
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
    response: Any,
    latency_ms: float,
) -> CallRecord:
    """Build a CallRecord from a Cohere V2 rerank call.

    ``prompt_tokens`` is the char-count proxy ``len(query) + sum(len(d)
    for d in documents)`` per spec — Cohere does NOT expose per-token
    counts for rerank (billing is per-search-unit, not per-token).

    ``raw_request["search_units"]`` carries the billing meter from
    ``response.meta.billed_units.search_units`` for planned
    ``usage_extra`` surfacing. ``top_n`` and ``document_count`` are
    captured in ``raw_request`` as the spec's ``model_specific_meta``
    placeholder;  will lift them to ``usage_extra``.

    NOTE FOR : a ``rerank_thrash`` rule candidate would fire when
    the same (query, doc_set) is reranked multiple times within a
    session. ``request_hash`` is keyed on (model, query, documents)
    so the rule can use the existing dedup_window infrastructure
    without any wrapper changes. The rule itself is scope.
    """
    query, documents = _extract_rerank_inputs(args, kwargs)
    model = kwargs.get("model") or "unknown"
    if not isinstance(model, str):
        model = str(model)
    top_n = kwargs.get("top_n")

    prompt_tokens = len(query) + _char_count(documents)
    request_hash = _request_hash_for_rerank(model, query, documents)
    search_units = _extract_rerank_search_units(response)
    return CallRecord(
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        provider=PROVIDER,
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
            "top_n": top_n,
            "document_count": len(documents),
            "search_units": search_units,
        },
        raw_response_meta={},
    )
