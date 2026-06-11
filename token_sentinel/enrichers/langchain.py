"""LangChain ``BaseCallbackHandler`` enricher for TokenSentinel.

Customers who build agents on LangChain can attach
:class:`TokenSentinelCallbackHandler` to their chain/agent config to get the
same coverage as ``sentinel.wrap(client)`` would provide for a raw provider
client, without rewriting their stack:

.. code-block:: python

    from token_sentinel import Sentinel
    from token_sentinel.enrichers import TokenSentinelCallbackHandler

    sentinel = Sentinel(project="my-agent", mode="alert")
    handler = TokenSentinelCallbackHandler(sentinel)

    @sentinel.on_leak
    def alert(event):
        print(event)

    result = chain.invoke({"input": "..."}, config={"callbacks": [handler]})

The handler observes LangChain's callback bus and produces one
:class:`CallRecord` per LLM call (``on_llm_start`` paired with
``on_llm_end``/``on_llm_error``), one tool-call entry per
``on_tool_start``/``on_tool_end`` pair, and a session boundary per
top-level ``on_chain_start``/``on_chain_end`` cycle. The records flow
through :meth:`Sentinel.record_call` exactly as the wrappers do — rules
fire, ``on_leak`` handlers receive events, block mode raises
:class:`LeakDetected` from the callback frame (LangChain treats raised
callback exceptions as fatal, which matches the wrapper semantics).

Design notes:
    1. *Module import is dependency-free.* ``langchain_core`` is imported
       lazily inside a ``try``; missing-package customers get a clean
       ``ImportError`` at handler construction time, never at import.
    2. *Per-``run_id`` state.* LangChain emits start/end events with a
       UUID ``run_id``. We bucket start-time, model, prompts, etc. under
       the run_id and pop on end. Concurrent calls (different chains,
       same callback handler instance) work because the bucket is keyed.
    3. *Session-id management.* By default we mint a single session UUID
       per handler — that matches the "one agent run = one session"
       mental model. A customer can override with ``session_id=...``
       (multi-tenant agents reusing one handler with their own bucket)
       or call :meth:`new_session` between independent invocations.
    4. *Provider detection.* The ``serialized`` dict in ``on_llm_start``
       commonly contains a class chain like
       ``["langchain", "chat_models", "anthropic", "ChatAnthropic"]``.
       We sniff that to populate the ``provider`` field; falling back to
       ``"langchain"`` keeps records routable when the shape is unknown.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Soft import of langchain_core — module loads cleanly without LangChain so
# `from token_sentinel.enrichers import TokenSentinelCallbackHandler` never
# crashes a base install. Instantiation raises ImportError with a hint.
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LangChainBase

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when LC not installed
    # Fallback so the module imports cleanly without LangChain installed.
    # The class below conditionally subclasses _LangChainBase, but if a
    # customer manages to instantiate ``TokenSentinelCallbackHandler``
    # without LangChain they hit the ImportError in __init__ before any
    # base-class machinery matters. ``Any`` keeps mypy happy in both
    # branches without per-line ignores.
    _LangChainBase = object  # type: ignore[misc,assignment]
    _LANGCHAIN_AVAILABLE = False


from token_sentinel.events import CallRecord, LeakDetected

if TYPE_CHECKING:
    from token_sentinel.sentinel import Sentinel


# ---------------------------------------------------------------------------
# Internal per-run state
# ---------------------------------------------------------------------------


class _RunState:
    """Mutable bookkeeping for a single in-flight LangChain run.

    ``on_llm_start`` populates this; ``on_llm_end`` / ``on_llm_error`` reads
    it back to build a :class:`CallRecord` covering the entire call window.
    Kept as a class (not a dict) so mypy strict can verify the attribute
    shape; ``__slots__`` keeps memory minimal under high run rates.
    """

    __slots__ = (
        "start_perf",
        "model",
        "provider",
        "prompts",
        "kind",
        "extra",
    )

    def __init__(
        self,
        *,
        start_perf: float,
        model: str,
        provider: str,
        prompts: list[str],
        kind: str,
        extra: dict[str, Any],
    ) -> None:
        self.start_perf = start_perf
        self.model = model
        self.provider = provider
        self.prompts = prompts
        self.kind = kind  # "llm" | "chat_model" | "tool"
        self.extra = extra


# ---------------------------------------------------------------------------
# Provider sniffer
# ---------------------------------------------------------------------------


_PROVIDER_HINTS: tuple[tuple[str, str], ...] = (
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("azure", "openai"),
    ("vertex", "gemini"),
    ("gemini", "gemini"),
    ("google", "gemini"),
    ("bedrock", "bedrock"),
    ("cohere", "cohere"),
    ("voyage", "voyage"),
    ("replicate", "replicate"),
    ("deepgram", "deepgram"),
    ("elevenlabs", "elevenlabs"),
)


def _sniff_provider(serialized: dict[str, Any] | None) -> str:
    """Pick a provider string from LangChain's ``serialized`` payload.

    Strategy: walk the ``id`` class-path list (``["langchain", "chat_models",
    "anthropic", "ChatAnthropic"]``) and match the first known provider hint.
    Falls back to scanning ``kwargs`` for ``model``/``model_name`` keys when
    the class path is ambiguous. Returns ``"langchain"`` when nothing
    matches — keeps records routable instead of crashing on partial mocks.
    """
    if not isinstance(serialized, dict):
        return "langchain"
    tokens: list[str] = []
    raw_id = serialized.get("id")
    if isinstance(raw_id, list):
        tokens.extend(str(t).lower() for t in raw_id if isinstance(t, str | int | float))
    name = serialized.get("name")
    if isinstance(name, str):
        tokens.append(name.lower())
    blob = " ".join(tokens)
    for needle, provider in _PROVIDER_HINTS:
        if needle in blob:
            return provider
    return "langchain"


def _extract_model(
    serialized: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> str:
    """Find the best ``model`` string available in a start event.

    LangChain populates the model name in one of three places depending on
    the LLM class and how it was instantiated:

      * ``serialized["kwargs"]["model"]`` — typical for ``ChatOpenAI``,
        ``ChatAnthropic`` and most ``BaseChatModel`` subclasses.
      * ``metadata["ls_model_name"]`` — populated by the LangSmith tracer
        and forwarded into the callback metadata stream.
      * ``invocation_params["model"]`` — LangChain stuffs the resolved
        invocation params under this key in ``**kwargs`` for advanced
        callback users.

    We probe each in turn and fall back to ``"unknown"`` — same default the
    Anthropic wrapper uses, so the rule engine sees a uniform sentinel.
    """
    for source in (serialized, metadata):
        if isinstance(source, dict):
            inner_kwargs = source.get("kwargs")
            if isinstance(inner_kwargs, dict):
                for key in ("model", "model_name", "deployment_name"):
                    value = inner_kwargs.get(key)
                    if isinstance(value, str) and value:
                        return value
            for key in ("ls_model_name", "model", "model_name"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    invocation_params = kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        for key in ("model", "model_name", "deployment_name"):
            value = invocation_params.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from an ``LLMResult``.

    LangChain providers stuff token usage into ``response.llm_output`` with
    several historically-supported key conventions:

      * Anthropic: ``{"usage": {"input_tokens": ..., "output_tokens": ...}}``
      * OpenAI:    ``{"token_usage": {"prompt_tokens": ..., "completion_tokens": ...}}``
      * Bedrock:   ``{"usage": {"prompt_tokens": ..., "completion_tokens": ...}}``
      * Cohere:    ``{"meta": {"billed_units": {"input_tokens": ..., "output_tokens": ...}}}``

    Per-generation ``usage_metadata`` (LangChain >= 0.2 standard) is the
    fallback when ``llm_output`` is missing — sum across all generations.
    """
    prompt = 0
    completion = 0
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        # Common shape #1 — OpenAI / Bedrock.
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict):
            prompt = _coerce_int(usage.get("prompt_tokens"), usage.get("input_tokens"))
            completion = _coerce_int(usage.get("completion_tokens"), usage.get("output_tokens"))
        # Common shape #2 — Cohere's nested billed_units.
        if prompt == 0 and completion == 0:
            meta = llm_output.get("meta")
            if isinstance(meta, dict):
                billed = meta.get("billed_units")
                if isinstance(billed, dict):
                    prompt = _coerce_int(billed.get("input_tokens"), billed.get("prompt_tokens"))
                    completion = _coerce_int(
                        billed.get("output_tokens"),
                        billed.get("completion_tokens"),
                    )

    # Fallback to per-generation usage_metadata (BaseMessage.usage_metadata).
    if prompt == 0 and completion == 0:
        generations = getattr(response, "generations", None) or []
        for batch in generations:
            for gen in batch or []:
                # ChatGeneration.message.usage_metadata
                message = getattr(gen, "message", None)
                usage_md = getattr(message, "usage_metadata", None)
                if isinstance(usage_md, dict):
                    prompt += _coerce_int(usage_md.get("input_tokens"))
                    completion += _coerce_int(usage_md.get("output_tokens"))
                    continue
                # Generation.generation_info["usage"] (legacy)
                info = getattr(gen, "generation_info", None)
                if isinstance(info, dict):
                    usage = info.get("usage") or info.get("token_usage")
                    if isinstance(usage, dict):
                        prompt += _coerce_int(usage.get("input_tokens"), usage.get("prompt_tokens"))
                        completion += _coerce_int(
                            usage.get("output_tokens"),
                            usage.get("completion_tokens"),
                        )

    return prompt, completion


def _coerce_int(*candidates: Any) -> int:
    """Return the first non-``None`` candidate coerced to a non-negative int.

    The cloud cost estimator chokes on negative token counts (a buggy LC
    provider once emitted ``-1`` for "unavailable"). Clamp at zero so a
    malformed payload can't poison the rule loop.
    """
    for c in candidates:
        if c is None:
            continue
        try:
            value = int(c)
        except (TypeError, ValueError):
            continue
        return max(0, value)
    return 0


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Pull ``tool_calls`` from each generation's message, if present.

    LangChain's ``AIMessage`` exposes ``tool_calls`` as a list of dicts
    matching ``{"name": str, "args": dict, "id": str}``. We normalise to
    the rule engine's shape: ``{"name": str, "arguments": dict}``.
    """
    out: list[dict[str, Any]] = []
    generations = getattr(response, "generations", None) or []
    for batch in generations:
        for gen in batch or []:
            message = getattr(gen, "message", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name", "")
                    args = tc.get("args") or tc.get("arguments") or {}
                    out.append({"name": str(name), "arguments": args})
    return out


def _has_text_output(response: Any) -> bool:
    """Return True when the LLM produced any user-facing text content."""
    generations = getattr(response, "generations", None) or []
    for batch in generations:
        for gen in batch or []:
            text = getattr(gen, "text", None)
            if isinstance(text, str) and text.strip():
                return True
    return False


def _request_hash(model: str, prompts: list[str], kind: str) -> str:
    """Stable per-call hash for the rule engine's retry-storm detector."""
    payload = {"model": model, "prompts": prompts, "kind": kind}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


class TokenSentinelCallbackHandler(_LangChainBase):
    """LangChain callback handler that records every LLM/tool call into
    :class:`Sentinel`.

    Args:
        sentinel: The :class:`Sentinel` instance to route records into.
        session_id: Optional explicit session id. When omitted, a UUID is
            minted at construction time so all calls under one handler land
            in the same session bucket. Pass an explicit id when a single
            handler is shared across logically-separate agent runs and
            you'd like to bucket them yourself.

    Raises:
        ImportError: when ``langchain_core`` is not installed. The handler
            cannot meaningfully exist without LangChain's
            :class:`BaseCallbackHandler`, so we fail loud at construction
            rather than silently no-op at runtime.
    """

    # LangChain inspects these class attributes to decide whether to fire
    # the relevant event channels. Setting all to True is the wrapper-parity
    # behaviour — we observe every event surface we instrument.
    raise_error: bool = False
    run_inline: bool = False

    def __init__(
        self,
        sentinel: Sentinel,
        *,
        session_id: str | None = None,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "TokenSentinelCallbackHandler requires langchain_core. "
                "Install via `pip install token-sentinel[langchain]` or "
                "`pip install langchain-core>=0.3`."
            )
        super().__init__()
        self._sentinel = sentinel
        self._session_id = session_id or str(uuid.uuid4())
        self._runs: dict[uuid.UUID, _RunState] = {}
        # Tracks the *top-level* chain id (parent_run_id is None) so nested
        # chains share the same logical session. Cleared on the matching
        # on_chain_end so a long-running handler doesn't accumulate state.
        self._top_level_chain: uuid.UUID | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Current session id used for all recorded calls."""
        return self._session_id

    def new_session(self, session_id: str | None = None) -> str:
        """Rotate the handler's session id.

        Useful when one handler instance is reused across independent agent
        invocations (e.g., a worker that pulls jobs off a queue) and the
        customer wants per-job session bucketing in the dashboards. Returns
        the new session id.
        """
        with self._lock:
            self._session_id = session_id or str(uuid.uuid4())
            self._runs.clear()
            self._top_level_chain = None
        return self._session_id

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a string-LLM call's start. Pairs with :meth:`on_llm_end`."""
        self._open_llm_run(
            run_id=run_id,
            serialized=serialized,
            prompts=list(prompts) if prompts else [],
            metadata=metadata,
            kwargs=kwargs,
            kind="llm",
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a chat-LLM call's start. ``messages`` is a list of message
        batches; we serialise each one to a prompt-ish string for hashing /
        evidence purposes (rules don't need the full message AST)."""
        flattened: list[str] = []
        try:
            for batch in messages or []:
                pieces: list[str] = []
                for msg in batch or []:
                    content = getattr(msg, "content", None)
                    if content is None:
                        content = str(msg)
                    pieces.append(str(content))
                flattened.append("\n".join(pieces))
        except Exception:
            # Never let prompt-extraction crash callback dispatch.
            flattened = [str(messages)] if messages else []
        self._open_llm_run(
            run_id=run_id,
            serialized=serialized,
            prompts=flattened,
            metadata=metadata,
            kwargs=kwargs,
            kind="chat_model",
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the completion of an LLM call. Builds a :class:`CallRecord`
        and routes it through :meth:`Sentinel.record_call`."""
        state = self._pop_run(run_id)
        if state is None:
            # Out-of-order end: a `parent_run_id` was tracked but the
            # corresponding start wasn't routed through this handler (e.g.,
            # mid-run handler attach). Skip silently — instrumentation must
            # never crash a real call.
            return
        elapsed_ms = (time.perf_counter() - state.start_perf) * 1000.0
        try:
            prompt_tokens, completion_tokens = _extract_token_usage(response)
            tool_calls = _extract_tool_calls(response)
            has_text = _has_text_output(response)
            llm_output = getattr(response, "llm_output", None)
            raw_response_meta: dict[str, Any] = {
                "via": "langchain",
                "kind": state.kind,
            }
            if isinstance(llm_output, dict):
                model_name = llm_output.get("model_name") or llm_output.get("model")
                if isinstance(model_name, str) and model_name:
                    # Models named in llm_output are sometimes more specific
                    # than the bound model (e.g., "gpt-4o-2024-08-06" vs
                    # "gpt-4o"). Prefer the more specific value when present.
                    state.model = model_name
                raw_response_meta["finish_reason"] = llm_output.get("finish_reason")
            record = CallRecord(
                session_id=self._session_id,
                timestamp=datetime.now(timezone.utc),
                provider=state.provider,
                model=state.model,
                method=f"langchain.{state.kind}",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
                request_hash=_request_hash(state.model, state.prompts, state.kind),
                tool_calls=tool_calls,
                user_facing_output=has_text and not tool_calls,
                raw_request={"prompts": state.prompts, **state.extra},
                raw_response_meta=raw_response_meta,
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            # Block-mode propagation. LangChain catches callback exceptions
            # and (by default) logs them; raising LeakDetected here means
            # the customer sees the block at the boundary where their
            # callbacks list is consumed — same contract as the wrapper.
            raise
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an LLM failure. Emits a CallRecord with zero tokens and
        ``raw_response_meta["error"]`` populated so the retry-storm rule
        can fire on repeated failure patterns."""
        state = self._pop_run(run_id)
        if state is None:
            return
        elapsed_ms = (time.perf_counter() - state.start_perf) * 1000.0
        try:
            record = CallRecord(
                session_id=self._session_id,
                timestamp=datetime.now(timezone.utc),
                provider=state.provider,
                model=state.model,
                method=f"langchain.{state.kind}",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                request_hash=_request_hash(state.model, state.prompts, state.kind),
                tool_calls=[],
                user_facing_output=False,
                raw_request={"prompts": state.prompts, **state.extra},
                raw_response_meta={
                    "via": "langchain",
                    "kind": state.kind,
                    "error": type(error).__name__,
                    "error_message": str(error)[:500],
                },
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool-call start. Tool calls are NOT priced as token
        spend on their own — they're context that the tool_loop and
        tool_definition_bloat rules use to spot suspicious patterns."""
        name = "unknown"
        if isinstance(serialized, dict):
            raw_name = serialized.get("name")
            if isinstance(raw_name, str) and raw_name:
                name = raw_name
        with self._lock:
            self._runs[run_id] = _RunState(
                start_perf=time.perf_counter(),
                model=name,
                provider="langchain.tool",
                prompts=[input_str if isinstance(input_str, str) else str(input_str)],
                kind="tool",
                extra={"tool_name": name, "inputs": inputs},
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the completion of a tool call. We emit a CallRecord with
        ``tool_calls=[{name, arguments}]`` so the tool_loop rule's
        repeat-detection window catches infinite loops of the same tool +
        same input."""
        state = self._pop_run(run_id)
        if state is None:
            return
        elapsed_ms = (time.perf_counter() - state.start_perf) * 1000.0
        try:
            record = CallRecord(
                session_id=self._session_id,
                timestamp=datetime.now(timezone.utc),
                provider="langchain.tool",
                model=state.model,
                method="langchain.tool",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                request_hash=_request_hash(state.model, state.prompts, "tool"),
                tool_calls=[
                    {
                        "name": state.model,
                        "arguments": state.extra.get("inputs")
                        or {"input": state.prompts[0] if state.prompts else ""},
                    }
                ],
                user_facing_output=False,
                raw_request={"input": state.prompts[0] if state.prompts else "", **state.extra},
                raw_response_meta={
                    "via": "langchain",
                    "kind": "tool",
                    "output_repr": repr(output)[:200],
                },
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool-call failure. Same shape as on_tool_end but with
        ``raw_response_meta["error"]`` populated."""
        state = self._pop_run(run_id)
        if state is None:
            return
        elapsed_ms = (time.perf_counter() - state.start_perf) * 1000.0
        try:
            record = CallRecord(
                session_id=self._session_id,
                timestamp=datetime.now(timezone.utc),
                provider="langchain.tool",
                model=state.model,
                method="langchain.tool",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                request_hash=_request_hash(state.model, state.prompts, "tool"),
                tool_calls=[
                    {
                        "name": state.model,
                        "arguments": state.extra.get("inputs")
                        or {"input": state.prompts[0] if state.prompts else ""},
                    }
                ],
                user_facing_output=False,
                raw_request={"input": state.prompts[0] if state.prompts else "", **state.extra},
                raw_response_meta={
                    "via": "langchain",
                    "kind": "tool",
                    "error": type(error).__name__,
                    "error_message": str(error)[:500],
                },
            )
        except Exception:
            return
        try:
            self._sentinel.record_call(record)
        except LeakDetected:
            raise
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Chain hooks (session boundary management)
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Mark the start of a chain run. The top-level chain (no parent)
        sets the session bucket; nested chains share it.

        We deliberately do NOT mint a new session id per chain — the
        customer's expectation is "one handler = one session" unless they
        explicitly rotate via :meth:`new_session`. Multiple chain.invoke()
        calls against the same handler all land in the same session, which
        is the right call for "agent-with-retries" patterns where
        retry_storm needs cross-chain visibility.
        """
        if parent_run_id is None:
            with self._lock:
                if self._top_level_chain is None:
                    self._top_level_chain = run_id

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Mark the end of a chain. When the top-level chain closes, clear
        the tracked top-level id so the next ``on_chain_start`` cycle
        starts fresh."""
        with self._lock:
            if self._top_level_chain == run_id:
                self._top_level_chain = None

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror ``on_chain_end`` for the error path so we don't leak the
        top-level chain id when an exception unwinds the run."""
        with self._lock:
            if self._top_level_chain == run_id:
                self._top_level_chain = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_llm_run(
        self,
        *,
        run_id: uuid.UUID,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        metadata: dict[str, Any] | None,
        kwargs: dict[str, Any],
        kind: str,
    ) -> None:
        """Common path for on_llm_start / on_chat_model_start."""
        model = _extract_model(serialized, metadata, kwargs)
        provider = _sniff_provider(serialized)
        invocation = kwargs.get("invocation_params")
        extra: dict[str, Any] = {}
        if isinstance(invocation, dict):
            # Capture the resolved invocation params verbatim — useful for
            # the rule engine's max_tokens / tools introspection without
            # us having to enumerate every provider's parameter name.
            for k in ("max_tokens", "max_new_tokens", "tools", "temperature"):
                if k in invocation:
                    extra[k] = invocation[k]
        with self._lock:
            self._runs[run_id] = _RunState(
                start_perf=time.perf_counter(),
                model=model,
                provider=provider,
                prompts=prompts,
                kind=kind,
                extra=extra,
            )

    def _pop_run(self, run_id: uuid.UUID) -> _RunState | None:
        """Pop and return the per-run state. Thread-safe."""
        with self._lock:
            return self._runs.pop(run_id, None)
