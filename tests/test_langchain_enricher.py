"""Tests for ``token_sentinel.enrichers.langchain.TokenSentinelCallbackHandler``.

NO real LangChain agent / chain is invoked — we synthesise the callback
events with ``unittest.mock`` and ``types.SimpleNamespace`` shapes so the
test suite stays fast and dependency-light. ``langchain_core`` IS imported
to verify the real base-class wiring; the ImportError path is exercised by
monkey-patching the module-level flag.

The handler's contract:

  1. Constructs cleanly when ``langchain_core`` is available.
  2. Raises a clear ``ImportError`` when it isn't.
  3. Each ``on_llm_start`` → ``on_llm_end`` pair produces exactly one
     ``CallRecord`` in the Sentinel's tracer.
  4. Token usage flows through both ``llm_output["token_usage"]`` and
     per-generation ``usage_metadata``.
  5. Error paths produce a zero-token ``CallRecord`` so retry_storm can
     fire on repeated failures.
  6. Tool events round-trip with the tool name + arguments.
  7. Chain start/end manage the session-id bucket correctly.
  8. Multiple LLM calls under a single chain land in one session.
  9. Block-mode propagates ``LeakDetected`` through callback dispatch.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from token_sentinel import Sentinel
from token_sentinel.enrichers import TokenSentinelCallbackHandler
from token_sentinel.enrichers import langchain as enricher_module
from token_sentinel.events import CallRecord, LeakDetected, LeakEvent

# ---------------------------------------------------------------------------
# Helpers — build LLMResult-shaped responses without depending on LC at
# runtime. The handler only reads attributes (``llm_output``, ``generations``)
# so a SimpleNamespace is enough.
# ---------------------------------------------------------------------------


def _make_llm_result(
    *,
    llm_output: dict[str, Any] | None = None,
    text: str = "hi",
    tool_calls: list[dict[str, Any]] | None = None,
    usage_metadata: dict[str, int] | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=text,
        tool_calls=list(tool_calls or []),
        usage_metadata=usage_metadata,
    )
    generation = SimpleNamespace(
        message=message,
        text=text,
        generation_info=None,
    )
    return SimpleNamespace(
        llm_output=llm_output,
        generations=[[generation]],
    )


def _serialized(class_path: list[str], model: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": class_path, "name": class_path[-1]}
    if model is not None:
        payload["kwargs"] = {"model": model}
    return payload


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_handler_constructs_when_langchain_available() -> None:
    """The constructor succeeds when ``langchain_core`` is importable.

    Verifies the handler is a ``BaseCallbackHandler`` instance so LangChain
    accepts it in a ``callbacks=[...]`` config without a type sniff.
    """
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    # Must be a BaseCallbackHandler so LangChain accepts it.
    from langchain_core.callbacks import BaseCallbackHandler

    assert isinstance(handler, BaseCallbackHandler)
    assert handler.session_id  # auto-minted UUID


def test_handler_raises_clear_importerror_when_langchain_missing() -> None:
    """When ``_LANGCHAIN_AVAILABLE`` is False, instantiation raises with a
    hint pointing to ``pip install token-sentinel[langchain]``."""
    sentinel = Sentinel(project="proj")
    with (
        mock.patch.object(enricher_module, "_LANGCHAIN_AVAILABLE", False),
        pytest.raises(ImportError, match="langchain_core"),
    ):
        TokenSentinelCallbackHandler(sentinel)


def test_explicit_session_id_threaded_into_records() -> None:
    """Passing ``session_id=...`` puts every record under that bucket."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel, session_id="my-session")
    assert handler.session_id == "my-session"

    run_id = uuid.uuid4()
    handler.on_llm_start(
        _serialized(["langchain", "chat_models", "openai", "ChatOpenAI"], model="gpt-4o"),
        ["hello"],
        run_id=run_id,
    )
    handler.on_llm_end(
        _make_llm_result(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
        run_id=run_id,
    )
    records = sentinel.tracer.session("my-session")
    assert len(records) == 1
    assert records[0].session_id == "my-session"


# ---------------------------------------------------------------------------
# LLM start/end pairing → CallRecord
# ---------------------------------------------------------------------------


def test_on_llm_start_and_end_produce_one_call_record() -> None:
    """The basic happy path: start + end emits exactly one record with the
    provider sniffed from ``serialized["id"]`` and the model from kwargs."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    serialized = _serialized(
        ["langchain", "chat_models", "anthropic", "ChatAnthropic"],
        model="claude-sonnet-4-6",
    )
    handler.on_chat_model_start(
        serialized,
        [[SimpleNamespace(content="hi there")]],
        run_id=run_id,
    )
    response = _make_llm_result(
        llm_output={"token_usage": {"prompt_tokens": 42, "completion_tokens": 17}},
        text="ack",
    )
    handler.on_llm_end(response, run_id=run_id)

    sessions = list(sentinel.tracer.all_sessions())
    assert len(sessions) == 1
    records = sentinel.tracer.session(sessions[0])
    assert len(records) == 1
    rec: CallRecord = records[0]
    assert rec.provider == "anthropic"
    assert rec.model == "claude-sonnet-4-6"
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 17
    assert rec.method == "langchain.chat_model"
    assert rec.latency_ms >= 0
    assert rec.user_facing_output is True


def test_token_usage_extracted_from_llm_output_token_usage_shape() -> None:
    """OpenAI-style ``token_usage`` keys land in ``prompt_tokens`` /
    ``completion_tokens``."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    handler.on_llm_start(
        _serialized(["langchain", "llms", "openai", "OpenAI"], model="gpt-3.5-turbo"),
        ["q?"],
        run_id=run_id,
    )
    handler.on_llm_end(
        _make_llm_result(
            llm_output={
                "token_usage": {"prompt_tokens": 120, "completion_tokens": 33},
                "model_name": "gpt-3.5-turbo-instruct",
            }
        ),
        run_id=run_id,
    )
    rec = sentinel.tracer.session(handler.session_id)[0]
    assert rec.prompt_tokens == 120
    assert rec.completion_tokens == 33
    # llm_output["model_name"] is more specific than the bound model — the
    # handler prefers it.
    assert rec.model == "gpt-3.5-turbo-instruct"


def test_token_usage_falls_back_to_per_generation_usage_metadata() -> None:
    """LangChain >= 0.2 ``BaseMessage.usage_metadata`` is the standard
    fallback when ``llm_output`` is empty."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    handler.on_chat_model_start(
        _serialized(
            ["langchain", "chat_models", "anthropic", "ChatAnthropic"],
            model="claude-haiku-4-5",
        ),
        [[SimpleNamespace(content="hi")]],
        run_id=run_id,
    )
    response = _make_llm_result(
        llm_output=None,
        text="ok",
        usage_metadata={"input_tokens": 7, "output_tokens": 4},
    )
    handler.on_llm_end(response, run_id=run_id)
    rec = sentinel.tracer.session(handler.session_id)[0]
    assert rec.prompt_tokens == 7
    assert rec.completion_tokens == 4


def test_on_llm_error_records_zero_token_call() -> None:
    """Failure paths record a zero-token CallRecord with ``error`` populated.
    This lets the retry_storm rule fire on repeated failures."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    handler.on_llm_start(
        _serialized(["langchain", "chat_models", "openai", "ChatOpenAI"], model="gpt-4o"),
        ["hello"],
        run_id=run_id,
    )
    handler.on_llm_error(RuntimeError("upstream 500"), run_id=run_id)
    rec = sentinel.tracer.session(handler.session_id)[0]
    assert rec.prompt_tokens == 0
    assert rec.completion_tokens == 0
    assert rec.raw_response_meta["error"] == "RuntimeError"
    assert "upstream 500" in rec.raw_response_meta["error_message"]


def test_tool_start_and_end_recorded_as_tool_call() -> None:
    """Tool events produce a CallRecord with ``tool_calls=[{name, arguments}]``
    so the tool_loop rule sees them."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    handler.on_tool_start(
        {"name": "calculator", "id": ["langchain", "tools", "Calculator"]},
        "1 + 1",
        run_id=run_id,
        inputs={"expression": "1 + 1"},
    )
    handler.on_tool_end("2", run_id=run_id)

    rec = sentinel.tracer.session(handler.session_id)[0]
    assert rec.provider == "langchain.tool"
    assert rec.method == "langchain.tool"
    assert len(rec.tool_calls) == 1
    assert rec.tool_calls[0]["name"] == "calculator"
    assert rec.tool_calls[0]["arguments"] == {"expression": "1 + 1"}


# ---------------------------------------------------------------------------
# Chain bracketing + multi-call sessions
# ---------------------------------------------------------------------------


def test_chain_start_end_manage_top_level_tracking() -> None:
    """Top-level chain (``parent_run_id is None``) registers as the session
    boundary; ``on_chain_end`` for that same run clears it. Nested chains
    don't touch the top-level marker."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    top_run = uuid.uuid4()
    nested_run = uuid.uuid4()

    handler.on_chain_start(
        {"name": "AgentExecutor"},
        {"input": "hi"},
        run_id=top_run,
        parent_run_id=None,
    )
    assert handler._top_level_chain == top_run

    # Nested chain doesn't overwrite the top-level marker.
    handler.on_chain_start(
        {"name": "LLMChain"},
        {},
        run_id=nested_run,
        parent_run_id=top_run,
    )
    assert handler._top_level_chain == top_run

    handler.on_chain_end({}, run_id=nested_run, parent_run_id=top_run)
    assert handler._top_level_chain == top_run

    # Closing the top-level clears it.
    handler.on_chain_end({"output": "done"}, run_id=top_run, parent_run_id=None)
    assert handler._top_level_chain is None


def test_multiple_llm_calls_under_one_chain_share_session() -> None:
    """Three back-to-back LLM calls under one chain.invoke() land in the
    same session bucket so cross-call rules (retry_storm, tool_loop) work."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    chain_run = uuid.uuid4()
    handler.on_chain_start({"name": "AgentExecutor"}, {}, run_id=chain_run, parent_run_id=None)

    for i in range(3):
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            _serialized(["langchain", "chat_models", "openai", "ChatOpenAI"], model="gpt-4o"),
            [[SimpleNamespace(content=f"q{i}")]],
            run_id=run_id,
            parent_run_id=chain_run,
        )
        handler.on_llm_end(
            _make_llm_result(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 10 + i,
                        "completion_tokens": 5,
                    }
                }
            ),
            run_id=run_id,
            parent_run_id=chain_run,
        )

    handler.on_chain_end({}, run_id=chain_run)

    # All three records belong to the handler's single session.
    sessions = list(sentinel.tracer.all_sessions())
    assert sessions == [handler.session_id]
    records = sentinel.tracer.session(handler.session_id)
    assert len(records) == 3
    # Tokens increase as expected so we know we didn't accidentally
    # collapse the records into one.
    assert [r.prompt_tokens for r in records] == [10, 11, 12]


# ---------------------------------------------------------------------------
# Provider sniffing + edge cases
# ---------------------------------------------------------------------------


def test_provider_sniffer_falls_back_to_langchain() -> None:
    """Unknown class path → ``provider='langchain'`` (never crashes)."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    run_id = uuid.uuid4()
    handler.on_llm_start({"id": ["mystery", "FineTune"]}, [""], run_id=run_id)
    handler.on_llm_end(_make_llm_result(llm_output=None), run_id=run_id)
    rec = sentinel.tracer.session(handler.session_id)[0]
    assert rec.provider == "langchain"
    assert rec.model == "unknown"


def test_unmatched_on_llm_end_does_not_crash() -> None:
    """An end without a matching start (mid-run attach) is silently
    skipped — instrumentation must never crash callback dispatch."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel)

    # No prior on_llm_start — must not raise.
    handler.on_llm_end(_make_llm_result(), run_id=uuid.uuid4())
    handler.on_llm_error(RuntimeError("boom"), run_id=uuid.uuid4())
    handler.on_tool_end("x", run_id=uuid.uuid4())
    assert sentinel.tracer.session(handler.session_id) == []


def test_block_mode_propagates_leak_detected_through_callback() -> None:
    """In ``mode='block'`` a leak in the rule loop propagates as
    ``LeakDetected`` through the callback frame. LangChain's executor will
    surface this to the caller, same as the wrapper raise path."""
    sentinel = Sentinel(project="proj", mode="block")

    # Force record_call to raise — easier than wiring a real rule.
    fake_event = LeakEvent(
        type="tool_loop",
        confidence=0.9,
        project="proj",
        session_id="x",
        rule="tool_loop",
        evidence={},
        estimated_burn=0.001,
        suggested_action="halt",
    )

    with mock.patch.object(sentinel, "record_call", side_effect=LeakDetected(fake_event)):
        handler = TokenSentinelCallbackHandler(sentinel)
        run_id = uuid.uuid4()
        handler.on_llm_start(
            _serialized(["langchain", "chat_models", "openai", "ChatOpenAI"], model="gpt-4o"),
            ["hi"],
            run_id=run_id,
        )
        with pytest.raises(LeakDetected):
            handler.on_llm_end(_make_llm_result(), run_id=run_id)


def test_new_session_rotates_session_id() -> None:
    """``handler.new_session()`` rotates the bucket so subsequent records
    land in a fresh session — useful for worker handlers reused across jobs."""
    sentinel = Sentinel(project="proj")
    handler = TokenSentinelCallbackHandler(sentinel, session_id="job-1")

    run_id = uuid.uuid4()
    handler.on_llm_start(
        _serialized(["langchain", "llms", "openai", "OpenAI"], model="x"),
        ["a"],
        run_id=run_id,
    )
    handler.on_llm_end(_make_llm_result(), run_id=run_id)

    new_id = handler.new_session("job-2")
    assert new_id == "job-2"
    assert handler.session_id == "job-2"

    run_id_2 = uuid.uuid4()
    handler.on_llm_start(
        _serialized(["langchain", "llms", "openai", "OpenAI"], model="x"),
        ["b"],
        run_id=run_id_2,
    )
    handler.on_llm_end(_make_llm_result(), run_id=run_id_2)

    assert len(sentinel.tracer.session("job-1")) == 1
    assert len(sentinel.tracer.session("job-2")) == 1
