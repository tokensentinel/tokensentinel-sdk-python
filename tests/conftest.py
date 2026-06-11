"""Shared pytest fixtures for the TokenSentinel test suite.

This conftest also pins ``sdk/python`` onto ``sys.path`` so tests can be run
from the repo root with a plain ``pytest tests/`` invocation, regardless of
whether the package has been ``pip install -e``'d.
"""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Make ``token_sentinel`` importable when running ``pytest tests/`` from the
# package root in a fresh checkout (the package lives in the repository root).
_SDK_ROOT = Path(__file__).resolve().parent.parent
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))

from token_sentinel.events import CallRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    """A fixed deterministic ``datetime`` used as the anchor for sessions."""
    return datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# CallRecord factory
# ---------------------------------------------------------------------------


def _build_call(
    *,
    session_id: str = "test-session",
    timestamp: datetime | None = None,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    method: str = "messages.create",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    latency_ms: float = 120.0,
    request_hash: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    user_facing_output: bool = False,
    raw_request: dict[str, Any] | None = None,
    raw_response_meta: dict[str, Any] | None = None,
) -> CallRecord:
    """Construct a ``CallRecord`` with sane defaults."""
    if timestamp is None:
        timestamp = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
    if request_hash is None:
        request_hash = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    return CallRecord(
        session_id=session_id,
        timestamp=timestamp,
        provider=provider,
        model=model,
        method=method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        request_hash=request_hash,
        tool_calls=tool_calls or [],
        user_facing_output=user_facing_output,
        raw_request=raw_request or {},
        raw_response_meta=raw_response_meta or {},
    )


@pytest.fixture
def make_call():
    """Factory fixture for ``CallRecord`` instances.

    Use this in tests instead of constructing ``CallRecord`` by hand:

    .. code-block:: python

        def test_something(make_call, now):
            call = make_call(prompt_tokens=500, timestamp=now)
    """
    return _build_call


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_session(now):
    """A small session of three innocuous user-facing calls."""
    return [
        _build_call(
            session_id="sample",
            timestamp=now + timedelta(seconds=i * 5),
            user_facing_output=True,
            prompt_tokens=500 + i * 50,
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Mock Anthropic response
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_tool_use_block(name: str, input_: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_)


@pytest.fixture
def mock_anthropic_response():
    """Factory for a mock ``anthropic.Anthropic`` ``messages.create`` response.

    Returns objects shaped like the real Anthropic SDK's response: ``content``
    is a list of blocks, ``usage`` exposes ``input_tokens``/``output_tokens``,
    ``stop_reason`` is a string.
    """

    def _make(
        *,
        input_tokens: int = 100,
        output_tokens: int = 25,
        stop_reason: str = "end_turn",
        text_blocks: list[str] | None = None,
        tool_uses: list[dict[str, Any]] | None = None,
    ) -> SimpleNamespace:
        content: list[Any] = []
        if text_blocks:
            content.extend(_make_text_block(t) for t in text_blocks)
        if tool_uses:
            content.extend(_make_tool_use_block(t["name"], t.get("input", {})) for t in tool_uses)
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    return _make


# ---------------------------------------------------------------------------
# Mock Anthropic client (whole client, not just response)
# ---------------------------------------------------------------------------


class _RecordingCreate:
    """Real callable that records calls and returns a configurable response.

    We use a plain class instead of MagicMock because ``functools.wraps`` (used
    inside ``wrap_anthropic``) tries to copy ``__name__``/``__qualname__`` from
    the wrapped object onto the wrapper. A MagicMock's auto-generated child
    mocks for those attributes are not strings, so Python's function-attribute
    type check raises ``TypeError``.
    """

    __name__ = "create"
    __qualname__ = "Messages.create"
    __module__ = "anthropic.resources.messages"
    __annotations__: dict = {}
    __doc__ = "mock create"

    def __init__(self):
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

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        actual_args, actual_kwargs = self.calls[0]
        assert actual_args == args, f"args mismatch: {actual_args} != {args}"
        assert actual_kwargs == kwargs, f"kwargs mismatch: {actual_kwargs} != {kwargs}"

    @property
    def call_args(self):
        if not self.calls:
            return None
        args, kwargs = self.calls[-1]
        return SimpleNamespace(args=args, kwargs=kwargs)


@pytest.fixture
def mock_anthropic_client():
    """A fake Anthropic client whose type reports ``__module__='anthropic'``.

    The Sentinel.wrap() dispatch decision is purely module-string based:
    ``type(client).__module__.startswith("anthropic")``. We construct a class
    in the ``anthropic`` module namespace so ``Sentinel.wrap`` routes to
    ``wrap_anthropic``.

    ``messages.create`` is a real callable (``_RecordingCreate``) instead of a
    ``MagicMock`` because ``functools.wraps`` inside ``wrap_anthropic`` cannot
    copy a MagicMock's auto-generated ``__name__`` onto a real function.
    """
    fake_anthropic_class = type("Anthropic", (), {"__module__": "anthropic"})
    client = fake_anthropic_class()
    create = _RecordingCreate()
    client.messages = SimpleNamespace(create=create)
    return client


@pytest.fixture
def mock_openai_client():
    """A fake sync OpenAI client whose type reports ``__module__='openai'``.

    Mirrors ``mock_anthropic_client`` for the OpenAI dispatch path. Provides
    `chat.completions.create` and `embeddings.create` as real callables that
    `wrap_openai` can `functools.wraps` over.
    """
    fake_openai_class = type("OpenAI", (), {"__module__": "openai"})
    client = fake_openai_class()
    chat_create = _RecordingCreate()
    chat_create.__module__ = "openai.resources.chat.completions"
    embed_create = _RecordingCreate()
    embed_create.__module__ = "openai.resources.embeddings"
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=chat_create))
    client.embeddings = SimpleNamespace(create=embed_create)
    return client


# ---------------------------------------------------------------------------
# Hash helper used in retry-storm fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stable_hash():
    """Compute a deterministic SHA-256 hex digest from a string."""

    def _hash(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()

    return _hash


@pytest.fixture
def request_hash_for():
    """Hash a (model, messages, tools, max_tokens) tuple the way the wrapper does."""

    def _hash(
        *,
        model: str = "claude-sonnet-4-6",
        messages: list | None = None,
        tools: list | None = None,
        max_tokens: int = 0,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "model": model,
                    "messages": messages or [],
                    "tools": tools or [],
                    "max_tokens": max_tokens,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    return _hash
