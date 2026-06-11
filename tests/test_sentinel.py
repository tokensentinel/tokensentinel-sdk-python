"""Tests for the ``Sentinel`` public API."""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from token_sentinel import LeakDetected, LeakEvent, Sentinel
from token_sentinel.events import CallRecord

# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_default_construction():
    s = Sentinel(project="proj")
    assert s.project == "proj"
    assert s.mode == "log"
    assert s.min_confidence == 0.5
    # 8 rules + 3  vision rules + 1 audio rule + 2
    # rules + 1  rule (repair_loop) = 15.
    assert len(s._rules) == 15
    assert s._handlers == []
    assert s.tracer is not None


def test_rules_all_loaded():
    s = Sentinel(project="proj", rules="all")
    names = sorted(r.name for r in s._rules)
    assert names == sorted(
        [
            "tool_loop",
            "context_bloat",
            "embedding_waste",
            "zombie",
            "model_misroute",
            "retry_storm",
            "tool_definition_bloat",
            "retrieval_thrash",
            # vision rules.
            "vision_re_upload",
            "vision_high_detail_misroute",
            "vision_cost_concentration",
            # audio rule.
            "audio_multichannel_doubling",
            # rules.
            "voice_switching_loop",
            "rerank_thrash",
            # rules.
            "repair_loop",
        ]
    )


def test_rules_subset_loaded():
    s = Sentinel(project="proj", rules=["tool_loop", "retry_storm"])
    names = {r.name for r in s._rules}
    assert names == {"tool_loop", "retry_storm"}


def test_rules_empty_list():
    """Empty list means no rules — nothing fires regardless of input."""
    s = Sentinel(project="proj", rules=[])
    assert s._rules == []


def test_rules_unknown_name_dropped_with_warning():
    """An unknown rule name is dropped from the list — but emits a UserWarning
    so the customer notices the typo (UserWarning). Construction does not
    raise; the silent-drop behaviour is preserved for backwards compat.
    """
    with pytest.warns(UserWarning, match="unknown rule names"):
        s = Sentinel(project="proj", rules=["tool_loop", "fake_rule"])
    names = {r.name for r in s._rules}
    assert names == {"tool_loop"}


def test_config_passed_to_rules():
    """Customer config flows through to each rule."""
    cfg = {"tool_loop.cosine_threshold": 0.99, "context_bloat.slope_threshold": 5}
    s = Sentinel(project="proj", config=cfg)
    for rule in s._rules:
        assert rule.config is cfg


# ---------------------------------------------------------------------------
# wrap() dispatch
# ---------------------------------------------------------------------------


def test_wrap_anthropic_module_dispatches(mock_anthropic_client):
    """Mock client whose type's __module__ starts with 'anthropic' is wrapped."""
    s = Sentinel(project="proj")
    original = mock_anthropic_client.messages.create
    wrapped = s.wrap(mock_anthropic_client)
    # wrap_anthropic mutates and returns the same instance.
    assert wrapped is mock_anthropic_client
    # messages.create has been replaced with the instrumented function.
    assert mock_anthropic_client.messages.create is not original
    assert callable(mock_anthropic_client.messages.create)


def test_wrap_openai_module_dispatches(mock_openai_client):
    """Mock OpenAI client whose type's __module__ starts with 'openai' is wrapped."""
    s = Sentinel(project="proj")
    original = mock_openai_client.chat.completions.create
    wrapped = s.wrap(mock_openai_client)
    # wrap_openai mutates and returns the same instance.
    assert wrapped is mock_openai_client
    # chat.completions.create has been replaced with the instrumented function.
    assert mock_openai_client.chat.completions.create is not original
    assert callable(mock_openai_client.chat.completions.create)


def test_wrap_unsupported_client_raises_typeerror():
    """A non-LLM-SDK object should be rejected with TypeError."""
    s = Sentinel(project="proj")

    class SomeRandomClient:
        pass

    with pytest.raises(TypeError) as exc:
        s.wrap(SomeRandomClient())
    assert "Unsupported client type" in str(exc.value)


# ---------------------------------------------------------------------------
# on_leak() decorator
# ---------------------------------------------------------------------------


def test_on_leak_registers_handler():
    s = Sentinel(project="proj")

    @s.on_leak
    def handler(ev):
        pass

    assert handler in s._handlers
    assert len(s._handlers) == 1


def test_on_leak_returns_handler():
    """Decorator must return the function unchanged so ``@`` works idiomatically."""
    s = Sentinel(project="proj")

    def h(ev):
        return ev

    out = s.on_leak(h)
    assert out is h


def test_multiple_handlers_all_fire(make_call, now):
    s = Sentinel(project="proj", rules=["embedding_waste"])
    seen_a = []
    seen_b = []
    s.on_leak(lambda ev: seen_a.append(ev))
    s.on_leak(lambda ev: seen_b.append(ev))

    # Cause a leak: two identical embedding calls.
    for i in range(2):
        s.record_call(
            CallRecord(
                session_id="s1",
                timestamp=now + timedelta(seconds=i),
                provider="openai",
                model="text-embedding-3-small",
                method="embeddings.create",
                prompt_tokens=10,
                completion_tokens=0,
                latency_ms=20.0,
                request_hash=hashlib.sha256(b"x").hexdigest(),
                tool_calls=[],
                user_facing_output=False,
                raw_request={"input": "hello"},
            )
        )

    assert len(seen_a) == 1
    assert len(seen_b) == 1
    assert seen_a[0] is seen_b[0]


def test_handler_exception_does_not_kill_dispatch(make_call, now):
    """A throwing handler must not stop other handlers or kill the agent."""
    s = Sentinel(project="proj", rules=["embedding_waste"])
    seen = []

    @s.on_leak
    def bad(ev):
        raise RuntimeError("boom")

    @s.on_leak
    def good(ev):
        seen.append(ev)

    for i in range(2):
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# mode='block'
# ---------------------------------------------------------------------------


def test_block_mode_raises_leak_detected(make_call, now):
    s = Sentinel(project="proj", mode="block", rules=["embedding_waste"])
    s.record_call(
        make_call(
            method="embeddings.create",
            timestamp=now,
            raw_request={"input": "x"},
        )
    )
    # Second identical embed → fires → block mode raises.
    with pytest.raises(LeakDetected) as exc:
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=1),
                raw_request={"input": "x"},
            )
        )
    assert isinstance(exc.value.event, LeakEvent)
    assert exc.value.event.type == "embedding_waste"


def test_log_mode_does_not_raise(make_call, now):
    s = Sentinel(project="proj", mode="log", rules=["embedding_waste"])
    for i in range(2):
        # Should not raise even though a leak fires.
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )


def test_alert_mode_does_not_raise(make_call, now):
    """alert mode emits to handlers but does not raise."""
    s = Sentinel(project="proj", mode="alert", rules=["embedding_waste"])
    for i in range(2):
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )


# ---------------------------------------------------------------------------
# min_confidence filtering
# ---------------------------------------------------------------------------


def test_min_confidence_filters_low_confidence(make_call, now):
    """Bump min_confidence high enough that no rule's max can clear it."""
    s = Sentinel(project="proj", min_confidence=1.0, rules=["embedding_waste"])
    seen = []
    s.on_leak(lambda ev: seen.append(ev))
    for i in range(2):
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )
    # embedding_waste fires at 0.99 — below 1.0 threshold.
    assert seen == []


def test_min_confidence_passes_high_confidence(make_call, now):
    """At min_confidence=0.0, even the lowest-confidence event passes."""
    s = Sentinel(project="proj", min_confidence=0.0, rules=["embedding_waste"])
    seen = []
    s.on_leak(lambda ev: seen.append(ev))
    for i in range(2):
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )
    assert len(seen) == 1


def test_record_call_returns_event_list(make_call, now):
    """``record_call`` returns the list of fired events."""
    s = Sentinel(project="proj", rules=["embedding_waste"])
    events_first = s.record_call(
        make_call(method="embeddings.create", timestamp=now, raw_request={"input": "x"})
    )
    assert events_first == []
    events_second = s.record_call(
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=1),
            raw_request={"input": "x"},
        )
    )
    assert len(events_second) == 1
    assert events_second[0].type == "embedding_waste"


# ---------------------------------------------------------------------------
# Rule exception isolation
# ---------------------------------------------------------------------------


def test_rule_exception_does_not_kill_pipeline(make_call, now, monkeypatch):
    """If one rule raises, the others still run."""
    s = Sentinel(project="proj", rules=["tool_loop", "embedding_waste"])

    # Find tool_loop rule and patch its evaluate to throw.
    target = next(r for r in s._rules if r.name == "tool_loop")

    def boom(*a, **k):
        raise ValueError("rule explodes")

    monkeypatch.setattr(target, "evaluate", boom)

    seen = []
    s.on_leak(lambda ev: seen.append(ev))
    # Embedding waste should still fire.
    for i in range(2):
        s.record_call(
            make_call(
                method="embeddings.create",
                timestamp=now + timedelta(seconds=i),
                raw_request={"input": "x"},
            )
        )
    assert len(seen) == 1
    assert seen[0].type == "embedding_waste"


# ---------------------------------------------------------------------------
# Session isolation across record_call
# ---------------------------------------------------------------------------


def test_two_sessions_isolated(make_call, now):
    s = Sentinel(project="proj", rules=["embedding_waste"])
    # One identical embed in each of two sessions — neither should fire because
    # there's only one duplicate per session.
    s.record_call(
        make_call(
            session_id="A",
            method="embeddings.create",
            timestamp=now,
            raw_request={"input": "x"},
        )
    )
    s.record_call(
        make_call(
            session_id="B",
            method="embeddings.create",
            timestamp=now + timedelta(seconds=1),
            raw_request={"input": "x"},
        )
    )
    # Now duplicate in A only.
    events = s.record_call(
        make_call(
            session_id="A",
            method="embeddings.create",
            timestamp=now + timedelta(seconds=2),
            raw_request={"input": "x"},
        )
    )
    assert len(events) == 1
    assert events[0].session_id == "A"
