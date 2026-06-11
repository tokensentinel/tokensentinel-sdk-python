"""Tests for ``RerankThrashRule``.

 Cohere rerank-duplication rule. Fires when the same ``request_hash``
(keyed on (model, query, documents)) appears across ``>= N`` calls
within ``window_seconds`` (defaults: 2 calls / 30s) on
``provider == "cohere"`` ``method == "rerank"``. The Cohere wrapper
populates ``request_hash`` from the rerank inputs; tests construct
CallRecord directly with the desired hash.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from token_sentinel.events import CallRecord
from token_sentinel.rules.rerank_thrash import RerankThrashRule

_BASE_TS = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cohere_rerank_record(
    *,
    request_hash: str = "rerank-hash-A",
    provider: str = "cohere",
    method: str = "rerank",
    model: str = "rerank-english-v3.0",
    timestamp_offset_s: float = 0.0,
    query: str = "best restaurants",
    documents: list[str] | None = None,
) -> CallRecord:
    """Build a Cohere-rerank-shaped CallRecord.

    ``request_hash`` is the rule's key — tests vary it to model
    same-vs-different rerank inputs.
    """
    if documents is None:
        documents = ["restaurant A", "restaurant B", "restaurant C"]
    ts = _BASE_TS + timedelta(seconds=timestamp_offset_s)
    return CallRecord(
        session_id="rerank-session",
        timestamp=ts,
        provider=provider,
        model=model,
        method=method,
        prompt_tokens=len(query) + sum(len(d) for d in documents),
        completion_tokens=0,
        latency_ms=120.0,
        request_hash=request_hash,
        user_facing_output=False,
        raw_request={
            "query": query,
            "documents": documents,
            "model": model,
            "document_count": len(documents),
        },
    )


# ---------------------------------------------------------------------------
# 1. Fires on 2 identical reranks within window
# ---------------------------------------------------------------------------


def test_fires_on_two_identical_reranks_in_window() -> None:
    """Baseline positive: same request_hash twice in 30s default window."""
    rule = RerankThrashRule({})
    session = [
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=5),
    ]
    event = rule.evaluate(session, project="p")
    assert event is not None
    assert event.type == "rerank_thrash"
    assert event.rule == "v0.rerank_thrash"
    assert event.suggested_action == "cache_rerank_results_by_query_hash"
    assert event.evidence["call_count"] == 2


# ---------------------------------------------------------------------------
# 2. Doesn't fire on 1 rerank
# ---------------------------------------------------------------------------


def test_does_not_fire_on_single_call() -> None:
    """One rerank is not a thrash — must not fire."""
    rule = RerankThrashRule({})
    session = [_cohere_rerank_record(request_hash="abc123", timestamp_offset_s=0)]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 3. Doesn't fire on 2 different reranks (different hashes)
# ---------------------------------------------------------------------------


def test_does_not_fire_on_different_request_hashes() -> None:
    """Two reranks with DIFFERENT inputs (different request_hash) — no fire."""
    rule = RerankThrashRule({})
    session = [
        _cohere_rerank_record(request_hash="hash-A", timestamp_offset_s=0),
        _cohere_rerank_record(
            request_hash="hash-B",
            timestamp_offset_s=5,
            query="different question",
        ),
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 4. Doesn't fire outside the window
# ---------------------------------------------------------------------------


def test_does_not_fire_outside_window() -> None:
    """Two identical reranks spread across 60s — outside default 30s window."""
    rule = RerankThrashRule({})
    session = [
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=0),
        # Anchor is the latest at 60s; the first call at 0s is 60s old.
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=60),
    ]
    # Only the second call is in-window — count=1 < 2.
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 5. Doesn't fire on non-cohere CallRecords
# ---------------------------------------------------------------------------


def test_does_not_fire_on_non_cohere_provider() -> None:
    """Provider gate: identically-hashed non-cohere records must not fire.

    A future ``voyage rerank`` or ``jinaai rerank`` would have its own
    rule; this one is Cohere-only.
    """
    rule = RerankThrashRule({})
    session = [
        _cohere_rerank_record(request_hash="abc123", provider="voyage", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", provider="voyage", timestamp_offset_s=5),
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 6. Doesn't fire on non-rerank methods (chat / embed)
# ---------------------------------------------------------------------------


def test_does_not_fire_on_non_rerank_methods() -> None:
    """Method gate: chat / embed must not trigger this rule even with
    duplicate request_hash. ``retry_storm`` handles general chat dedup."""
    rule = RerankThrashRule({})
    session = [
        _cohere_rerank_record(request_hash="abc123", method="chat", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", method="chat", timestamp_offset_s=5),
    ]
    assert rule.evaluate(session, project="p") is None

    session_embed = [
        _cohere_rerank_record(request_hash="abc123", method="embed", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", method="embed", timestamp_offset_s=5),
    ]
    assert rule.evaluate(session_embed, project="p") is None


# ---------------------------------------------------------------------------
# 7. Confidence scaling — +0.05 per extra repetition beyond N, capped 0.9
# ---------------------------------------------------------------------------


def test_confidence_scales_with_repetition_and_caps_at_0_9() -> None:
    """Confidence: 0.75 baseline at N=2, +0.05 per extra call, cap 0.9."""
    rule = RerankThrashRule({})

    def session_with_n_calls(n: int) -> list[CallRecord]:
        return [
            _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=float(i))
            for i in range(n)
        ]

    ev_2 = rule.evaluate(session_with_n_calls(2), project="p")
    assert ev_2 is not None
    assert ev_2.confidence == 0.75

    ev_3 = rule.evaluate(session_with_n_calls(3), project="p")
    assert ev_3 is not None
    assert ev_3.confidence == 0.8

    ev_5 = rule.evaluate(session_with_n_calls(5), project="p")
    assert ev_5 is not None
    assert ev_5.confidence == 0.9  # 0.75 + 3*0.05 = 0.9 at exactly the cap

    ev_10 = rule.evaluate(session_with_n_calls(10), project="p")
    assert ev_10 is not None
    assert ev_10.confidence == 0.9  # capped


# ---------------------------------------------------------------------------
# 8. Evidence shape — request_hash, call_count, time_span_seconds, model
# ---------------------------------------------------------------------------


def test_evidence_contains_all_required_fields() -> None:
    """Evidence dict must carry request_hash, call_count, time_span_seconds,
    and model — dashboard renders them verbatim."""
    rule = RerankThrashRule({})
    long_hash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
    session = [
        _cohere_rerank_record(
            request_hash=long_hash,
            model="rerank-english-v3.0",
            timestamp_offset_s=0,
        ),
        _cohere_rerank_record(
            request_hash=long_hash,
            model="rerank-english-v3.0",
            timestamp_offset_s=12,
        ),
    ]
    event = rule.evaluate(session, project="p")
    assert event is not None
    ev = event.evidence
    # request_hash is truncated to first 16 chars (privacy-preserving
    # convention; mirrors retry_storm).
    assert ev["request_hash"] == long_hash[:16]
    assert ev["call_count"] == 2
    assert ev["time_span_seconds"] == 12.0
    assert ev["model"] == "rerank-english-v3.0"


# ---------------------------------------------------------------------------
# 9. Custom window_seconds + min_calls config
# ---------------------------------------------------------------------------


def test_custom_window_seconds_and_min_calls_config() -> None:
    """Both ``window_seconds`` and ``min_calls`` honor config overrides."""
    # Narrow window: only the last 5 seconds count. Two identical
    # reranks 10s apart no longer fire.
    rule_narrow = RerankThrashRule({"rerank_thrash.window_seconds": 5})
    session = [
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=10),
    ]
    assert rule_narrow.evaluate(session, project="p") is None

    # Stricter min_calls: now we need 4 identical calls to fire.
    rule_strict = RerankThrashRule({"rerank_thrash.min_calls": 4})
    session_3 = [
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=float(i)) for i in range(3)
    ]
    # 3 < 4 threshold — must not fire.
    assert rule_strict.evaluate(session_3, project="p") is None

    # Add a 4th and it fires.
    session_4 = session_3 + [_cohere_rerank_record(request_hash="abc123", timestamp_offset_s=3)]
    ev = rule_strict.evaluate(session_4, project="p")
    assert ev is not None
    assert ev.evidence["call_count"] == 4


# ---------------------------------------------------------------------------
# 10. Defensive — empty session, missing request_hash, mixed cluster
# ---------------------------------------------------------------------------


def test_defensive_edge_cases() -> None:
    """Empty session, missing request_hash, and mixed providers in one
    session all short-circuit cleanly."""
    rule = RerankThrashRule({})

    # Empty session.
    assert rule.evaluate([], project="p") is None

    # Mixed: one cohere rerank + two anthropic chats with same hash.
    # Provider gate filters out the chats; only 1 cohere rerank remains
    # in the cluster.
    session_mixed = [
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=0),
        _cohere_rerank_record(
            request_hash="abc123",
            provider="anthropic",
            method="messages.create",
            timestamp_offset_s=1,
        ),
        _cohere_rerank_record(
            request_hash="abc123",
            provider="anthropic",
            method="messages.create",
            timestamp_offset_s=2,
        ),
    ]
    assert rule.evaluate(session_mixed, project="p") is None

    # Two cohere reranks but one has an empty request_hash — filtered out.
    session_one_empty_hash = [
        _cohere_rerank_record(request_hash="", timestamp_offset_s=0),
        _cohere_rerank_record(request_hash="abc123", timestamp_offset_s=2),
    ]
    assert rule.evaluate(session_one_empty_hash, project="p") is None
