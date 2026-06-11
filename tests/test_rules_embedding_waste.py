"""Tests for ``EmbeddingWasteRule``.

Behavior: same SHA-256 of the embedding ``input`` argument repeated within a
session fires immediately at confidence 0.99.
"""

from __future__ import annotations

from datetime import timedelta

from token_sentinel.rules.embedding_waste import EmbeddingWasteRule

# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_empty_session_returns_none():
    assert EmbeddingWasteRule({}).evaluate([], project="p") is None


def test_single_embedding_call_no_fire(make_call, now):
    """A single embedding call cannot duplicate anything."""
    session = [
        make_call(
            method="v1.embeddings.create",
            model="text-embedding-3-small",
            timestamp=now,
            raw_request={"input": "hello"},
            prompt_tokens=10,
        )
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


def test_two_distinct_embeddings_no_fire(make_call, now):
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": payload},
        )
        for i, payload in enumerate(["hello", "world"])
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


def test_non_embedding_calls_ignored(make_call, now):
    """messages.create calls don't count even if input matches."""
    session = [
        make_call(
            method="messages.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": "hello"},
        )
        for i in range(5)
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


def test_embedding_calls_without_input_ignored(make_call, now):
    """``raw_request`` lacking ``input`` is skipped silently."""
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={},
        )
        for i in range(3)
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


def test_one_with_input_one_without(make_call, now):
    """One valid + one missing input — only one counted, no duplicate."""
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now,
            raw_request={"input": "hello"},
        ),
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=1),
            raw_request={},
        ),
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_two_identical_embeddings_fire(make_call, now):
    session = [
        make_call(
            method="embeddings.create",
            model="text-embedding-3-small",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": "user query: top 5 movies"},
            prompt_tokens=12,
        )
        for i in range(2)
    ]
    ev = EmbeddingWasteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.type == "embedding_waste"
    assert ev.rule == "v0.embedding_waste"
    assert ev.confidence == 0.99
    assert ev.evidence["duplicate_count"] == 2
    assert ev.evidence["model"] == "text-embedding-3-small"
    assert ev.evidence["wasted_tokens"] == 12  # only the second is wasted
    assert ev.suggested_action == "add_embedding_cache"
    assert len(ev.evidence["input_hash"]) == 16  # truncated digest


def test_method_suffix_match(make_call, now):
    """Rule uses ``method.endswith('embeddings.create')`` — versioned methods match."""
    session = [
        make_call(
            method="v1.embeddings.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": "same"},
        )
        for i in range(2)
    ]
    ev = EmbeddingWasteRule({}).evaluate(session, project="p")
    assert ev is not None


def test_three_duplicates_wasted_token_count(make_call, now):
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": "x"},
            prompt_tokens=20,
        )
        for i in range(3)
    ]
    ev = EmbeddingWasteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["duplicate_count"] == 3
    # First call is the canonical, the next two are wasted: 20 + 20 = 40
    assert ev.evidence["wasted_tokens"] == 40


def test_duplicate_input_as_list(make_call, now):
    """``input`` may be a list (batch). Sort-keys serialization should still match."""
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=i),
            raw_request={"input": ["one", "two", "three"]},
            prompt_tokens=10,
        )
        for i in range(2)
    ]
    ev = EmbeddingWasteRule({}).evaluate(session, project="p")
    assert ev is not None


def test_lists_in_different_order_treated_as_distinct(make_call, now):
    """A list with a different order has a different hash — not flagged."""
    session = [
        make_call(
            method="embeddings.create",
            timestamp=now,
            raw_request={"input": ["a", "b", "c"]},
        ),
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=1),
            raw_request={"input": ["c", "b", "a"]},
        ),
    ]
    assert EmbeddingWasteRule({}).evaluate(session, project="p") is None


def test_intermixed_with_non_embedding_calls(make_call, now):
    """Embedding waste is detected even with chat calls interleaved."""
    session = [
        make_call(method="messages.create", timestamp=now),
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=1),
            raw_request={"input": "x"},
        ),
        make_call(method="messages.create", timestamp=now + timedelta(seconds=2)),
        make_call(
            method="embeddings.create",
            timestamp=now + timedelta(seconds=3),
            raw_request={"input": "x"},
        ),
    ]
    ev = EmbeddingWasteRule({}).evaluate(session, project="p")
    assert ev is not None
    assert ev.evidence["duplicate_count"] == 2
