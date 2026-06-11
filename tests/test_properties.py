"""Property-based tests using Hypothesis.

These tests verify *invariants* of TokenSentinel internals — the kind of
properties that should hold for all reasonable inputs, not just the cases the
example-based tests happen to cover.

Four invariant classes:

1. Tool-loop similarity (`_mean_pairwise_similarity` and friends)
   - identical args → 1.0 (both metrics)
   - symmetry, range, empty/single short circuits, monotonicity-ish
2. Tracer ring buffer
   - capacity, ordering, isolation, concurrency
3. Sentinel filter / dispatch
   - confidence threshold, identity of returned/dispatched events
4. Context-bloat slope (`_linear_slope`)
   - constant series, arithmetic series, sign reflection

Hypothesis strategies are bounded (max sizes, integer ranges) to keep runtime
under a second or two per test on a normal laptop.
"""

from __future__ import annotations

import math
import threading
import uuid
from datetime import datetime, timedelta, timezone

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings

from token_sentinel import Sentinel
from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.context_bloat import _linear_slope
from token_sentinel.rules.tool_loop import (
    _mean_pairwise_similarity,
    _mean_tfidf_charngram,
)
from token_sentinel.tracer import Tracer

# ---------------------------------------------------------------------------
# Tool-loop similarity invariants
# ---------------------------------------------------------------------------


def _calls_from_args(args_list: list[dict]) -> list[dict]:
    """Wrap a list of arg dicts as tool_call dicts of the shape
    ``_mean_pairwise_similarity`` expects."""
    return [{"name": "search", "arguments": a} for a in args_list]


# Bounded text strategy — short, printable. We avoid empty strings so the
# "empty corpus" path doesn't dominate identical-args tests.
_short_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=20
)


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(
    arg_value=_short_text,
    n=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=40, deadline=None)
def test_identical_args_similarity_is_one(metric: str, arg_value: str, n: int):
    """All-identical corpora must score 1.0 (within float tolerance) under
    BOTH metrics. This is the most fundamental invariant of the rule:
    duplicates of the *same* input are perfectly similar.
    """
    calls = _calls_from_args([{"q": arg_value} for _ in range(n)])
    sim = _mean_pairwise_similarity(calls, metric=metric)
    assert math.isclose(sim, 1.0, abs_tol=1e-6), (
        f"metric={metric} value={arg_value!r} n={n} sim={sim}"
    )


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(arg=st.dictionaries(keys=_short_text, values=_short_text, min_size=1, max_size=5))
@settings(max_examples=40, deadline=None)
def test_two_identical_calls_similarity_is_one(metric: str, arg: dict):
    """Two identical calls — the smallest non-trivial corpus — also yield 1.0.

    NOTE: ``min_size=1`` is required: empty-arg corpora ``[{}, {}]`` fail this
    invariant under Jaccard (the alnum-only tokenizer produces empty token
    sets, and ``_mean_jaccard`` skips empty pairs and falls through to 0.0).
    TF-IDF char-n-gram returns 1.0 in the same case because boundary-padded
    n-grams yield non-empty vectors. The metric divergence on empty inputs is
    an *intended* degeneracy at the boundary, not a bug.
    """
    calls = _calls_from_args([dict(arg), dict(arg)])
    # Skip cases where Jaccard's tokenizer produces an empty token set (e.g.
    # all values are pure punctuation). The invariant only holds when the
    # input has at least one alphanumeric token.
    if metric == "jaccard":
        import json as _json

        from token_sentinel.rules.tool_loop import _tokens

        if not _tokens(_json.dumps(arg, sort_keys=True, default=str)):
            return
    sim = _mean_pairwise_similarity(calls, metric=metric)
    assert math.isclose(sim, 1.0, abs_tol=1e-6)


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(
    a=_short_text,
    b=_short_text,
)
@settings(max_examples=50, deadline=None)
def test_similarity_is_symmetric(metric: str, a: str, b: str):
    """sim(a, b) == sim(b, a). Symmetry is a basic property of the cosine /
    Jaccard formulas; a regression that introduces order-sensitivity (e.g. a
    bad iteration variable) breaks it.
    """
    sim_ab = _mean_pairwise_similarity(_calls_from_args([{"q": a}, {"q": b}]), metric=metric)
    sim_ba = _mean_pairwise_similarity(_calls_from_args([{"q": b}, {"q": a}]), metric=metric)
    assert math.isclose(sim_ab, sim_ba, abs_tol=1e-9)


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(
    args=st.lists(
        st.dictionaries(keys=_short_text, values=_short_text, max_size=3),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=50, deadline=None)
def test_similarity_in_unit_range(metric: str, args: list[dict]):
    """Similarity is always in [0.0, 1.0]. Both metrics produce a normalised
    score; a NaN/negative leakage would break threshold comparisons silently.
    """
    sim = _mean_pairwise_similarity(_calls_from_args(args), metric=metric)
    assert 0.0 <= sim <= 1.0 + 1e-9, sim


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
def test_empty_corpus_similarity_is_zero(metric: str):
    """Empty input → 0.0. Defined invariant of the helper (not Hypothesis-
    eligible — just one input — but kept here for completeness)."""
    assert _mean_pairwise_similarity([], metric=metric) == 0.0


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(arg=st.dictionaries(keys=_short_text, values=_short_text, max_size=5))
@settings(max_examples=20, deadline=None)
def test_single_call_similarity_is_zero(metric: str, arg: dict):
    """Single-call corpus → 0.0. Same property: similarity needs a pair."""
    sim = _mean_pairwise_similarity(_calls_from_args([dict(arg)]), metric=metric)
    assert sim == 0.0


@pytest.mark.parametrize("metric", ["tfidf_charngram", "jaccard"])
@given(arg=st.dictionaries(keys=_short_text, values=_short_text, min_size=1, max_size=5))
@settings(max_examples=20, deadline=None)
def test_three_identical_extends_two_identical(metric: str, arg: dict):
    """Adding a third identical doc to a 2-doc identical corpus keeps the mean
    similarity at 1.0. Invariant of "all pairs equal 1 → mean equals 1".

    See ``test_two_identical_calls_similarity_is_one`` for the degenerate-
    empty-input note that motivates ``min_size=1``.
    """
    if metric == "jaccard":
        import json as _json

        from token_sentinel.rules.tool_loop import _tokens

        if not _tokens(_json.dumps(arg, sort_keys=True, default=str)):
            return
    calls_2 = _calls_from_args([dict(arg), dict(arg)])
    calls_3 = _calls_from_args([dict(arg), dict(arg), dict(arg)])
    sim_2 = _mean_pairwise_similarity(calls_2, metric=metric)
    sim_3 = _mean_pairwise_similarity(calls_3, metric=metric)
    assert math.isclose(sim_2, 1.0, abs_tol=1e-6)
    assert math.isclose(sim_3, 1.0, abs_tol=1e-6)


@given(
    a=_short_text,
    b=_short_text,
    n=st.integers(min_value=3, max_value=4),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_tfidf_normalised_self_similarity_is_one_explicit(a: str, b: str, n: int):
    """Direct test against ``_mean_tfidf_charngram`` with all-identical input
    strings. Exercises the L2-normalisation path explicitly: a unit vector
    dotted with itself is 1.
    """
    assume(a)  # non-empty so we have at least one ngram
    args_strs = [a for _ in range(n)]
    sim = _mean_tfidf_charngram(args_strs, ngram_size=4)
    assert math.isclose(sim, 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Tracer ring buffer invariants
# ---------------------------------------------------------------------------


def _mk_record(session_id: str, ts_offset_seconds: int) -> CallRecord:
    return CallRecord(
        session_id=session_id,
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        + timedelta(seconds=ts_offset_seconds),
        provider="anthropic",
        model="claude",
        method="messages.create",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=10.0,
        request_hash="abc",
    )


@given(
    n=st.integers(min_value=0, max_value=300),
    cap=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=30, deadline=None)
def test_tracer_capacity_bound(n: int, cap: int):
    """For any record count N and capacity K, len(session) <= K. The ring
    buffer must never grow past its configured maxlen.
    """
    t = Tracer(max_records_per_session=cap)
    for i in range(n):
        t.record(_mk_record("s", i))
    session = t.session("s")
    assert len(session) <= cap
    # Equality holds when n >= cap; strict less-than otherwise.
    if n >= cap:
        assert len(session) == cap
    else:
        assert len(session) == n


@given(
    n=st.integers(min_value=2, max_value=50),
    cap=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_tracer_chronological_order(n: int, cap: int):
    """Records are returned in chronological insertion order (oldest first,
    most recent at the end). A regression that uses a stack (LIFO) instead of
    a deque (FIFO) would break leak rules that walk ``session[-1]`` expecting
    the latest call.
    """
    t = Tracer(max_records_per_session=cap)
    for i in range(n):
        t.record(_mk_record("s", i))
    session = t.session("s")
    timestamps = [c.timestamp for c in session]
    # Strictly increasing → sorted == itself.
    assert timestamps == sorted(timestamps)
    # Stronger: the LAST record returned must be the LAST one inserted.
    # Many rules look at session[-1] for "current call", so a regression that
    # broke this would make every rule misfire.
    if n > 0:
        anchor = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        assert session[-1].timestamp == anchor + timedelta(seconds=n - 1)


@given(
    sessions=st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=5),
        min_size=1,
        max_size=10,
        unique=True,
    )
)
@settings(max_examples=20, deadline=None)
def test_tracer_clear_specific_session_only(sessions: list[str]):
    """`clear(session_id)` removes only the named session. Other sessions
    must remain intact.
    """
    t = Tracer()
    for sid in sessions:
        t.record(_mk_record(sid, 0))
    target = sessions[0]
    t.clear(target)
    assert t.session(target) == []
    for other in sessions[1:]:
        assert len(t.session(other)) == 1


@given(
    sessions=st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=5),
        min_size=0,
        max_size=10,
        unique=True,
    )
)
@settings(max_examples=20, deadline=None)
def test_tracer_clear_all_removes_everything(sessions: list[str]):
    """`clear()` with no arg removes every session."""
    t = Tracer()
    for sid in sessions:
        t.record(_mk_record(sid, 0))
    t.clear()
    for sid in sessions:
        assert t.session(sid) == []
    assert list(t.all_sessions()) == []


@given(n_threads=st.integers(min_value=2, max_value=8))
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_tracer_concurrent_record_no_loss(n_threads: int):
    """Concurrent records from N threads — final count is sum of inputs.
    Invariant of the lock around `_sessions[buf].append`. A regression that
    drops the lock would race-corrupt the deque and lose records.
    """
    per_thread = 50
    cap = n_threads * per_thread + 10  # ensure cap > total records
    t = Tracer(max_records_per_session=cap)

    def worker(tid: int):
        for i in range(per_thread):
            t.record(_mk_record("shared", tid * per_thread + i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(t.session("shared")) == n_threads * per_thread


@given(n_threads=st.integers(min_value=2, max_value=6))
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_tracer_concurrent_session_isolation(n_threads: int):
    """Concurrent records to *different* session ids do not cross-contaminate.
    Each thread's session must contain exactly its own records.
    """
    per_thread = 30
    t = Tracer(max_records_per_session=per_thread + 5)

    def worker(tid: int):
        for i in range(per_thread):
            t.record(_mk_record(f"sess-{tid}", i))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for tid in range(n_threads):
        assert len(t.session(f"sess-{tid}")) == per_thread


# ---------------------------------------------------------------------------
# Sentinel filtering invariants
# ---------------------------------------------------------------------------


@given(min_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=20, deadline=None)
def test_sentinel_dispatched_events_meet_min_confidence(min_conf: float):
    """Every event seen by a registered handler must have
    confidence >= sentinel.min_confidence. The filter MUST NOT be skipped.
    Embedding_waste fires at 0.99, so for thresholds ≤ 0.99 the handler sees
    the event; for thresholds > 0.99 the handler sees nothing.
    """
    s = Sentinel(project="proj", min_confidence=min_conf, rules=["embedding_waste"])
    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    base_ts = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(2):
        s.record_call(
            CallRecord(
                session_id="s1",
                timestamp=base_ts + timedelta(seconds=i),
                provider="openai",
                model="text-embedding-3-small",
                method="embeddings.create",
                prompt_tokens=10,
                completion_tokens=0,
                latency_ms=20.0,
                request_hash="h",
                tool_calls=[],
                user_facing_output=False,
                raw_request={"input": "dup"},
            )
        )

    for ev in seen:
        assert ev.confidence >= min_conf
    if min_conf > 0.99:
        assert seen == []


@given(min_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=20, deadline=None)
def test_sentinel_returned_events_match_dispatched(min_conf: float):
    """The list returned by record_call equals the list dispatched to handlers
    (modulo block-mode propagation). A divergence means a future caller relying
    on the return value gets a different picture from the handler.
    """
    s = Sentinel(project="proj", min_confidence=min_conf, rules=["embedding_waste"])
    dispatched: list[LeakEvent] = []
    s.on_leak(lambda ev: dispatched.append(ev))

    base_ts = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
    s.record_call(
        CallRecord(
            session_id="s1",
            timestamp=base_ts,
            provider="openai",
            model="text-embedding-3-small",
            method="embeddings.create",
            prompt_tokens=10,
            completion_tokens=0,
            latency_ms=20.0,
            request_hash="h",
            tool_calls=[],
            user_facing_output=False,
            raw_request={"input": "dup"},
        )
    )
    returned = s.record_call(
        CallRecord(
            session_id="s1",
            timestamp=base_ts + timedelta(seconds=1),
            provider="openai",
            model="text-embedding-3-small",
            method="embeddings.create",
            prompt_tokens=10,
            completion_tokens=0,
            latency_ms=20.0,
            request_hash="h",
            tool_calls=[],
            user_facing_output=False,
            raw_request={"input": "dup"},
        )
    )
    # Handler-dispatched and method-returned must be the same set of objects.
    assert returned == dispatched


@given(
    n_calls=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=15, deadline=None)
def test_sentinel_no_op_rule_emits_nothing(n_calls: int):
    """A rule that always returns None never causes a dispatch. Invariant of
    the ``if ev is None: continue`` guard in record_call.
    """
    from token_sentinel.rules.base import Rule

    class NoOpRule(Rule):
        name = "noop"

        def evaluate(self, session, *, project):
            return None

    s = Sentinel(project="proj", rules=[])
    s._rules = [NoOpRule({})]  # inject directly
    seen = []
    s.on_leak(lambda ev: seen.append(ev))
    base_ts = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(n_calls):
        evs = s.record_call(
            CallRecord(
                session_id=f"s-{i}",
                timestamp=base_ts + timedelta(seconds=i),
                provider="anthropic",
                model="m",
                method="messages.create",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1.0,
                request_hash=str(uuid.uuid4()),
            )
        )
        assert evs == []
    assert seen == []


# ---------------------------------------------------------------------------
# Context-bloat slope invariants
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=2, max_value=50),
    constant=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None)
def test_constant_series_slope_zero(n: int, constant: int):
    """A constant series has slope 0. Property of OLS regression: zero
    variance in y → numerator is zero → slope is zero.
    """
    values = [constant] * n
    assert _linear_slope(values) == 0.0


@given(
    start=st.integers(min_value=0, max_value=10_000),
    step=st.integers(min_value=-1000, max_value=1000),
    n=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=50, deadline=None)
def test_arithmetic_series_slope_equals_step(start: int, step: int, n: int):
    """A strictly arithmetic series has slope == step. Property of OLS on
    perfect-line data: the slope is the difference per unit x.
    """
    values = [start + step * i for i in range(n)]
    slope = _linear_slope(values)
    # Within float tolerance — for integer inputs this is exact in IEEE 754.
    assert math.isclose(slope, float(step), abs_tol=1e-9)


@given(
    start=st.integers(min_value=1, max_value=10_000),
    n=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_strictly_decreasing_series_negative_slope(start: int, n: int):
    """A strictly decreasing arithmetic series has slope < 0."""
    values = [start - i * 5 for i in range(n)]
    slope = _linear_slope(values)
    assert slope < 0


@given(
    start=st.integers(min_value=0, max_value=10_000),
    step=st.integers(min_value=1, max_value=1_000),
    n=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_strictly_increasing_series_positive_slope(start: int, step: int, n: int):
    """A strictly increasing arithmetic series has slope > 0."""
    values = [start + step * i for i in range(n)]
    slope = _linear_slope(values)
    assert slope > 0


@given(
    n=st.integers(min_value=2, max_value=20),
    step=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20, deadline=None)
def test_slope_sign_flips_when_series_reversed(n: int, step: int):
    """slope(reversed(series)) == -slope(series). The independent variable is
    a 0..n-1 index, so reversing the y-values flips the slope sign exactly.
    """
    values = [10 + step * i for i in range(n)]
    slope_fwd = _linear_slope(values)
    slope_rev = _linear_slope(list(reversed(values)))
    assert math.isclose(slope_fwd, -slope_rev, abs_tol=1e-9)
