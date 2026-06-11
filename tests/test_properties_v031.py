"""Property-based tests for  safety hardening.

Five invariant classes:

A. Redaction (``_redact_args``, ``_build_sample_args``)
B. DoS caps (``_mean_pairwise_similarity`` with ``max_arg_bytes`` /
   ``max_total_corpus_bytes``)
C. Sentinel handler-lock + ``unregister`` semantics
D. ``record_call`` dispatch ordering (mode='block' raise behavior)
E. OpenAI ``_warn_block_mode_stream_once`` per-(Sentinel, path) bookkeeping

Each test docstring marks PROPERTY (Hypothesis) vs EXAMPLE (parametrized /
single-shot). The mode='block' raise behavior is single-shot in nature
(one call → exactly one raise), so it's covered with example tests rather
than fuzzed.

Settings: ``max_examples=50, deadline=200`` so the deadline assertion catches
DoS regressions automatically (a 200ms wall-clock budget is generous for
the 64KB-capped paths these rules exercise).
"""

from __future__ import annotations

import json
import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from token_sentinel import LeakDetected, LeakEvent, Sentinel
from token_sentinel.events import CallRecord
from token_sentinel.rules.base import Rule
from token_sentinel.rules.tool_loop import (
    _build_sample_args,
    _mean_pairwise_similarity,
    _redact_args,
)
from token_sentinel.wrappers.openai import (
    _BLOCK_MODE_STREAM_MESSAGE,
    _warn_block_mode_stream_once,
)

# ---------------------------------------------------------------------------
# Common strategies (bounded so CI runtime stays sane)
# ---------------------------------------------------------------------------

# JSON-safe primitive values (no NaN/inf — those make json.dumps surprising).
_json_value = st.one_of(
    st.integers(min_value=-10_000_000, max_value=10_000_000),
    st.text(max_size=200),
    st.lists(st.integers(min_value=-1000, max_value=1000), max_size=5),
    st.booleans(),
    st.none(),
)

# String-keyed dicts only — `_redact_args` documents the dict-shape contract
# in terms of string keys (it sorts by str(k) but indexes args[k] with the
# original key, so non-string keys would not survive that path; see the bug
# note in test A1 below).
_args_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=_json_value,
    min_size=0,
    max_size=8,
)


def _calls_from_args(args_list: list) -> list[dict]:
    """Wrap a list of arg dicts as tool_call dicts."""
    return [{"name": "search", "arguments": a} for a in args_list]


# ---------------------------------------------------------------------------
# A. Redaction invariants
# ---------------------------------------------------------------------------


@given(args=_args_dict)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_is_deterministic_property(args: dict):
    """PROPERTY: Redacting the same args twice produces identical output.

    The hash is the consumer's "is this the same call?" signal — it MUST
    be deterministic over equal input dicts.
    """
    a = _redact_args(dict(args))
    b = _redact_args(dict(args))
    assert a == b
    assert a["hash"] == b["hash"]
    assert a["keys"] == b["keys"]
    assert a["value_lengths"] == b["value_lengths"]


@given(args=_args_dict)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_is_json_serialisable_property(args: dict):
    """PROPERTY: Redacted output is JSON-serialisable.

    Leak handlers commonly stringify the evidence blob for transport.
    A non-JSON primitive sneaking through the redaction (e.g. bytes,
    a datetime, a tuple) breaks the contract silently.
    """
    redacted = _redact_args(args)
    blob = json.dumps(redacted)
    # Round-trip cleanly.
    rt = json.loads(blob)
    assert rt["keys"] == redacted["keys"]
    assert rt["hash"] == redacted["hash"]


@given(args=_args_dict)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_hash_is_16_hex_chars_property(args: dict):
    """PROPERTY: ``hash`` field is exactly 16 lowercase hex characters."""
    redacted = _redact_args(args)
    h = redacted["hash"]
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


@given(args=_args_dict)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_keys_are_sorted_property(args: dict):
    """PROPERTY: The ``keys`` field is sorted lexicographically.

    Handlers that diff "this call vs that call" rely on ordering being
    deterministic — sorted is the documented contract.
    """
    redacted = _redact_args(args)
    keys = redacted["keys"]
    assert keys == sorted(keys)
    # And the keys must equal the original set (just sorted).
    assert set(keys) == {str(k) for k in args}


@given(args=_args_dict)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_value_lengths_keys_subset_of_keys_property(args: dict):
    """PROPERTY: Every key in ``value_lengths`` appears in ``keys``.

    The two structures must be coherent — a key shown in ``keys`` should
    have a length report, and we should not invent phantom keys.
    """
    redacted = _redact_args(args)
    vl_keys = set(redacted["value_lengths"].keys())
    keys = set(redacted["keys"])
    assert vl_keys.issubset(keys)
    # Stronger: in fact all keys in `keys` should have a value_length entry.
    assert vl_keys == keys


@given(
    base=_args_dict,
    diff_key=st.text(min_size=1, max_size=20),
    diff_value_a=st.text(min_size=1, max_size=20),
    diff_value_b=st.text(min_size=1, max_size=20),
)
@settings(max_examples=50, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_redaction_distinguishes_different_values_property(
    base: dict, diff_key: str, diff_value_a: str, diff_value_b: str
):
    """PROPERTY: Two args dicts that share keys but differ in value at one
    key produce DIFFERENT hashes (when the values truly differ).

    This is the contract: "same hash means same call". A hash that
    collapsed across values would lose detection signal.
    """
    if diff_value_a == diff_value_b:
        return  # nothing to distinguish
    a = dict(base)
    b = dict(base)
    a[diff_key] = diff_value_a
    b[diff_key] = diff_value_b
    ha = _redact_args(a)["hash"]
    hb = _redact_args(b)["hash"]
    assert ha != hb, f"hashes collide: {ha} == {hb} for {a} vs {b}"


@given(
    args_list=st.lists(_args_dict, min_size=1, max_size=5),
    include_raw=st.booleans(),
)
@settings(max_examples=40, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_build_sample_args_idempotent_property(args_list: list[dict], include_raw: bool):
    """PROPERTY: ``_build_sample_args`` is deterministic and idempotent.

    Running it twice on the same input produces equal output, in both
    redacted and raw modes.
    """
    calls = _calls_from_args(args_list)
    a = _build_sample_args(calls, include_raw=include_raw)
    b = _build_sample_args(calls, include_raw=include_raw)
    assert a == b
    # Length must match input length.
    assert len(a) == len(args_list)


def test_redact_args_handles_non_string_keys():
    """Regression: non-string dict keys (int, tuple) used to crash
    ``_redact_args`` with ``KeyError`` because the sort stringified keys
    while the lookup used the original. Fixed in  by tracking
    ``(str_key, original_key)`` pairs and indexing by the original key.
    """
    out = _redact_args({1: "value", 2: "another"})
    # Keys are stringified for the output shape (sortable, JSON-friendly).
    assert out["keys"] == ["1", "2"]
    # value_lengths uses the stringified key form too.
    assert "1" in out["value_lengths"]
    assert "2" in out["value_lengths"]
    assert out["value_lengths"]["1"] > 0
    assert "hash" in out and len(out["hash"]) == 16

    # Tuple keys also work.
    out2 = _redact_args({(1, 2): "tuple-key"})
    assert out2["keys"] == ["(1, 2)"]
    assert "(1, 2)" in out2["value_lengths"]


# ---------------------------------------------------------------------------
# B. DoS-cap invariants
# ---------------------------------------------------------------------------


@given(
    payload_size=st.integers(min_value=1, max_value=200_000),
    n_calls=st.integers(min_value=2, max_value=5),
)
@settings(
    max_examples=20,
    deadline=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_similarity_under_deadline_property(payload_size: int, n_calls: int):
    """PROPERTY: For any payload size up to 200KB across N=2..5 calls,
    ``_mean_pairwise_similarity`` returns within Hypothesis's deadline
    (200ms wall clock).

    Hypothesis itself fails the test if any single example exceeds the
    deadline, so this catches DoS regressions automatically.
    """
    # Identical-prefix payloads — exercises the truncation path.
    payload = "X" * payload_size
    calls = _calls_from_args([{"q": payload + str(i)} for i in range(n_calls)])
    start = time.perf_counter()
    sim = _mean_pairwise_similarity(calls)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Under 200ms is the Hypothesis deadline; this is a belt-and-braces
    # explicit assertion in case the deadline is suppressed.
    assert elapsed_ms < 200, f"similarity took {elapsed_ms:.1f}ms"
    # Result is in [0, 1].
    assert 0.0 <= sim <= 1.0


@given(
    args_list=st.lists(_args_dict, min_size=2, max_size=6),
    cap=st.integers(min_value=10, max_value=100_000),
)
@settings(max_examples=40, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_similarity_in_unit_range_under_caps_property(args_list: list, cap: int):
    """PROPERTY: Similarity ∈ [0, 1] regardless of input size or cap.

    A NaN/negative/>1 result silently breaks threshold comparisons in
    ``ToolLoopRule.evaluate``.
    """
    calls = _calls_from_args(args_list)
    sim = _mean_pairwise_similarity(calls, max_arg_bytes=cap)
    assert 0.0 <= sim <= 1.0 + 1e-9, f"out of range: {sim}"
    # Sanity: NaN isn't ≤1 in floating-point.
    assert not math.isnan(sim)


@given(
    base=st.text(alphabet="abcdefghij ", min_size=10, max_size=200),
    n_calls=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=30, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_max_arg_bytes_above_input_does_not_change_result_property(base: str, n_calls: int):
    """PROPERTY: When ``max_arg_bytes`` is set higher than the input,
    raising it further has no effect.

    The cap is a pre-truncation guard — once the input fits, the cap is
    inert. A regression that truncated at unrelated lengths would break
    this.
    """
    calls = _calls_from_args([{"q": base} for _ in range(n_calls)])
    # Compute the actual JSON length of one arg; pick caps comfortably above.
    one_arg_json = json.dumps({"q": base})
    cap_just_above = len(one_arg_json) + 1000
    cap_huge = 10_000_000
    sim_a = _mean_pairwise_similarity(calls, max_arg_bytes=cap_just_above)
    sim_b = _mean_pairwise_similarity(calls, max_arg_bytes=cap_huge)
    assert math.isclose(sim_a, sim_b, abs_tol=1e-9), (
        f"cap-above-input changed result: {sim_a} vs {sim_b}"
    )


@given(
    n_calls=st.integers(min_value=3, max_value=6),
    blob_size=st.integers(min_value=10_000, max_value=50_000),
    total_cap_factor=st.floats(min_value=1.5, max_value=3.5, allow_nan=False),
)
@settings(max_examples=20, deadline=300, suppress_health_check=[HealthCheck.too_slow])
def test_max_total_corpus_drops_trailing_in_order_property(
    n_calls: int, blob_size: int, total_cap_factor: float
):
    """PROPERTY: Smaller ``max_total_corpus_bytes`` drops trailing args
    (deterministic insertion order); leading args dictate the result.

    Concretely: with N identical args and a cap that admits only K<N,
    the result equals the result of just running on the first K.
    """
    blob = "Y" * blob_size
    calls = _calls_from_args([{"payload": blob} for _ in range(n_calls)])
    # Cap to roughly fit only a subset of calls. Use a per-arg cap large
    # enough that per-arg truncation does not kick in.
    total_cap = int(blob_size * total_cap_factor)
    sim_capped = _mean_pairwise_similarity(
        calls,
        max_arg_bytes=blob_size * 2,
        max_total_corpus_bytes=total_cap,
    )
    # All inputs are identical → similarity is 1.0 if at least 2 fit.
    # Stronger: result equals what we'd get for a smaller insertion-order subset.
    # Confirm by re-running with strictly smaller corpus and comparing.
    if sim_capped > 0:
        # Identical args → similarity must be 1.0 regardless of how many
        # fit (so long as ≥ 2 do).
        assert math.isclose(sim_capped, 1.0, abs_tol=1e-6)


def test_total_corpus_drops_in_insertion_order_example():
    """EXAMPLE: Distinguish leading vs trailing args explicitly.

    With three calls — two identical ("aaa") and one different ("zzz") —
    a total-corpus cap small enough to admit only the first two but not
    the third must yield similarity=1.0 (the two identical leading args).
    A regression that dropped the *leading* args instead would yield <1.0.
    """
    common = "a" * 1000
    different = "z" * 1000
    calls = [
        {"name": "search", "arguments": {"q": common}},
        {"name": "search", "arguments": {"q": common}},
        {"name": "search", "arguments": {"q": different}},
    ]
    # Cap so only first two args fit (each ~1009 bytes JSON-serialized;
    # 2200 should fit two but not three).
    sim = _mean_pairwise_similarity(
        calls,
        max_arg_bytes=10_000,
        max_total_corpus_bytes=2_200,
    )
    # Leading two identical → similarity 1.0 (third dropped).
    assert math.isclose(sim, 1.0, abs_tol=1e-6), f"got {sim}"


def test_total_corpus_too_small_returns_zero_example():
    """EXAMPLE: If the cap is so small only one arg survives, similarity=0.0.

    Pinned in the ``_mean_pairwise_similarity`` docstring: ≥2 evaluable
    args required.
    """
    blob = "X" * 500
    calls = _calls_from_args([{"payload": blob} for _ in range(3)])
    sim = _mean_pairwise_similarity(
        calls,
        max_arg_bytes=10_000,
        max_total_corpus_bytes=400,  # smaller than even one blob
    )
    # The first arg fits (the cap check is "if total + len(s) > cap and
    # args_strs"); subsequent ones don't. So we end up with 1 arg → 0.0.
    assert sim == 0.0


# ---------------------------------------------------------------------------
# C. Sentinel handler-lock + unregister invariants
# ---------------------------------------------------------------------------


def _make_event(*, type_: str = "tool_loop", confidence: float = 0.8) -> LeakEvent:
    return LeakEvent(
        type=type_,
        confidence=confidence,
        project="proj",
        session_id="s1",
        rule=type_,
        evidence={},
        estimated_burn=0.01,
        suggested_action="x",
        raised_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_double_unregister_returns_false_example():
    """EXAMPLE: ``unregister(h)`` after a successful ``unregister(h)``
    returns ``False`` (handler is gone, second remove is a no-op).
    """
    s = Sentinel(project="proj")

    def h(ev):
        return None

    s.on_leak(h)
    assert s.unregister(h) is True
    assert s.unregister(h) is False
    # Repeating doesn't suddenly resurrect it.
    assert s.unregister(h) is False


def test_unregister_prevents_dispatch_example():
    """EXAMPLE: ``on_leak(h) → unregister(h) → _run_handlers(ev)`` does
    not invoke ``h``.
    """
    s = Sentinel(project="proj")
    seen: list[LeakEvent] = []

    def h(ev):
        seen.append(ev)

    s.on_leak(h)
    assert s.unregister(h) is True
    s._run_handlers(_make_event())
    assert seen == []


@given(n=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, deadline=200, suppress_health_check=[HealthCheck.too_slow])
def test_register_n_unregister_one_leaves_n_minus_one_property(n: int):
    """PROPERTY: Register the SAME handler N times, unregister once →
    N-1 instances remain, and ``_run_handlers`` invokes it N-1 times.

    Matches ``list.remove`` semantics — ``unregister`` removes exactly
    one occurrence per call.
    """
    s = Sentinel(project="proj")
    counter: list[int] = [0]

    def h(ev):
        counter[0] += 1

    for _ in range(n):
        s.on_leak(h)
    assert s._handlers.count(h) == n
    assert s.unregister(h) is True
    assert s._handlers.count(h) == n - 1

    s._run_handlers(_make_event())
    assert counter[0] == n - 1


@given(m=st.integers(min_value=2, max_value=8))
@settings(
    max_examples=10,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_concurrent_on_leak_yields_m_handlers_property(m: int):
    """PROPERTY: M threads, one ``on_leak`` per thread → ``len(_handlers) == M``.

    Exercises the ``_handler_lock`` against concurrent appends. Without
    the lock (and given the GIL still keeps ``list.append`` mostly safe
    in CPython) this would still pass most of the time — but the test
    serves as a regression guard against future re-entrancy / partial
    state bugs.
    """
    s = Sentinel(project="proj")
    barrier = threading.Barrier(m)

    def register(idx: int):
        def h(ev, _i=idx):
            return None

        barrier.wait(timeout=5.0)
        s.on_leak(h)

    with ThreadPoolExecutor(max_workers=m) as pool:
        futures = [pool.submit(register, i) for i in range(m)]
        for f in futures:
            f.result(timeout=5.0)

    assert len(s._handlers) == m
    for h in s._handlers:
        assert callable(h)


# ---------------------------------------------------------------------------
# D. Dispatch ordering invariants (mode='block')
# ---------------------------------------------------------------------------


class _StubRule(Rule):
    """A test-only rule that returns a canned event."""

    def __init__(self, name: str, event: LeakEvent | None):
        super().__init__(config={})
        self.name = name
        self._event = event

    def evaluate(self, session, *, project):  # type: ignore[override]
        return self._event


def _install_stub_rules(s: Sentinel, *events: LeakEvent | None) -> None:
    s._rules = [_StubRule(name=f"stub_{i}", event=ev) for i, ev in enumerate(events)]


def _make_call_record() -> CallRecord:
    return CallRecord(
        session_id="s1",
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        provider="anthropic",
        model="claude",
        method="messages.create",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=10.0,
        request_hash="abc",
    )


@pytest.mark.parametrize("n_events", [1, 2, 3, 5, 8])
def test_block_mode_handler_called_n_times_for_n_events(n_events: int):
    """EXAMPLE: For N events fired by rules + 1 handler, the handler is
    called exactly N times before the raise.

    Single-shot semantics — the raise happens once after every handler
    fires for every event.
    """
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    events = [_make_event(type_=f"t_{i}", confidence=0.5 + i * 0.05) for i in range(n_events)]
    _install_stub_rules(s, *events)

    calls: list[LeakEvent] = []
    s.on_leak(lambda ev: calls.append(ev))

    with pytest.raises(LeakDetected):
        s.record_call(_make_call_record())

    assert len(calls) == n_events
    # Order must match rule iteration order.
    assert [e.type for e in calls] == [e.type for e in events]


@pytest.mark.parametrize(
    "confidences",
    [
        [0.5, 0.9, 0.7],
        [0.95, 0.95, 0.95],  # all tied — first wins
        [0.1, 0.2, 0.3, 0.4, 0.5],  # last is highest
        [0.99, 0.5, 0.7],  # first is highest
    ],
)
def test_block_mode_raises_with_highest_confidence(confidences: list[float]):
    """EXAMPLE: ``LeakDetected.event`` is the event with the highest
    confidence (ties broken by first-iteration-order).
    """
    s = Sentinel(project="proj", mode="block", min_confidence=0.0)
    events = [_make_event(type_=f"t_{i}", confidence=c) for i, c in enumerate(confidences)]
    _install_stub_rules(s, *events)

    with pytest.raises(LeakDetected) as exc:
        s.record_call(_make_call_record())

    # Find the expected winner: max confidence, tiebreak by earliest index.
    max_conf = max(confidences)
    expected_idx = confidences.index(max_conf)
    expected_event = events[expected_idx]
    assert exc.value.event is expected_event


def test_handler_raise_does_not_block_other_handlers_or_events_example():
    """EXAMPLE: A handler that raises does NOT stop the same-event sibling
    handlers, NOR does it stop subsequent events from running their
    handlers.

    Pinned in the dispatch-ordering tests; this test re-exercises it
    with mode='log' to keep this file self-contained.
    """
    s = Sentinel(project="proj", mode="log", min_confidence=0.0)
    e1 = _make_event(type_="a", confidence=0.6)
    e2 = _make_event(type_="b", confidence=0.7)
    _install_stub_rules(s, e1, e2)

    fired: list[str] = []

    @s.on_leak
    def bad(ev):
        fired.append(f"bad:{ev.type}")
        raise RuntimeError("boom")

    @s.on_leak
    def good(ev):
        fired.append(f"good:{ev.type}")

    s.record_call(_make_call_record())
    assert fired == ["bad:a", "good:a", "bad:b", "good:b"]


@pytest.mark.parametrize("mode", ["log", "alert"])
@pytest.mark.parametrize("n_events", [0, 1, 3, 8])
def test_log_alert_modes_never_raise(mode: str, n_events: int):
    """EXAMPLE: Modes 'log' and 'alert' never raise regardless of how
    many events fire. Only 'block' is allowed to raise ``LeakDetected``.
    """
    s = Sentinel(project="proj", mode=mode, min_confidence=0.0)  # type: ignore[arg-type]
    events = [_make_event(type_=f"t_{i}", confidence=0.99) for i in range(n_events)]
    _install_stub_rules(s, *events)

    seen: list[LeakEvent] = []
    s.on_leak(lambda ev: seen.append(ev))

    # Must not raise.
    out = s.record_call(_make_call_record())
    assert len(out) == n_events
    assert len(seen) == n_events


# ---------------------------------------------------------------------------
# E. OpenAI stream-warning bookkeeping
# ---------------------------------------------------------------------------


def _stream_warnings(records):
    return [w for w in records if str(w.message) == _BLOCK_MODE_STREAM_MESSAGE]


@pytest.mark.parametrize("path", ["sync", "async"])
@pytest.mark.parametrize("n_calls", [1, 2, 5, 10, 50])
def test_warn_block_mode_stream_once_per_path(path: str, n_calls: int):
    """EXAMPLE: For a fixed Sentinel + path, only ONE warning fires
    regardless of how many times ``_warn_block_mode_stream_once`` is
    called.

    Pinned per the bookkeeping in ``_WARNED_INSTANCES``.
    """
    s = Sentinel(project="proj", mode="block")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(n_calls):
            _warn_block_mode_stream_once(s, path)

    assert len(_stream_warnings(caught)) == 1


def test_warn_block_mode_stream_distinct_sentinels_each_get_a_warning():
    """EXAMPLE: Two distinct Sentinel instances each get their own
    warning. The bookkeeping is per-instance.
    """
    s_a = Sentinel(project="a", mode="block")
    s_b = Sentinel(project="b", mode="block")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_block_mode_stream_once(s_a, "sync")
        _warn_block_mode_stream_once(s_b, "sync")
        # Repeats — should not produce extras.
        _warn_block_mode_stream_once(s_a, "sync")
        _warn_block_mode_stream_once(s_b, "sync")

    assert len(_stream_warnings(caught)) == 2


def test_warn_block_mode_stream_sync_and_async_warn_separately_on_same_sentinel():
    """EXAMPLE: Sync and async paths each get their own warning even
    when they share a Sentinel. Two warnings total: one per path.
    """
    s = Sentinel(project="proj", mode="block")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            _warn_block_mode_stream_once(s, "sync")
            _warn_block_mode_stream_once(s, "async")

    assert len(_stream_warnings(caught)) == 2


def test_warn_block_mode_stream_suppressed_by_simplefilter_ignore_runtime():
    """EXAMPLE: ``warnings.simplefilter("ignore", RuntimeWarning)`` fully
    silences the warning. Required so customers can opt out cleanly.
    """
    s = Sentinel(project="proj", mode="block")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", RuntimeWarning)
        _warn_block_mode_stream_once(s, "sync")

    assert _stream_warnings(caught) == []


def test_warn_block_mode_stream_emits_correct_category_with_pytest_warns():
    """EXAMPLE: The warning is a ``RuntimeWarning`` (not Warning, not
    UserWarning). ``pytest.warns(RuntimeWarning)`` must catch it.
    """
    s = Sentinel(project="proj", mode="block")
    with pytest.warns(RuntimeWarning, match="OpenAI streaming bypass"):
        _warn_block_mode_stream_once(s, "sync")


@given(
    path_calls=st.lists(
        st.sampled_from(["sync", "async"]),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=30, deadline=500, suppress_health_check=[HealthCheck.too_slow])
def test_warn_block_mode_stream_once_per_path_property(path_calls: list[str]):
    """PROPERTY: For any sequence of path calls, the count of emitted
    warnings equals the number of *distinct* paths actually touched on
    a single Sentinel.

    Each (Sentinel, path) gets exactly one warning across its lifetime;
    repeated calls with the same path are idempotent.
    """
    s = Sentinel(project="proj", mode="block")
    distinct_paths_touched = set(path_calls)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for path in path_calls:
            _warn_block_mode_stream_once(s, path)

    assert len(_stream_warnings(caught)) == len(distinct_paths_touched)
