"""Tests for Tag-based Chargeback — ``Sentinel.session(tags=...)`` plus
the propagation contract from ``Session`` → :class:`CallRecord` →
:class:`LeakEvent` → wire payload.

The acceptance criteria mirror the v2.1 spec:

  - Tags propagate from ``session()`` to every CallRecord the session
    records, and from every CallRecord to the LeakEvent the rule emits.
  - Validation rejects (a) unknown tag keys, (b) non-URL-safe values,
    (c) values over the length cap, and (d) dicts over the entry cap.
  - The wire shape (``cloud_client._event_to_wire``) carries the tag
    dict on every outbound event so the cloud's by-tag aggregation
    has a value to bucket by.
  - Back-compat: a pre- call site that never touches ``session()``
    keeps working — CallRecord / LeakEvent both default ``tags`` to
    the empty dict and the wire payload echoes that.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from token_sentinel import LeakEvent, Sentinel
from token_sentinel.cloud_client import _event_to_wire
from token_sentinel.events import CallRecord

# ---------------------------------------------------------------------------
# session() — happy path: tag propagation to CallRecord
# ---------------------------------------------------------------------------


def test_session_with_tags_propagates_to_call_record(make_call, now):
    """The  headline: tags set on ``session()`` land on the CallRecord
    when the session's ``record_call`` is used."""
    sentinel = Sentinel(project="proj")
    sess = sentinel.session(tags={"team": "growth"})

    call = make_call(session_id=sess.session_id, timestamp=now)
    sess.record_call(call)

    # CallRecord.tags was populated from the session.
    assert call.tags == {"team": "growth"}


def test_session_supports_full_v21_tag_allowlist(make_call, now):
    """All five allowlist keys (team / feature / customer /
    environment / version) round-trip through ``session()`` and onto the
    CallRecord. Future tag-key additions land in the same dict shape."""
    sentinel = Sentinel(project="proj")
    tags = {
        "team": "growth",
        "feature": "summarizer",
        "customer": "acme_corp",
        "environment": "production",
        "version": "v1.2.3",
    }
    sess = sentinel.session(tags=tags)

    call = make_call(session_id=sess.session_id, timestamp=now)
    sess.record_call(call)
    assert call.tags == tags


def test_multiple_sessions_have_independent_tags(make_call, now):
    """Two sessions on the same Sentinel keep distinct tag dicts. The
    second session must NOT see the first session's tags — chargeback
    accuracy depends on this."""
    sentinel = Sentinel(project="proj")
    growth = sentinel.session(tags={"team": "growth"})
    payments = sentinel.session(tags={"team": "payments"})

    call_a = make_call(session_id=growth.session_id, timestamp=now)
    call_b = make_call(
        session_id=payments.session_id,
        timestamp=now + timedelta(seconds=1),
    )
    growth.record_call(call_a)
    payments.record_call(call_b)

    assert call_a.tags == {"team": "growth"}
    assert call_b.tags == {"team": "payments"}


def test_session_without_tags_defaults_to_empty_dict(make_call, now):
    """Back-compat: a session opened without ``tags=`` keeps the
    CallRecord.tags default empty dict."""
    sentinel = Sentinel(project="proj")
    sess = sentinel.session()

    call = make_call(session_id=sess.session_id, timestamp=now)
    sess.record_call(call)
    assert call.tags == {}


def test_call_record_default_tags_is_empty_dict():
    """The dataclass default — directly-built CallRecords (no session)
    keep ``tags`` as ``{}`` so the wire payload doesn't drift."""
    from datetime import datetime, timezone

    call = CallRecord(
        session_id="x",
        timestamp=datetime.now(timezone.utc),
        provider="anthropic",
        model="claude-sonnet-4-6",
        method="messages.create",
        prompt_tokens=10,
        completion_tokens=2,
        latency_ms=10.0,
        request_hash="abc",
    )
    assert call.tags == {}


# ---------------------------------------------------------------------------
# session() — validation: keys
# ---------------------------------------------------------------------------


def test_session_rejects_unknown_tag_key():
    """A tag key not in the allowlist raises ValueError. The error
    message names the rejected key and lists the allowed set."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match=r"not_in_allowlist"):
        sentinel.session(tags={"not_in_allowlist": "value"})


def test_session_allowlist_message_lists_valid_keys():
    """The ValueError message names every allowed key so the customer
    can fix the call site without a docs round-trip."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError) as exc:
        sentinel.session(tags={"bogus": "x"})
    msg = str(exc.value)
    # Every allowed key shows up in the message.
    for k in ("team", "feature", "customer", "environment", "version"):
        assert k in msg


def test_session_rejects_non_string_tag_key():
    """A non-string key (someone passing an int) is caught with a clear
    error rather than silently coerced."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match="must be a string"):
        sentinel.session(tags={42: "value"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# session() — validation: values
# ---------------------------------------------------------------------------


def test_session_rejects_value_with_spaces():
    """Values must be URL-safe — spaces are rejected so we can round-trip
    tag values through ``?tag=team&value=...`` query strings without
    escaping (and so dashboards have consistent rendering)."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match=r"URL-safe"):
        sentinel.session(tags={"team": "with space"})


def test_session_rejects_value_with_special_chars():
    """Values must match ``^[a-zA-Z0-9._-]+$`` — special chars rejected."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match=r"URL-safe"):
        sentinel.session(tags={"team": "a/b"})


def test_session_rejects_value_over_length_cap():
    """Values longer than 64 chars raise ValueError. The cap matches the
    cloud-side validator so the SDK can't accept a value the cloud
    rejects."""
    sentinel = Sentinel(project="proj")
    long_value = "a" * 65
    with pytest.raises(ValueError, match=r"exceeds 64"):
        sentinel.session(tags={"team": long_value})


def test_session_accepts_exact_length_cap():
    """The boundary case — exactly 64 chars must be accepted."""
    sentinel = Sentinel(project="proj")
    value = "a" * 64
    sess = sentinel.session(tags={"team": value})
    assert sess.tags == {"team": value}


def test_session_rejects_empty_string_value():
    """An empty string fails the regex (``+`` requires at least one
    char), so a customer passing ``tags={"team": ""}`` sees a clear
    error rather than landing an unbucketable event."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match=r"URL-safe"):
        sentinel.session(tags={"team": ""})


def test_session_rejects_non_string_value():
    """Non-string values are rejected explicitly."""
    sentinel = Sentinel(project="proj")
    with pytest.raises(ValueError, match="must be a string"):
        sentinel.session(tags={"team": 42})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# session() — validation: dict size
# ---------------------------------------------------------------------------


def test_session_rejects_too_many_tags():
    """The per-session cap of 8 entries protects the cloud's by-tag
    aggregation cardinality. Going over raises."""
    sentinel = Sentinel(project="proj")
    # 9 entries — all valid keys would max out at 5, so we need to
    # construct a synthetic "too many" case by going past the cap with
    # repeated valid keys (impossible in Python dict). Instead, the
    # validator checks length BEFORE per-entry validation, so we can
    # pass 9 entries that LOOK valid (we still error on length first).
    nine = {f"team{i}": "x" for i in range(9)}
    # The length check fires first regardless of key validity.
    with pytest.raises(ValueError, match=r"≤ 8 entries"):
        sentinel.session(tags=nine)


# ---------------------------------------------------------------------------
# Tag → CallRecord → LeakEvent propagation
# ---------------------------------------------------------------------------


def test_tags_appear_on_leak_event_after_rule_fires(now):
    """The  propagation contract: a CallRecord with tags drives a
    LeakEvent with the same tags so the customer's ``on_leak`` handler
    can route / attribute the event by team."""
    sentinel = Sentinel(project="proj", rules=["embedding_waste"])
    sess = sentinel.session(tags={"team": "growth", "feature": "rag"})

    captured: list[LeakEvent] = []

    @sentinel.on_leak
    def cap(ev: LeakEvent) -> None:
        captured.append(ev)

    # Fire embedding_waste: same embed call twice.
    for i in range(2):
        sess.record_call(
            CallRecord(
                session_id=sess.session_id,
                timestamp=now + timedelta(seconds=i),
                provider="openai",
                model="text-embedding-3-small",
                method="embeddings.create",
                prompt_tokens=10,
                completion_tokens=0,
                latency_ms=20.0,
                request_hash=hashlib.sha256(b"x").hexdigest(),
                raw_request={"input": "hello"},
            )
        )

    assert len(captured) == 1
    assert captured[0].tags == {"team": "growth", "feature": "rag"}


def test_leak_event_default_tags_is_empty_dict():
    """The dataclass default: a LeakEvent constructed without ``tags``
    has ``tags == {}`` so pre- rule code (which doesn't set tags)
    keeps producing the same shape."""
    ev = LeakEvent(
        type="x",
        confidence=0.9,
        project="proj",
        session_id="s",
        rule="r",
        evidence={},
        estimated_burn=0.0,
        suggested_action="halt",
    )
    assert ev.tags == {}


# ---------------------------------------------------------------------------
# Wire serialization
# ---------------------------------------------------------------------------


def test_event_to_wire_includes_tags():
    """The cloud sink's ``_event_to_wire`` helper round-trips ``tags``
    onto the JSON payload so the cloud's by-tag aggregation has a
    value to bucket by."""
    ev = LeakEvent(
        type="tool_loop",
        confidence=0.9,
        project="proj",
        session_id="s",
        rule="v0.tool_loop",
        evidence={"call_count": 4},
        estimated_burn=0.1,
        suggested_action="halt",
        tags={"team": "growth"},
    )
    wire = _event_to_wire(ev, sdk_version="0.18.0", mode="log")
    assert wire["tags"] == {"team": "growth"}


def test_event_to_wire_default_tags_empty_dict():
    """An event without tags serializes ``tags: {}`` so pre- cloud
    Pydantic models (which ignore unknown fields) accept the wire
    payload identically to previous versions."""
    ev = LeakEvent(
        type="tool_loop",
        confidence=0.9,
        project="proj",
        session_id="s",
        rule="v0.tool_loop",
        evidence={},
        estimated_burn=0.1,
        suggested_action="halt",
    )
    wire = _event_to_wire(ev, sdk_version="0.18.0", mode="log")
    assert wire["tags"] == {}


# ---------------------------------------------------------------------------
# Back-compat
# ---------------------------------------------------------------------------


def test_legacy_call_record_without_session_still_works(make_call, now):
    """A previous versions call site that constructs a CallRecord and passes it to
    ``sentinel.record_call`` (no Session involved) keeps working — the
    CallRecord's default empty ``tags`` flows through unchanged."""
    sentinel = Sentinel(project="proj")
    call = make_call(session_id="legacy", timestamp=now)
    # No exception, no surprising tag population.
    events = sentinel.record_call(call)
    assert events == []
    assert call.tags == {}


def test_session_id_auto_generated_when_omitted():
    """``session()`` mints a UUID when the customer doesn't pass one,
    matching the wrapper-level default. Two sessions get distinct ids."""
    sentinel = Sentinel(project="proj")
    a = sentinel.session()
    b = sentinel.session()
    assert a.session_id != b.session_id
    assert len(a.session_id) > 0


def test_explicit_session_id_preserved():
    """A customer-supplied ``session_id`` is preserved verbatim."""
    sentinel = Sentinel(project="proj")
    sess = sentinel.session(session_id="my-job-42")
    assert sess.session_id == "my-job-42"


def test_session_tags_are_defensive_copy():
    """Mutating the caller's tag dict after ``session()`` returns must
    NOT change the session's tags — chargeback accuracy depends on the
    tags being frozen at session-open time."""
    sentinel = Sentinel(project="proj")
    src = {"team": "growth"}
    sess = sentinel.session(tags=src)
    src["team"] = "mutated"
    src["feature"] = "rag"
    assert sess.tags == {"team": "growth"}


def test_session_record_call_rejects_mismatched_session_id(make_call, now):
    """A CallRecord whose ``session_id`` doesn't match the Session's
    ``session_id`` is rejected — silently overwriting would mask a
    real bug at the customer's call site."""
    sentinel = Sentinel(project="proj")
    sess = sentinel.session(session_id="A", tags={"team": "growth"})

    call = make_call(session_id="B", timestamp=now)
    with pytest.raises(ValueError, match="session_id"):
        sess.record_call(call)
