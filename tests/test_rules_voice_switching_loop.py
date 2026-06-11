"""Tests for ``VoiceSwitchingLoopRule``.

 ElevenLabs voice-experimentation rule. Fires when the same input
``text_hash`` is synthesized against ``>= N`` distinct ``voice_id``
values within ``window_seconds`` (defaults: 3 voices / 10s). The
ElevenLabs wrapper populates ``usage_extra.model_specific_meta`` with
``voice_id`` and ``text_hash``; tests construct CallRecord directly
(``make_call`` does not pass through ``usage_extra``).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from token_sentinel.events import CallRecord
from token_sentinel.rules.voice_switching_loop import VoiceSwitchingLoopRule

_BASE_TS = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    """Match the ElevenLabs wrapper's truncation: 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _elevenlabs_record(
    *,
    voice_id: str = "voice-rachel",
    text: str = "hello world",
    method: str = "text_to_speech.convert",
    provider: str = "elevenlabs",
    timestamp_offset_s: float = 0.0,
    char_count: int | None = None,
    include_usage_extra: bool = True,
    include_model_specific_meta: bool = True,
    text_hash_override: str | None = None,
    voice_id_override: Any = ...,  # sentinel: "not provided"
) -> CallRecord:
    """Build a CallRecord shaped like the ElevenLabs wrapper emits.

    ``include_usage_extra=False`` / ``include_model_specific_meta=False``
    let tests exercise the defensive paths.
    """
    if char_count is None:
        char_count = len(text)
    ts = _BASE_TS + timedelta(seconds=timestamp_offset_s)
    resolved_text_hash = text_hash_override if text_hash_override is not None else _text_hash(text)

    if include_usage_extra:
        msm: dict[str, Any] = {}
        if include_model_specific_meta:
            msm["text_hash"] = resolved_text_hash
            # Use the sentinel so a test can explicitly pass None / "" /
            # missing voice_id to exercise defensive paths.
            if voice_id_override is ...:
                msm["voice_id"] = voice_id
            elif voice_id_override is not ...:
                msm["voice_id"] = voice_id_override
            msm["model_id"] = "eleven_turbo_v2"
            msm["output_format"] = "mp3_44100_128"
        usage_extra: dict[str, Any] = {
            "dimension_kind": "per_character",
            "dimension_value": char_count,
        }
        if include_model_specific_meta:
            usage_extra["model_specific_meta"] = msm
    else:
        usage_extra = {}

    return CallRecord(
        session_id="voice-session",
        timestamp=ts,
        provider=provider,
        model=voice_id,
        method=method,
        prompt_tokens=char_count,
        completion_tokens=0,
        latency_ms=350.0,
        request_hash="elevenlabs-hash",
        user_facing_output=True,
        usage_extra=usage_extra,
    )


# ---------------------------------------------------------------------------
# 1. Fires on 3 voices × same text within window
# ---------------------------------------------------------------------------


def test_fires_on_three_voices_same_text_in_window() -> None:
    """Baseline positive: same text against 3 distinct voices in 10s."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="welcome", timestamp_offset_s=2),
        _elevenlabs_record(voice_id="bella", text="welcome", timestamp_offset_s=5),
    ]
    event = rule.evaluate(session, project="p")
    assert event is not None
    assert event.type == "voice_switching_loop"
    assert event.rule == "v0.voice_switching_loop"
    assert event.suggested_action == "lock_voice_id_or_cache_synthesis"
    assert event.evidence["call_count"] == 3
    # Voice IDs returned sorted for deterministic evidence.
    assert event.evidence["voice_ids_observed"] == ["adam", "bella", "rachel"]
    assert event.evidence["text_hash"] == _text_hash("welcome")


# ---------------------------------------------------------------------------
# 2. Doesn't fire on 3 voices × different texts
# ---------------------------------------------------------------------------


def test_does_not_fire_when_each_call_has_different_text() -> None:
    """Three distinct voices but EACH against different text — no loop."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(voice_id="rachel", text="greeting one", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="greeting two", timestamp_offset_s=2),
        _elevenlabs_record(voice_id="bella", text="greeting three", timestamp_offset_s=5),
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 3. Doesn't fire on 1 voice × same text (no voice variation)
# ---------------------------------------------------------------------------


def test_does_not_fire_when_only_one_voice_used() -> None:
    """Three calls, same text, but ALL same voice — no voice-switching."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=2),
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=5),
    ]
    # That's retry-storm shaped, NOT voice-switching. The retry_storm
    # rule handles it; this rule must stay quiet.
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 4. Doesn't fire outside the window
# ---------------------------------------------------------------------------


def test_does_not_fire_outside_window() -> None:
    """Three voices × same text BUT spread across 20s — outside default 10s."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="welcome", timestamp_offset_s=8),
        # Anchor is the latest call's timestamp (20s). The first two
        # calls are 20s and 12s in the past — outside the 10s window.
        _elevenlabs_record(voice_id="bella", text="welcome", timestamp_offset_s=20),
    ]
    # Only one voice (bella) is in-window; the other two voices are
    # outside it. Threshold not met.
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 5. Doesn't fire on non-elevenlabs CallRecords
# ---------------------------------------------------------------------------


def test_does_not_fire_on_non_elevenlabs_provider() -> None:
    """Provider gate: identically-shaped non-elevenlabs records must not fire.

    Even if a future provider populated identically-named meta fields,
    the rule's remediation (``lock_voice_id_or_cache_synthesis``) is
    ElevenLabs-specific; firing elsewhere would be wrong advice.
    """
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(
            voice_id="rachel", text="welcome", provider="openai", timestamp_offset_s=0
        ),
        _elevenlabs_record(
            voice_id="adam", text="welcome", provider="openai", timestamp_offset_s=2
        ),
        _elevenlabs_record(
            voice_id="bella", text="welcome", provider="openai", timestamp_offset_s=5
        ),
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 6. Doesn't fire on non-text_to_speech methods
# ---------------------------------------------------------------------------


def test_does_not_fire_on_non_text_to_speech_methods() -> None:
    """Method gate: a hypothetical future ElevenLabs ``voice.clone`` etc.
    must not trigger this rule even if it shipped the same meta shape."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(
            voice_id="rachel", text="welcome", method="voice.clone", timestamp_offset_s=0
        ),
        _elevenlabs_record(
            voice_id="adam", text="welcome", method="voice.clone", timestamp_offset_s=2
        ),
        _elevenlabs_record(
            voice_id="bella", text="welcome", method="voice.clone", timestamp_offset_s=5
        ),
    ]
    assert rule.evaluate(session, project="p") is None


# ---------------------------------------------------------------------------
# 7. Confidence scaling — +0.05 per extra voice beyond threshold, capped 0.9
# ---------------------------------------------------------------------------


def test_confidence_scales_with_extra_voices_and_caps_at_0_9() -> None:
    """Confidence: 0.7 baseline at N=3, +0.05 per extra voice, cap 0.9."""
    rule = VoiceSwitchingLoopRule({})

    def session_with_n_voices(n: int) -> list[CallRecord]:
        return [
            _elevenlabs_record(voice_id=f"voice-{i}", text="welcome", timestamp_offset_s=float(i))
            for i in range(n)
        ]

    # N=3 -> base 0.7
    ev_3 = rule.evaluate(session_with_n_voices(3), project="p")
    assert ev_3 is not None
    assert abs(ev_3.confidence - 0.7) < 1e-9

    # N=4 -> 0.7 + 0.05 = 0.75
    ev_4 = rule.evaluate(session_with_n_voices(4), project="p")
    assert ev_4 is not None
    assert abs(ev_4.confidence - 0.75) < 1e-9

    # N=5 -> 0.7 + 0.10 = 0.80
    ev_5 = rule.evaluate(session_with_n_voices(5), project="p")
    assert ev_5 is not None
    assert abs(ev_5.confidence - 0.8) < 1e-9

    # N=7 -> would be 0.7 + 0.20 = 0.90 — at cap. Float-arithmetic
    # quirk in the accumulator means we compare with tolerance.
    ev_7 = rule.evaluate(session_with_n_voices(7), project="p")
    assert ev_7 is not None
    assert abs(ev_7.confidence - 0.9) < 1e-9

    # N=10 -> would be 1.05, capped to 0.9 (exact value because the
    # min() call returns the literal 0.9, not the accumulated sum).
    ev_10 = rule.evaluate(session_with_n_voices(10), project="p")
    assert ev_10 is not None
    assert ev_10.confidence == 0.9


# ---------------------------------------------------------------------------
# 8. Evidence shape — voice_ids, text_hash, call_count, time_span_seconds
# ---------------------------------------------------------------------------


def test_evidence_contains_all_required_fields() -> None:
    """Evidence dict must carry voice_ids_observed, text_hash, call_count,
    time_span_seconds — dashboard renders them verbatim."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(voice_id="rachel", text="hi", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="hi", timestamp_offset_s=3),
        _elevenlabs_record(voice_id="bella", text="hi", timestamp_offset_s=7),
    ]
    event = rule.evaluate(session, project="p")
    assert event is not None
    ev = event.evidence
    assert isinstance(ev["voice_ids_observed"], list)
    assert ev["voice_ids_observed"] == ["adam", "bella", "rachel"]
    assert ev["text_hash"] == _text_hash("hi")
    assert ev["call_count"] == 3
    # 0 -> 7 seconds = 7.0
    assert ev["time_span_seconds"] == 7.0


# ---------------------------------------------------------------------------
# 9. Custom window_seconds — narrower window lets us cluster aggressively
# ---------------------------------------------------------------------------


def test_custom_window_seconds_config() -> None:
    """``voice_switching_loop.window_seconds`` config override is honored.

    With the default 10s window the second of three calls falls within
    the window only if it's within 10s of the latest. We narrow to 3s
    here and verify that calls beyond the narrow window are excluded.
    """
    rule_narrow = VoiceSwitchingLoopRule({"voice_switching_loop.window_seconds": 3})

    session = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="welcome", timestamp_offset_s=1),
        # The latest call is at 6s; window=3 means anything before 3s
        # is excluded. Only adam (1s) and bella (6s) are inside.
        _elevenlabs_record(voice_id="bella", text="welcome", timestamp_offset_s=6),
    ]
    # Only 2 voices in the narrow window — below min_voices=3.
    assert rule_narrow.evaluate(session, project="p") is None

    # Same session under the wide default — fires.
    rule_default = VoiceSwitchingLoopRule({})
    assert rule_default.evaluate(session, project="p") is not None


# ---------------------------------------------------------------------------
# 10. Custom min_voices config — N=4 means 3-voice session no longer fires
# ---------------------------------------------------------------------------


def test_custom_min_voices_config() -> None:
    """``voice_switching_loop.min_voices`` config override changes threshold."""
    rule_strict = VoiceSwitchingLoopRule({"voice_switching_loop.min_voices": 4})
    session = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="welcome", timestamp_offset_s=2),
        _elevenlabs_record(voice_id="bella", text="welcome", timestamp_offset_s=5),
    ]
    # 3 voices, but threshold is 4 — must not fire.
    assert rule_strict.evaluate(session, project="p") is None

    # Add a 4th voice within the window and confirm it fires.
    session.append(_elevenlabs_record(voice_id="antoni", text="welcome", timestamp_offset_s=8))
    event = rule_strict.evaluate(session, project="p")
    assert event is not None
    assert event.evidence["call_count"] == 4


# ---------------------------------------------------------------------------
# 11. All three text_to_speech methods fire (defensive matrix coverage)
# ---------------------------------------------------------------------------


def test_fires_on_all_text_to_speech_methods() -> None:
    """``convert`` / ``convert_as_stream`` / ``stream`` are all matched by
    the ``text_to_speech.*`` prefix gate."""
    rule = VoiceSwitchingLoopRule({})
    session = [
        _elevenlabs_record(
            voice_id="rachel",
            text="welcome",
            method="text_to_speech.convert",
            timestamp_offset_s=0,
        ),
        _elevenlabs_record(
            voice_id="adam",
            text="welcome",
            method="text_to_speech.convert_as_stream",
            timestamp_offset_s=2,
        ),
        _elevenlabs_record(
            voice_id="bella",
            text="welcome",
            method="text_to_speech.stream",
            timestamp_offset_s=5,
        ),
    ]
    event = rule.evaluate(session, project="p")
    assert event is not None
    assert event.evidence["call_count"] == 3


# ---------------------------------------------------------------------------
# 12. Defensive — missing usage_extra / model_specific_meta short-circuits
# ---------------------------------------------------------------------------


def test_handles_missing_usage_extra_gracefully() -> None:
    """Records without ``usage_extra`` or ``model_specific_meta`` are ignored,
    NOT crashes. Mirrors the audio_multichannel_doubling rule's defensive
    posture."""
    rule = VoiceSwitchingLoopRule({})

    # Empty session: returns None.
    assert rule.evaluate([], project="p") is None

    # All three records have NO usage_extra. The rule must short-circuit.
    session = [
        _elevenlabs_record(
            voice_id="rachel",
            text="welcome",
            timestamp_offset_s=0,
            include_usage_extra=False,
        ),
        _elevenlabs_record(
            voice_id="adam",
            text="welcome",
            timestamp_offset_s=2,
            include_usage_extra=False,
        ),
        _elevenlabs_record(
            voice_id="bella",
            text="welcome",
            timestamp_offset_s=5,
            include_usage_extra=False,
        ),
    ]
    assert rule.evaluate(session, project="p") is None

    # usage_extra present but model_specific_meta missing — same path.
    session_no_meta = [
        _elevenlabs_record(
            voice_id="rachel",
            text="welcome",
            timestamp_offset_s=0,
            include_model_specific_meta=False,
        ),
        _elevenlabs_record(
            voice_id="adam",
            text="welcome",
            timestamp_offset_s=2,
            include_model_specific_meta=False,
        ),
        _elevenlabs_record(
            voice_id="bella",
            text="welcome",
            timestamp_offset_s=5,
            include_model_specific_meta=False,
        ),
    ]
    assert rule.evaluate(session_no_meta, project="p") is None

    # Records with missing voice_id (empty string) are ignored from the
    # cluster. Two valid voices + one missing => 2 voices < 3, no fire.
    session_one_missing = [
        _elevenlabs_record(voice_id="rachel", text="welcome", timestamp_offset_s=0),
        _elevenlabs_record(voice_id="adam", text="welcome", timestamp_offset_s=2),
        _elevenlabs_record(
            voice_id="bella",
            text="welcome",
            timestamp_offset_s=5,
            voice_id_override="",
        ),
    ]
    assert rule.evaluate(session_one_missing, project="p") is None
