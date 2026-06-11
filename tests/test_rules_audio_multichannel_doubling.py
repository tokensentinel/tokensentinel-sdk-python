"""Tests for ``AudioMultichannelDoublingRule``.

audio leak rule. Fires only on Deepgram transcribe calls where
``usage_extra.model_specific_meta`` reports ``channels >= 2`` AND
``multichannel is True``. The Deepgram wrapper populates that meta
shape; tests construct ``CallRecord`` directly (bypassing the
``make_call`` fixture, which does not pass through ``usage_extra``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from token_sentinel.events import CallRecord
from token_sentinel.rules.audio_multichannel_doubling import (
    AudioMultichannelDoublingRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deepgram_record(
    *,
    provider: str = "deepgram",
    method: str = "transcribe_file",
    channels: int | None = 2,
    multichannel: bool | None = True,
    duration_seconds: float = 60.0,
    diarize: bool = False,
    include_usage_extra: bool = True,
    include_model_specific_meta: bool = True,
    model: str = "nova-2",
) -> CallRecord:
    """Build a CallRecord shaped like what the Deepgram wrapper emits.

    ``include_usage_extra=False`` / ``include_model_specific_meta=False``
    let tests exercise the defensive paths where the wrapper (or a custom
    subclass) doesn't populate the expected fields.
    """
    if include_usage_extra:
        model_specific_meta: dict[str, Any] = {}
        if include_model_specific_meta:
            if channels is not None:
                model_specific_meta["channels"] = channels
            if multichannel is not None:
                model_specific_meta["multichannel"] = multichannel
            model_specific_meta["diarize"] = diarize
        usage_extra: dict[str, Any] = {
            "dimension_kind": "per_second",
            "dimension_value": duration_seconds,
        }
        if include_model_specific_meta:
            usage_extra["model_specific_meta"] = model_specific_meta
    else:
        usage_extra = {}

    return CallRecord(
        session_id="audio-session",
        timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
        provider=provider,
        model=model,
        method=method,
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=850.0,
        request_hash="deepgram-hash",
        usage_extra=usage_extra,
    )


# ---------------------------------------------------------------------------
# 1. test_fires_on_deepgram_multichannel_stereo
# ---------------------------------------------------------------------------


def test_fires_on_deepgram_multichannel_stereo() -> None:
    """Baseline positive: Deepgram + transcribe_file + stereo + multichannel=True."""
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=2, multichannel=True)
    event = rule.evaluate([record], project="p")
    assert event is not None
    assert event.type == "audio_multichannel_doubling"
    assert event.rule == "v0.audio_multichannel_doubling"
    assert event.evidence["channels"] == 2
    assert event.evidence["multichannel"] is True


# ---------------------------------------------------------------------------
# 2. test_fires_with_higher_confidence_on_4plus_channels
# ---------------------------------------------------------------------------


def test_fires_with_higher_confidence_on_4plus_channels() -> None:
    """``channels >= 4`` bumps confidence from 0.75 to 0.85."""
    rule = AudioMultichannelDoublingRule({})
    stereo = _deepgram_record(channels=2, multichannel=True)
    four_channel = _deepgram_record(channels=4, multichannel=True)
    six_channel = _deepgram_record(channels=6, multichannel=True)

    stereo_event = rule.evaluate([stereo], project="p")
    four_event = rule.evaluate([four_channel], project="p")
    six_event = rule.evaluate([six_channel], project="p")

    assert stereo_event is not None and stereo_event.confidence == 0.75
    assert four_event is not None and four_event.confidence == 0.85
    assert six_event is not None and six_event.confidence == 0.85


# ---------------------------------------------------------------------------
# 3. test_does_not_fire_on_mono
# ---------------------------------------------------------------------------


def test_does_not_fire_on_mono() -> None:
    """``channels == 1`` is mono — no multichannel multiplier to charge."""
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=1, multichannel=True)
    assert rule.evaluate([record], project="p") is None


# ---------------------------------------------------------------------------
# 4. test_does_not_fire_when_multichannel_false
# ---------------------------------------------------------------------------


def test_does_not_fire_when_multichannel_false() -> None:
    """``multichannel=False`` on stereo means Deepgram mixes to mono and bills 1x.

    The whole rule's premise is that the customer is paying the 2x
    multiplier because of the flag — when the flag is off, there is no
    waste to flag.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=2, multichannel=False)
    assert rule.evaluate([record], project="p") is None


# ---------------------------------------------------------------------------
# 5. test_does_not_fire_when_multichannel_key_missing
# ---------------------------------------------------------------------------


def test_does_not_fire_when_multichannel_key_missing() -> None:
    """Defensive: ``multichannel`` absent from meta == not-flagged → no fire.

    The wrapper always populates this key, but a buggy custom
    wrapper / test fixture may not. The rule must short-circuit cleanly
    rather than crash or false-positive.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=2, multichannel=None)
    assert rule.evaluate([record], project="p") is None


# ---------------------------------------------------------------------------
# 6. test_does_not_fire_on_non_deepgram_provider
# ---------------------------------------------------------------------------


def test_does_not_fire_on_non_deepgram_provider() -> None:
    """Provider filter: ElevenLabs / Whisper / AssemblyAI / etc. must not fire.

    Even if a non-Deepgram wrapper happens to populate identically-named
    fields, the rule's recommendation
    (``disable_multichannel_or_downmix_to_mono``) doesn't apply to them.
    A generic "channels >= 2" check would mis-fire here; the
    ``provider == "deepgram"`` gate is intentional.
    """
    rule = AudioMultichannelDoublingRule({})
    # Fake "elevenlabs" record with the same shape — only difference is
    # the provider string. The wrapper for elevenlabs would never set
    # ``multichannel`` in this shape, but we simulate a worst-case
    # false-positive that the provider gate blocks.
    record = _deepgram_record(provider="elevenlabs", channels=2, multichannel=True)
    assert rule.evaluate([record], project="p") is None


# ---------------------------------------------------------------------------
# 7. test_does_not_fire_on_chat_methods
# ---------------------------------------------------------------------------


def test_does_not_fire_on_chat_methods() -> None:
    """Method filter: non-transcribe Deepgram methods must not fire.

    The rule recommends a transcription-specific fix
    (``disable_multichannel_or_downmix_to_mono``); applying it to a
    hypothetical future Deepgram chat / TTS / embedding surface would
    be nonsensical. The method allowlist is the gate.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(
        method="some.future.method",
        channels=2,
        multichannel=True,
    )
    assert rule.evaluate([record], project="p") is None


# ---------------------------------------------------------------------------
# 8. test_confidence_boundary_at_2_channels
# ---------------------------------------------------------------------------


def test_confidence_boundary_at_2_channels() -> None:
    """The lower boundary: channels=2 must fire at 0.75 (NOT below the threshold)."""
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=2, multichannel=True)
    event = rule.evaluate([record], project="p")
    assert event is not None
    assert event.confidence == 0.75
    # And channels=3 should ALSO be at 0.75 (below the 4-channel boundary).
    record_3 = _deepgram_record(channels=3, multichannel=True)
    event_3 = rule.evaluate([record_3], project="p")
    assert event_3 is not None
    assert event_3.confidence == 0.75


# ---------------------------------------------------------------------------
# 9. test_confidence_boundary_at_4_channels
# ---------------------------------------------------------------------------


def test_confidence_boundary_at_4_channels() -> None:
    """The boost boundary: channels=4 must hit 0.85 (the 4+ threshold).

    The boundary is closed at 4 (``channels >= 4``); channels=3 stays at
    0.75 per the previous test. Locking the boundary down here so a
    future refactor that flips it to ``channels > 4`` is caught.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=4, multichannel=True)
    event = rule.evaluate([record], project="p")
    assert event is not None
    assert event.confidence == 0.85
    # channels=5 also gets the boost (sanity check that the threshold
    # is "at least 4", not "exactly 4").
    record_5 = _deepgram_record(channels=5, multichannel=True)
    event_5 = rule.evaluate([record_5], project="p")
    assert event_5 is not None
    assert event_5.confidence == 0.85


# ---------------------------------------------------------------------------
# 10. test_suggested_action_set
# ---------------------------------------------------------------------------


def test_suggested_action_set() -> None:
    """``suggested_action`` must be the documented remediation string.

    Locked down because the dashboard renders a per-action help card —
    a rename here would silently break the customer-facing copy.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(channels=2, multichannel=True)
    event = rule.evaluate([record], project="p")
    assert event is not None
    assert event.suggested_action == "disable_multichannel_or_downmix_to_mono"


# ---------------------------------------------------------------------------
# 11. test_evidence_includes_channels_and_multichannel_flag
# ---------------------------------------------------------------------------


def test_evidence_includes_channels_and_multichannel_flag() -> None:
    """Evidence dict must carry the channel count, the flag, and provenance.

    Dashboard renders these fields verbatim — provider, model, method,
    channels, multichannel, duration. The rule's confidence is only
    actionable if the operator can see the underlying numbers.
    """
    rule = AudioMultichannelDoublingRule({})
    record = _deepgram_record(
        channels=2,
        multichannel=True,
        duration_seconds=120.5,
        model="nova-2",
        method="transcribe_url",
    )
    event = rule.evaluate([record], project="p")
    assert event is not None
    assert event.evidence["channels"] == 2
    assert event.evidence["multichannel"] is True
    assert event.evidence["provider"] == "deepgram"
    assert event.evidence["model"] == "nova-2"
    assert event.evidence["method"] == "transcribe_url"
    assert event.evidence["duration_seconds"] == 120.5
    # Wasted channel-seconds = (channels - 1) * duration = 1 * 120.5
    assert event.evidence["wasted_channel_seconds"] == 120.5


# ---------------------------------------------------------------------------
# 12. test_handles_missing_usage_extra_gracefully
# ---------------------------------------------------------------------------


def test_handles_missing_usage_extra_gracefully() -> None:
    """A CallRecord without ``usage_extra`` (legacy / non-wrapper)
    must short-circuit without crashing.

    Pre-wrappers leave ``usage_extra`` empty; the rule has to
    handle that gracefully rather than KeyError. Also covers the case
    where ``usage_extra`` is present but ``model_specific_meta`` is
    missing — same defensive path.
    """
    rule = AudioMultichannelDoublingRule({})

    # 1. usage_extra entirely missing/empty
    no_usage_extra = _deepgram_record(include_usage_extra=False)
    assert rule.evaluate([no_usage_extra], project="p") is None

    # 2. usage_extra present but model_specific_meta missing
    no_meta = _deepgram_record(include_model_specific_meta=False)
    assert rule.evaluate([no_meta], project="p") is None


# ---------------------------------------------------------------------------
# Bonus: ensure the rule short-circuits on empty session and all three
# transcribe methods fire (defensive coverage above the 12 required).
# ---------------------------------------------------------------------------


def test_empty_session_returns_none() -> None:
    """Defensive: empty session should never crash, must return None."""
    rule = AudioMultichannelDoublingRule({})
    assert rule.evaluate([], project="p") is None


def test_all_three_transcribe_methods_fire() -> None:
    """All three Deepgram transcribe entry points must fire the rule."""
    rule = AudioMultichannelDoublingRule({})
    for method in ("transcribe_file", "transcribe_url", "transcribe_live"):
        record = _deepgram_record(method=method, channels=2, multichannel=True)
        event = rule.evaluate([record], project="p")
        assert event is not None, f"method={method} did not fire"
        assert event.evidence["method"] == method
