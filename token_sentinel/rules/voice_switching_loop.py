"""Rule: ElevenLabs voice-switching loop .

When an agent synthesizes the SAME text against MULTIPLE different
``voice_id`` values inside a short window, it is almost always a
voice-selection experimentation pattern that leaked into production:

  - A developer was comparing voices in a notebook and forgot to pin
    one before shipping.
  - An agent loop is re-rolling voices on a "vibe check" / quality
    classifier output, never landing on a final choice.
  - A UI A/B test was deployed without a cohort gate, so every
    end-user receives 3-4 voices for the same prompt.

At a per-character billing rate, even short experimentation loops can
run up real dollar amounts — every additional voice replays the entire
input text against ElevenLabs' synthesis engine.

The ElevenLabs wrapper (``wrappers/elevenlabs.py``) populates
``CallRecord.usage_extra.model_specific_meta`` with ``voice_id`` and
``text_hash`` (SHA-256 of the input text, truncated to 16 hex chars) for
exactly this rule. The hash is what we read here — the raw text never
leaves the customer's process, so the rule operates on a privacy-safe
fingerprint.

Firing conditions
-----------------

The rule fires when within ``window_seconds`` (default 10s) BEFORE the
most recent call, the same ``text_hash`` appears across ``>= N`` (default
3) DISTINCT ``voice_id`` values, all on ``provider == "elevenlabs"`` and
``method`` starting with ``text_to_speech.`` (matches ``convert``,
``convert_as_stream``, and ``stream``).

The window is anchored on ``session[-1].timestamp`` so the rule fires
on the call that PUSHES the cluster across the threshold — not later,
when the cluster has aged out of the window.

Confidence
----------

- Base: **0.7** at exactly ``N`` distinct voices.
- ``+0.05`` per additional voice beyond ``N``, capped at **0.9**.

The cap is deliberate: at 5+ distinct voices in 10s the customer is
unambiguously thrashing, but the rule is best surfaced as "high
confidence" rather than 1.0 because there are rare legitimate uses
(an end-user side-by-side voice comparison UI, a multi-character story
narration where the same line is read by each character). The cap
prevents pinning all events at the maximum so the dashboard's
confidence histogram stays informative.

Suggested action
----------------

``lock_voice_id_or_cache_synthesis`` — the two valid remediations:
  1. Pin a single ``voice_id`` for the production code path. Removes
     the loop entirely.
  2. If a voice loop is intentional (rare), cache the per-voice
     synthesis result by ``(voice_id, text_hash)`` so the second
     iteration is free.

Evidence
--------

  - ``voice_ids_observed``: sorted list of the distinct voice_ids in
    the cluster — operators can see which voices the agent was
    bouncing between.
  - ``text_hash``: the truncated 16-hex SHA-256 fingerprint. Privacy-
    safe; matches the hash on every CallRecord in the cluster so
    operators can correlate dashboard rows.
  - ``call_count``: number of calls in the cluster.
  - ``time_span_seconds``: oldest-to-newest gap, rounded to 0.1s.
"""

from __future__ import annotations

from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Method-name prefix matched against ``CallRecord.method``. The
# ElevenLabs wrapper writes labels like ``"text_to_speech.convert"``,
# ``"text_to_speech.convert_as_stream"``, ``"text_to_speech.stream"``.
# Matching by prefix keeps the rule resilient to future SDK
# additions that add a fourth ``text_to_speech.*`` method label.
_METHOD_PREFIX: str = "text_to_speech."

# Approximate ElevenLabs price per character for the May 2026 Turbo
# default. Used purely for the ``estimated_burn`` rendering on the
# LeakEvent. Waste estimate = char_count * (extra_voices_beyond_one).
_ELEVENLABS_USD_PER_CHAR: float = 0.00018 / 1  # ~$0.18/1k chars Turbo


class VoiceSwitchingLoopRule(Rule):
    """Fires when the same text is synthesized against many voices fast.

    See module docstring for the full motivation. Operates by scanning
    ALL ElevenLabs TTS calls in the window ending at
    ``session[-1].timestamp``; groups by ``text_hash``; fires on the
    first group whose distinct-voice-count crosses the threshold.
    """

    name = "voice_switching_loop"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window = self.get("window_seconds", 10)
        min_voices = self.get("min_voices", 3)

        # Anchor the window on the most recent call so the rule fires
        # at the moment the cluster crosses the threshold. Older calls
        # outside the window are ignored — voice experimentation that
        # happened a minute ago and stopped is not a current leak.
        now = session[-1].timestamp
        recent: list[CallRecord] = []
        for c in session:
            if c.provider != "elevenlabs":
                continue
            if not c.method.startswith(_METHOD_PREFIX):
                continue
            if (now - c.timestamp).total_seconds() > window:
                continue
            recent.append(c)

        if not recent:
            return None

        # Group by text_hash. For each group, collect the set of distinct
        # voice_ids. Fire on the first group whose distinct-voice count
        # crosses min_voices.
        groups: dict[str, list[CallRecord]] = {}
        for c in recent:
            meta = _extract_model_specific_meta(c)
            if meta is None:
                continue
            text_hash = meta.get("text_hash")
            voice_id = meta.get("voice_id")
            if not isinstance(text_hash, str) or not text_hash:
                continue
            if not isinstance(voice_id, str) or not voice_id:
                continue
            groups.setdefault(text_hash, []).append(c)

        for text_hash, group in groups.items():
            voice_ids = _distinct_voice_ids(group)
            if len(voice_ids) < min_voices:
                continue

            # Confidence: 0.7 baseline, +0.05 per extra voice beyond
            # threshold, capped at 0.9.
            extra_voices = len(voice_ids) - min_voices
            confidence = min(0.7 + 0.05 * extra_voices, 0.9)

            # Time span: oldest-to-newest in the cluster, rounded to 0.1s.
            timestamps = sorted(c.timestamp for c in group)
            time_span_seconds = round((timestamps[-1] - timestamps[0]).total_seconds(), 1)

            # Burn estimate: each extra voice replays the entire text
            # against the ElevenLabs synthesizer. Sum char counts from
            # the calls except the first (the first is the one the
            # customer "should have" stopped at). Char count is on
            # ``usage_extra.dimension_value`` per the  schema.
            wasted_chars = _sum_wasted_chars(group)
            estimated_burn = round(wasted_chars * _ELEVENLABS_USD_PER_CHAR, 4)

            return LeakEvent(
                type="voice_switching_loop",
                confidence=confidence,
                project=project,
                session_id=session[-1].session_id,
                rule="v0.voice_switching_loop",
                evidence={
                    "voice_ids_observed": voice_ids,
                    "text_hash": text_hash,
                    "call_count": len(group),
                    "time_span_seconds": time_span_seconds,
                },
                estimated_burn=estimated_burn,
                suggested_action="lock_voice_id_or_cache_synthesis",
            )
        return None


def _extract_model_specific_meta(call: CallRecord) -> dict[str, Any] | None:
    """Return ``call.usage_extra["model_specific_meta"]`` if it's a dict.

    Defensive lookup so a malformed ``usage_extra`` (None, list, str) or
    a missing ``model_specific_meta`` key short-circuits the rule
    cleanly rather than crashing the rule loop.
    """
    usage_extra = getattr(call, "usage_extra", None)
    if not isinstance(usage_extra, dict):
        return None
    meta = usage_extra.get("model_specific_meta")
    if not isinstance(meta, dict):
        return None
    return meta


def _distinct_voice_ids(group: list[CallRecord]) -> list[str]:
    """Return a sorted list of unique voice_ids in ``group``.

    Sorted for deterministic evidence shape — the evidence dict is part
    of the wire contract with the cloud sink, and unstable ordering would
    show up as spurious diffs in golden-master tests.
    """
    seen: set[str] = set()
    for c in group:
        meta = _extract_model_specific_meta(c)
        if meta is None:
            continue
        voice_id = meta.get("voice_id")
        if isinstance(voice_id, str) and voice_id:
            seen.add(voice_id)
    return sorted(seen)


def _sum_wasted_chars(group: list[CallRecord]) -> int:
    """Sum the char counts on all but the first call in the cluster.

    The "first" call is the one the customer would have made anyway;
    every additional call against a different voice is the waste. Char
    count lives on ``usage_extra.dimension_value`` (per the  schema:
    ElevenLabs ``dimension_kind`` is ``"per_character"``).
    """
    if len(group) <= 1:
        return 0
    # Iterate in arrival order so "first" is consistent with how the
    # customer experienced the loop.
    total = 0
    for c in group[1:]:
        usage_extra = getattr(c, "usage_extra", None)
        if not isinstance(usage_extra, dict):
            continue
        value = usage_extra.get("dimension_value")
        if isinstance(value, (int, float)) and value >= 0:
            total += int(value)
    return total
