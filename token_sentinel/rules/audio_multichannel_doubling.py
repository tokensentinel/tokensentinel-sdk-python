"""Rule: Deepgram multichannel doubling on stereo (or higher-channel) audio.

Deepgram's pricing model bills **per-second of audio multiplied by the
number of channels transcribed**. The Deepgram dashboard surfaces an
aggregate "minutes" meter and does NOT break the figure down by channel,
so customers who set ``multichannel=True`` on a stereo recording silently
pay double for the entire history of the integration. A 60s stereo file
with ``multichannel=True`` is billed as 120 audio-seconds; a 60s 4-channel
field recording with ``multichannel=True`` is billed as 240 audio-seconds.

This is a real dollar-meaningful leak that today's observability tools do
not catch:

- The vendor dashboard shows aggregate minutes, not channel-level
  breakdown — operators rarely realize their bill is 2-4x what they
  expected.
- The ``multichannel=True`` flag is enabled speculatively by most agents
  ("might as well, the model can sort it out"); in practice, the second
  channel is usually redundant mixed audio rather than a separate speaker
  whose transcript the agent actually consumes.
- Downmixing to mono before sending OR setting ``multichannel=False``
  and relying on diarization for speaker separation is the right fix
  in 90%+ of cases — but customers don't make that call until someone
  flags the burn.

The Deepgram wrapper (see ``wrappers/deepgram.py``) populates the
``CallRecord.usage_extra.model_specific_meta`` field with the
``channels`` and ``multichannel`` flags exactly so this rule can match
on them without inspecting raw response bytes.

Firing conditions
-----------------

The rule fires when ALL of the following are true on the most recent
``CallRecord`` in the session:

  1. ``provider == "deepgram"`` — Deepgram is the only -scope STT
     provider that exposes a ``multichannel`` knob with this billing
     semantics. OpenAI Whisper (Agent D's  extension) does not have
     a comparable flag — it always transcribes a single mixed track and
     bills per-second regardless of source channel count. AssemblyAI
     bills per-second of input audio regardless of channel count.
     Restricting the rule to Deepgram is a deliberate provider-specific
     choice rather than a generic "channels >= 2" check, because the
     other STT vendors do NOT have the underlying waste pattern this
     rule diagnoses. A generic check would mis-fire on Whisper /
     AssemblyAI traffic and recommend an action that does not apply.
  2. ``method`` in ``{"transcribe_file", "transcribe_url",
     "transcribe_live"}`` — the three Deepgram entry points wrapped in
     . Chat / non-STT methods (e.g., a future ``embeddings.create``
     surface) MUST NOT trigger this rule even if the metadata shape
     happens to match.
  3. ``usage_extra.model_specific_meta`` is a dict containing
     ``channels >= 2`` AND ``multichannel is True``. The Deepgram wrapper
     guarantees both keys are present when the request was a transcribe
     call; defensive lookups (``.get`` with default) handle the unlikely
     case where the wrapper or a custom subclass omits them.

Confidence
----------

- Base confidence: **0.75** for stereo (``channels == 2``). The customer
  may genuinely have two separate speakers on two channels they want
  diarized separately, in which case the cost is intentional — but the
  vast majority of stereo audio sent to Deepgram with
  ``multichannel=True`` is just a mixed-down stereo recording where the
  flag adds no value.
- Boosted to **0.85** when ``channels >= 4``. Four or more channels is
  almost always a field-recording rig (lavaliers + ambient + boom) where
  the agent rarely consumes per-channel transcripts — and the cost
  multiplier (4-8x) is large enough that even a 25% false-positive rate
  is worth the operator's attention.

Suggested action
----------------

``disable_multichannel_or_downmix_to_mono`` — the two valid remediations.
Operators who genuinely need per-speaker output should switch to
``multichannel=False`` + ``diarize=True``, which costs the same as mono
transcription while still attributing utterances to speakers via
Deepgram's speaker-clustering pass.

Scope note
---------- ships this as a **Deepgram-only** rule. If + adds wrappers for
other STT providers that expose a comparable per-channel billing knob
(Speechmatics, Symbl.ai), the rule's provider check should grow to a
set rather than be cloned per-provider — but that is a future-cycle
concern, not a  one.
"""

from __future__ import annotations

from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Method labels written by the Deepgram wrapper. We import-by-value
# rather than importing the constants from ``wrappers.deepgram`` to keep
# the rules package free of any wrapper-import dependency (the wrapper
# imports the SDK lazily, but the rule must never touch it). The values
# are pinned here as the public contract — any wrapper that ships
# Deepgram telemetry must use these exact strings.
_DEEPGRAM_TRANSCRIBE_METHODS: frozenset[str] = frozenset(
    {
        "transcribe_file",
        "transcribe_url",
        "transcribe_live",
    }
)

# Approximate Deepgram price per audio-second for the Nova-2 model family
# (the May 2026 default). Used purely for the ``estimated_burn`` rendering
# on the LeakEvent — the rule does not depend on the exact figure being
# right, only on it being roughly representative so dashboards show a
# meaningful dollar number. The waste estimate is the *extra* channels
# beyond mono: a 2-channel call wastes one channel's worth of seconds.
_DEEPGRAM_USD_PER_SECOND: float = 0.0043 / 60  # ~$0.0043/min Nova-2 pre-recorded


class AudioMultichannelDoublingRule(Rule):
    """Fires when a Deepgram transcribe call paid the multichannel multiplier.

    See module docstring for the full motivation. Operates on
    ``session[-1]`` exactly like other Deepgram-aware rules — multichannel
    waste is a per-call property, not a session-level pattern, so there
    is no value in scanning earlier calls.
    """

    name = "audio_multichannel_doubling"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None
        c = session[-1]

        # Provider filter: Deepgram-only. See module docstring for the
        # reasoning behind the per-provider gate.
        if c.provider != "deepgram":
            return None

        # Method filter: only the three transcribe entry points.
        if c.method not in _DEEPGRAM_TRANSCRIBE_METHODS:
            return None

        # Inspect ``model_specific_meta`` defensively — the wrapper
        # always populates it, but tests / custom subclasses / future
        # SDK shapes might not.
        meta = _extract_model_specific_meta(c)
        if meta is None:
            return None
        channels = meta.get("channels")
        multichannel = meta.get("multichannel")
        if not isinstance(channels, int) or channels < 2:
            return None
        if multichannel is not True:
            return None

        # Confidence: 0.75 baseline, 0.85 when 4+ channels (almost always
        # waste at that count — see docstring).
        confidence = 0.85 if channels >= 4 else 0.75

        # Burn estimate: the extra channels beyond mono are wasted. For
        # stereo that's 1× duration; for 4-channel that's 3× duration.
        # The Deepgram wrapper records ``dimension_value`` (audio-seconds)
        # in ``usage_extra``.
        usage_extra = c.usage_extra if isinstance(c.usage_extra, dict) else {}
        duration_seconds = usage_extra.get("dimension_value")
        if not isinstance(duration_seconds, (int, float)) or duration_seconds < 0:
            duration_seconds = 0.0
        wasted_channel_seconds = float(duration_seconds) * (channels - 1)
        estimated_burn = round(wasted_channel_seconds * _DEEPGRAM_USD_PER_SECOND, 4)

        return LeakEvent(
            type="audio_multichannel_doubling",
            confidence=confidence,
            project=project,
            session_id=c.session_id,
            rule="v0.audio_multichannel_doubling",
            evidence={
                "provider": c.provider,
                "model": c.model,
                "method": c.method,
                "channels": channels,
                "multichannel": True,
                "duration_seconds": float(duration_seconds),
                "wasted_channel_seconds": round(wasted_channel_seconds, 3),
            },
            estimated_burn=estimated_burn,
            suggested_action="disable_multichannel_or_downmix_to_mono",
        )


def _extract_model_specific_meta(call: CallRecord) -> dict[str, Any] | None:
    """Return ``call.usage_extra["model_specific_meta"]`` if it's a dict, else None.

    Defensive lookup so a malformed ``usage_extra`` (None, list, str) or a
    missing ``model_specific_meta`` key short-circuits the rule cleanly
    rather than crashing the rule loop.
    """
    usage_extra = getattr(call, "usage_extra", None)
    if not isinstance(usage_extra, dict):
        return None
    meta = usage_extra.get("model_specific_meta")
    if not isinstance(meta, dict):
        return None
    return meta
