"""deterministic leak detection rules."""

from __future__ import annotations

from typing import Any

from token_sentinel.rules.audio_multichannel_doubling import (
    AudioMultichannelDoublingRule,
)
from token_sentinel.rules.base import Rule
from token_sentinel.rules.context_bloat import ContextBloatRule
from token_sentinel.rules.embedding_waste import EmbeddingWasteRule
from token_sentinel.rules.model_misroute import ModelMisrouteRule
from token_sentinel.rules.repair_loop import RepairLoopRule
from token_sentinel.rules.rerank_thrash import RerankThrashRule
from token_sentinel.rules.retrieval_thrash import RetrievalThrashRule
from token_sentinel.rules.retry_storm import RetryStormRule
from token_sentinel.rules.tool_definition_bloat import ToolDefinitionBloatRule
from token_sentinel.rules.tool_loop import ToolLoopRule
from token_sentinel.rules.vision_cost_concentration import VisionCostConcentrationRule
from token_sentinel.rules.vision_high_detail_misroute import (
    VisionHighDetailMisrouteRule,
)
from token_sentinel.rules.vision_re_upload import VisionReUploadRule
from token_sentinel.rules.voice_switching_loop import VoiceSwitchingLoopRule
from token_sentinel.rules.zombie import ZombieRule


def default_rules(config: dict[str, Any]) -> list[Rule]:
    return [
        ToolLoopRule(config),
        ContextBloatRule(config),
        EmbeddingWasteRule(config),
        ZombieRule(config),
        ModelMisrouteRule(config),
        RetryStormRule(config),
        ToolDefinitionBloatRule(config),
        RetrievalThrashRule(config),
        # vision rules — gap-closers for the image-video research.
        # None require new wrappers; they read fields already populated by
        # the Anthropic / OpenAI / Gemini wrappers (plus the new
        # ``prompt_tokens_details`` surfaced on Gemini in ).
        VisionReUploadRule(config),
        VisionHighDetailMisrouteRule(config),
        VisionCostConcentrationRule(config),
        # audio rule — reads Deepgram-specific telemetry the
        # wrapper populates on ``CallRecord.usage_extra.model_specific_meta``.
        # Deepgram-only by design; see the rule docstring for the
        # provider-filter rationale.
        AudioMultichannelDoublingRule(config),
        # rules — each reads telemetry the wrappers already
        # populate (zero wrapper-side work, just rule logic). See each
        # rule's module docstring for motivation.
        VoiceSwitchingLoopRule(config),
        RerankThrashRule(config),
        # rules
        RepairLoopRule(config),
    ]


__all__ = [
    "Rule",
    "default_rules",
    "ToolLoopRule",
    "ContextBloatRule",
    "EmbeddingWasteRule",
    "ZombieRule",
    "ModelMisrouteRule",
    "RepairLoopRule",
    "RetryStormRule",
    "ToolDefinitionBloatRule",
    "RetrievalThrashRule",
    "VisionReUploadRule",
    "VisionHighDetailMisrouteRule",
    "VisionCostConcentrationRule",
    "AudioMultichannelDoublingRule",
    "VoiceSwitchingLoopRule",
    "RerankThrashRule",
]
