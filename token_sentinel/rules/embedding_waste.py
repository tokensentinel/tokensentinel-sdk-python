"""Rule: same embedding lookup repeated within session."""

from __future__ import annotations

import hashlib
import json

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule


class EmbeddingWasteRule(Rule):
    name = "embedding_waste"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        # Method-name suffixes we accept as "this is an embedding call":
        #   - ``embeddings.create``  -> OpenAI sync + async embed surface
        #     (and any forks like ``alternative.embeddings.create``)
        #   - ``embed``              -> Voyage AI's ``Client.embed`` /
        #     ``AsyncClient.embed``
        # Future providers (Cohere ``embed``, etc.) drop into the same
        # bucket automatically. The wrapper layer keeps the method label
        # tied to the SDK's actual method name; the rule normalises here.
        embeds = [
            c for c in session if c.method.endswith("embeddings.create") or c.method == "embed"
        ]
        if len(embeds) < 2:
            return None

        seen: dict[str, list[CallRecord]] = {}
        for c in embeds:
            input_data = c.raw_request.get("input")
            if input_data is None:
                continue
            key = hashlib.sha256(
                json.dumps(input_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            seen.setdefault(key, []).append(c)

        for key, group in seen.items():
            if len(group) >= 2:
                wasted_tokens = sum(g.prompt_tokens for g in group[1:])
                return LeakEvent(
                    type="embedding_waste",
                    confidence=0.99,
                    project=project,
                    session_id=session[-1].session_id,
                    rule="v0.embedding_waste",
                    evidence={
                        "duplicate_count": len(group),
                        "input_hash": key[:16],
                        "model": group[0].model,
                        "wasted_tokens": wasted_tokens,
                    },
                    estimated_burn=round(wasted_tokens * 1e-7, 4),
                    suggested_action="add_embedding_cache",
                )
        return None
