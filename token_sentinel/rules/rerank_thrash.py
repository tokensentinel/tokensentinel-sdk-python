"""Rule: Cohere rerank thrashing .

When an agent calls Cohere's ``rerank`` endpoint MULTIPLE times against
the SAME ``(query, documents)`` pair inside a short window, every call
after the first is wasted spend:

  - Cohere's rerank scoring is deterministic for a given
    ``(model, query, document_set)`` triple. The second call returns
    the same top-K and the same scores.
  - The customer is billed per search-unit (one query + up to 100
    documents). Re-running the same rerank doubles the bill for zero
    incremental information.
  - This is the rerank-analog of the ``embedding_waste`` rule —
    embeddings cache by ``(model, input_hash)``, reranks cache by
    ``(model, query+documents_hash)``. Customers who write the first
    cache layer usually forget the second.

Common root causes:
  - A retrieval loop re-runs the same rerank step after each LLM
    refinement, even when the upstream candidate set hasn't changed.
  - An agent retry path replays a tool-call sequence whose rerank step
    is already deterministic.
  - A multi-agent coordinator forwards the same rerank request through
    multiple supervisor nodes without dedup.

The Cohere wrapper (``wrappers/cohere.py``) captures
``raw_request["query"]`` + ``raw_request["documents"]`` AND a
``request_hash`` keyed on ``(model, query, documents)``. This rule
reads ``request_hash`` — same key the retry-storm rule uses, but
scoped to rerank-only methods and with a different window /
remediation.

Firing conditions
-----------------

The rule fires when within ``window_seconds`` (default 30s) BEFORE
the most recent call, the same ``request_hash`` appears across ``>= N``
(default 2) calls on ``provider == "cohere"`` ``method == "rerank"``.

Default ``N=2`` is intentional. Unlike ``retry_storm`` (which uses
``min_retries=5`` because LLM chat retries can be legitimate),
**rerank thrashing has no legitimate use case at all**: every rerank
above the first against the same ``(query, documents)`` is pure waste.
Firing at N=2 gives operators the earliest possible signal.

The window is anchored on ``session[-1].timestamp`` so the rule fires
on the call that creates the duplicate, not later.

Confidence
----------

- Base: **0.75** at exactly N calls.
- ``+0.05`` per repetition beyond N, capped at **0.9**.

The cap is deliberate: at 5+ identical reranks in 30s the customer
is unambiguously thrashing, but 1.0 is reserved for rules with zero
documented false-positive vectors. The cap keeps the confidence
histogram informative.

Suggested action
----------------

``cache_rerank_results_by_query_hash`` — the canonical remediation:
add a ``(model, query_hash, docs_hash) -> reranked_result`` cache
between the agent's rerank call and the Cohere client. The hash is
already computed by the wrapper (``request_hash``) so customers
can copy-paste it from the CallRecord's evidence into their cache key.

Evidence
--------

  - ``request_hash``: truncated 16-hex hash of the ``(model, query,
    documents)`` triple. Privacy-safe (raw query and docs never leave
    the customer's process). Same value across all calls in the
    cluster so operators can correlate.
  - ``call_count``: number of calls in the cluster.
  - ``time_span_seconds``: oldest-to-newest gap, rounded to 0.1s.
  - ``model``: the rerank model name (e.g. ``rerank-english-v3.0``).
"""

from __future__ import annotations

from collections import defaultdict

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule

# Approximate Cohere price per rerank search-unit for May 2026
# rerank-english-v3.0. One search-unit covers (1 query + up to 100
# documents); a thrashed rerank wastes one search-unit per extra call.
# Used purely for the ``estimated_burn`` rendering on the LeakEvent.
_COHERE_USD_PER_SEARCH_UNIT: float = 0.002  # ~$2/1k search-units


class RerankThrashRule(Rule):
    """Fires when the same rerank is run multiple times in a window.

    See module docstring for the full motivation. Operates by scanning
    Cohere rerank calls in the window ending at ``session[-1].timestamp``;
    groups by ``request_hash``; fires on the first group whose call
    count crosses the threshold.
    """

    name = "rerank_thrash"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window = self.get("window_seconds", 30)
        min_calls = self.get("min_calls", 2)

        # Anchor the window on the most recent call so the rule fires
        # at the moment the cluster crosses the threshold.
        now = session[-1].timestamp
        recent: list[CallRecord] = []
        for c in session:
            if c.provider != "cohere":
                continue
            if c.method != "rerank":
                continue
            if (now - c.timestamp).total_seconds() > window:
                continue
            if not isinstance(c.request_hash, str) or not c.request_hash:
                continue
            recent.append(c)

        if not recent:
            return None

        # Group by request_hash. Fire on the first group whose call
        # count crosses min_calls.
        groups: dict[str, list[CallRecord]] = defaultdict(list)
        for c in recent:
            groups[c.request_hash].append(c)

        for request_hash, group in groups.items():
            if len(group) < min_calls:
                continue

            # Confidence: 0.75 baseline, +0.05 per repetition beyond
            # threshold, capped at 0.9.
            extra_calls = len(group) - min_calls
            confidence = min(0.75 + 0.05 * extra_calls, 0.9)

            # Time span: oldest-to-newest in the cluster, rounded to 0.1s.
            timestamps = sorted(c.timestamp for c in group)
            time_span_seconds = round((timestamps[-1] - timestamps[0]).total_seconds(), 1)

            # Burn estimate: every call after the first is wasted. One
            # search-unit per call (Cohere bills the unit, not the doc
            # count, so this is precise).
            wasted_search_units = len(group) - 1
            estimated_burn = round(wasted_search_units * _COHERE_USD_PER_SEARCH_UNIT, 4)

            return LeakEvent(
                type="rerank_thrash",
                confidence=confidence,
                project=project,
                session_id=session[-1].session_id,
                rule="v0.rerank_thrash",
                evidence={
                    "request_hash": request_hash[:16],
                    "call_count": len(group),
                    "time_span_seconds": time_span_seconds,
                    "model": group[0].model,
                },
                estimated_burn=estimated_burn,
                suggested_action="cache_rerank_results_by_query_hash",
            )
        return None
