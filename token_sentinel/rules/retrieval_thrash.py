"""Rule: a retrieval tool gets called repeatedly with overlapping queries.

This is ``tool_loop`` retuned for the realities of RAG. Three differences:

1. **Scoped to retrieval tools only.** A user-configurable list of substrings
   gates the rule. The default catches the vast majority of LangChain /
   LlamaIndex / Haystack / custom retrievers we surveyed: ``search``,
   ``retrieve``, ``query``, ``lookup``, ``find_documents``, ``vector_search``,
   ``similarity_search``, plus glob-style ``rag_*``, ``*_search``,
   ``*_query``, ``*_retrieve``.

2. **Looser similarity threshold.** Retrieval queries naturally overlap —
   "best Italian restaurants in NYC" vs "Italian restaurants NYC top rated"
   share most tokens but are *legitimate* widening attempts. We use 0.65
   instead of tool_loop's 0.70 (TF-IDF char-n-gram), so we still catch
   genuine thrashing (≥0.65 means "the agent is essentially asking the same
   question") while letting normal query-refinement pass.

3. **Wider window.** Default 120s vs tool_loop's 60s. A retrieval agent
   typically has higher latency per turn — a vector-store roundtrip + a
   reranker + the LLM call — so legitimate refinement loops play out over
   minutes, not seconds.

The remediation is also different: rather than "pause for human review",
the right fix is almost always to **cache the retrieval results** (vector
hits are deterministic for a given query) or **widen the initial query**
so the agent doesn't need to thrash, or **dedupe** if the agent is asking
multiple sub-questions that resolve to the same chunks.

Implementation reuses ``_mean_pairwise_similarity`` from ``tool_loop`` so we
have a single source of truth for the TF-IDF char-n-gram math (and the DoS
caps applied there). The same ``include_raw_args`` /
``max_arg_bytes`` / ``max_total_corpus_bytes`` knobs are honoured here under
the ``retrieval_thrash.*`` config namespace; raw retrieval queries are PII-
sensitive and are redacted by default for the same reasons documented in
``tool_loop``.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from token_sentinel.events import CallRecord, LeakEvent
from token_sentinel.rules.base import Rule
from token_sentinel.rules.tool_loop import (
    DEFAULT_MAX_ARG_BYTES,
    DEFAULT_MAX_TOTAL_CORPUS_BYTES,
    _build_sample_args,
    _estimate_burn,
    _mean_pairwise_similarity,
)

DEFAULT_RETRIEVAL_PATTERNS: tuple[str, ...] = (
    "search",
    "retrieve",
    "query",
    "lookup",
    "find_documents",
    "vector_search",
    "similarity_search",
    "rag_*",
    "*_search",
    "*_query",
    "*_retrieve",
)


class RetrievalThrashRule(Rule):
    name = "retrieval_thrash"

    def evaluate(self, session: list[CallRecord], *, project: str) -> LeakEvent | None:
        if not session:
            return None

        window = self.get("window_seconds", 120)
        min_calls = self.get("min_calls", 3)
        threshold = self.get("cosine_threshold", 0.65)
        metric = self.get("similarity_metric", "tfidf_charngram")
        ngram_size = self.get("charngram_size", 4)
        patterns = tuple(self.get("retrieval_tool_patterns", DEFAULT_RETRIEVAL_PATTERNS))
        max_arg_bytes = self.get("max_arg_bytes", DEFAULT_MAX_ARG_BYTES)
        max_total_corpus_bytes = self.get("max_total_corpus_bytes", DEFAULT_MAX_TOTAL_CORPUS_BYTES)
        include_raw_args = self.get("include_raw_args", False)

        now = session[-1].timestamp
        recent = [c for c in session if (now - c.timestamp).total_seconds() <= window]

        # Group only retrieval-shaped tool invocations by tool name.
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for call in recent:
            for tc in call.tool_calls:
                name = tc.get("name", "unknown")
                if not _is_retrieval_tool(name, patterns):
                    continue
                by_tool.setdefault(name, []).append(tc)

        for tool_name, calls in by_tool.items():
            if len(calls) < min_calls:
                continue

            similarity = _mean_pairwise_similarity(
                calls,
                metric=metric,
                ngram_size=ngram_size,
                max_arg_bytes=max_arg_bytes,
                max_total_corpus_bytes=max_total_corpus_bytes,
            )
            if similarity >= threshold:
                # Wider headroom on confidence than tool_loop because retrieval
                # arguments overlap more by nature — same formula, different
                # baseline so we don't pin every retrieval-thrash event at 0.99.
                confidence = min(0.55 + (similarity - threshold) * 4, 0.95)
                sample_args = _build_sample_args(calls[:3], include_raw=include_raw_args)
                return LeakEvent(
                    type="retrieval_thrash",
                    confidence=confidence,
                    project=project,
                    session_id=session[-1].session_id,
                    rule="v0.retrieval_thrash",
                    evidence={
                        "tool": tool_name,
                        "call_count": len(calls),
                        "window_seconds": window,
                        "mean_similarity": round(similarity, 3),
                        "sample_args": sample_args,
                        "matched_pattern": _matching_pattern(tool_name, patterns),
                    },
                    estimated_burn=_estimate_burn(recent),
                    suggested_action=("cache_retrieval_results_or_widen_initial_query_or_dedupe"),
                )
        return None


def _is_retrieval_tool(name: str, patterns: tuple[str, ...]) -> bool:
    """True iff ``name`` matches any retrieval pattern.

    We support two styles in the same list:
    - **Substring match** (no wildcards): ``"search"`` matches any tool whose
      name contains "search" — ``web_search``, ``search_docs``, ``do_search``.
    - **Glob match** (contains ``*``): ``"rag_*"`` matches anything starting
      with ``rag_``; ``"*_query"`` matches anything ending with ``_query``.

    The substring-match default is intentional. MCP tool names are wildly
    inconsistent across servers (``vector_search``, ``do_search``,
    ``hybrid_search_v2``), and forcing customers to enumerate them all leads
    to the rule never firing. False positives from the substring-match
    default are mitigated by the similarity threshold downstream — a tool
    *named* "search" but invoked with disjoint args won't fire anyway.
    """
    name_lc = name.lower()
    for pat in patterns:
        pat_lc = pat.lower()
        if "*" in pat_lc:
            if fnmatch.fnmatchcase(name_lc, pat_lc):
                return True
        else:
            if pat_lc in name_lc:
                return True
    return False


def _matching_pattern(name: str, patterns: tuple[str, ...]) -> str:
    """Return the first pattern that matched, for evidence reporting."""
    name_lc = name.lower()
    for pat in patterns:
        pat_lc = pat.lower()
        if "*" in pat_lc and fnmatch.fnmatchcase(name_lc, pat_lc):
            return pat
        if "*" not in pat_lc and pat_lc in name_lc:
            return pat
    return ""
