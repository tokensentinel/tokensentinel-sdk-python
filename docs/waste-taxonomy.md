# Leak taxonomy

The eight waste classes detected by V0. Each entry: definition, signal, default thresholds, false-positive hazards, example, and remediation.

## 1. Tool-loop

**Definition.** Agent calls the same tool repeatedly with semantically similar arguments, indicating it has not learned from prior responses.

**Signal.** ≥`min_calls` invocations of the same tool name within `window_seconds`, with mean pairwise similarity of argument JSON ≥`cosine_threshold`.

**V0 similarity.** Token-set Jaccard on argument JSON strings — fast, no model dependency. Acceptable precision for V0.

**V1 similarity.** Sentence-Transformers `all-MiniLM-L6-v2` embeddings + cosine similarity, ~10ms p95 for 100-arg corpus.

**Default thresholds.**
- `min_calls`: 3
- `window_seconds`: 60
- `cosine_threshold`: 0.70 for `tfidf_charngram` (default), 0.85 for `jaccard`. Per-metric defaults — TF-IDF char-n-gram dilutes cosine by document length so equivalent similarity returns lower scores than Jaccard.
- `similarity_metric`: `tfidf_charngram` (default) or `jaccard`
- `charngram_size`: 4 (clamped to 3–5)

** safety knobs.**
- `include_raw_args`: `false` (default) ships a redacted `sample_args` shape `[{"keys": [...], "value_lengths": {...}, "hash": "<16-hex>"}, ...]`. Set `true` to ship the raw arg dicts (use only for local dev / customer-controlled handlers).
- `max_arg_bytes`: 65536 (64KB) — per-arg JSON truncated before n-gram extraction.
- `max_total_corpus_bytes`: 1_048_576 (1MB) — cumulative cap; trailing args dropped.

**False-positive hazards.**
- Legitimate paged tool calls (page 1, page 2, page 3) with similar args. Mitigation: detect monotonic increment in any numeric arg field; suppress.
- Polling tools (e.g., `check_status` until ready). Mitigation: per-customer `polling_tools` allow-list.
- Multi-armed exploration where agent intentionally tries variations. Mitigation: LLM-as-judge ratification.

**Example.**
```
search("web of life game")     ─┐
search("Web of Life game")      ├─ similarity 0.91, 3 calls in 12s → fires
search("\"Web of Life\" game")  ─┘
```

**Suggested action.** `pause_for_human_review` or `add_short_circuit` (e.g., maximum 3 retrieval iterations).

## 2. Context bloat

**Definition.** Prompt-token count per turn is rising over time without progress signal, indicating the agent is carrying forward irrelevant history.

**Signal.** Linear regression slope of prompt_tokens over the last `lookback_turns` is positive and exceeds `slope_threshold` tokens/turn.

**Default thresholds.**
- `lookback_turns`: 10
- `slope_threshold`: 1500 tokens/turn
- `min_turns`: 5 (don't fire on tiny sessions)

**False-positive hazards.**
- Genuinely growing context for legitimate long tasks (multi-step research). Mitigation V1: combine with completion-token entropy — if the agent is producing novel work, suppress.

**Example.** Agent that started at 3k tokens/turn and is now at 19k tokens/turn after 11 turns, carrying forward the full transcript verbatim each turn. Quadratic token growth.

**Suggested action.** `truncate_or_summarize_history`. Customer adds a summarization layer or sliding-window memory.

## 3. Embedding waste

**Definition.** The same input is embedded multiple times within a session.

**Signal.** SHA-256 hash of the embedding `input` argument matches a prior embedding call in the same session.

**Default thresholds.**
- Match: exact hash. No false positives.
- Window: full session.

**False-positive hazards.** None inherent — exact hash match is unambiguous. The only question is whether the customer wants this flagged at all (some workflows intentionally re-embed).

**Example.** Agent embeds `"user query: top 5 movies"` at turn 1. Turn 4, agent embeds the same string again because it didn't cache. Fires.

**Suggested action.** `add_embedding_cache`. Cheap and high-signal — most teams catch this on first integration.

## 4. Zombie agent

**Definition.** An agent run is still firing API calls but has produced no user-facing output for an unusually long time, suggesting it's stuck.

**Signal.** `now() - last_user_facing_output > zombie_threshold` AND ≥`min_recent_calls` API calls in the window. A user-facing output is a non-tool-call response (final assistant message).

**Default thresholds.**
- `threshold_minutes`: 5
- `min_recent_calls`: 5

**False-positive hazards.**
- Long-running legitimate background agents (overnight research). Mitigation: opt-out per session via `sentinel.mark_long_running(session_id)`.

**Example.** Coding agent makes 38 tool calls over 12 minutes without ever returning a user-visible message. Fires at minute 5.

**Suggested action.** `kill_session_or_request_user_input`. The agent is stuck — kill it cleanly or surface a checkpoint to the user.

## 5. Model misroute

**Definition.** A prompt with classification-shaped structure (short, simple, single-output) is being sent to a frontier model when a small model would do.

**Signal.** All four heuristic features:
- Prompt < `max_prompt_tokens`
- Expected output < `max_completion_tokens`
- Prompt contains a classification keyword (`classify`, `is X true`, `yes/no`, `rate from 1 to N`, etc.)
- Routed to: `claude-opus-*`, `claude-sonnet-*`, `gpt-5-*`, `gpt-4-turbo-*`, `gpt-4o`

**Default thresholds.**
- `max_prompt_tokens`: 500
- `max_completion_tokens`: 50

**False-positive hazards.**
- Some classification tasks legitimately need a frontier model (nuanced legal/medical). Mitigation: customer allow-list per route name.

**Example.** Agent sends "Classify this sentence as positive or negative: 'I love this movie'" to Opus. Fires. Suggested action: route to Haiku, save ~190× per call.

**Suggested action.** Routes the customer toward a cheaper alternative on the same family — e.g., `route_to_claude-haiku-4-5` for Claude Sonnet/Opus, `route_to_gpt-5-mini` for GPT-5, `route_to_gemini-2.5-flash` for Gemini 2.5 Pro. The exact string is `route_to_<recommended_alternative>` and the alternative is also exposed in `evidence["recommended_alternative"]`.

## 6. Retry storm

**Definition.** Same call retried multiple times in a window without any change to the parameters, indicating a retry loop or upstream error not being handled.

**Signal.** SHA-256 hash of `(model, messages, tools, max_tokens)` repeats ≥`min_retries` times within `window_seconds`.

**Default thresholds.**
- `min_retries`: 5
- `window_seconds`: 30

**False-positive hazards.**
- Customer-implemented exponential backoff is fine and shouldn't fire often. Mitigation: 5+ retries in 30s suggests no backoff at all.

**Example.** Provider returns 529 (overloaded). Agent retries every 2 seconds with no jitter. Fires at retry 5.

**Suggested action.** `add_backoff_or_check_upstream_health`. The customer's HTTP layer is misbehaving.

## 7. Tool definition bloat

**Definition.** A single request ships an oversized block of tool definitions, burning context on every turn before the model has done any work. The canonical example is an MCP host with many servers attached: 12 servers × ~5 tools each = 58 tools, ~55K tokens of definitions, 72% of a 75K context window gone before the user has typed anything.

**Signal.** On each call, serialize `raw_request['tools']` (or `raw_request['toolConfig']['tools']` for AWS Bedrock Converse) to JSON and count both the tool count and the byte size. Fire if either exceeds its threshold.

**Default thresholds.**
- `tool_count_threshold`: 30 tools
- `tool_definition_bytes_threshold`: 30000 bytes (~7500 tokens at the standard 4-bytes-per-token approximation)
- Confidence: 0.85 if either threshold tripped, 0.95 if both are tripped by ≥50% (e.g., 45 tools at 45KB)

** safety knobs.**
- `max_tool_bytes`: 262144 (256KB) — per-tool serialization truncated. Original byte count is still used for the threshold comparison so the rule fires correctly on a single huge tool.
- `max_total_bytes`: 5_242_880 (5MB) — if cumulative serialized size exceeds this, short-circuits to a 0.50-confidence event with `evidence={"truncated": True, ...}` and `suggested_action="reduce_tool_count_or_use_lazy_tool_loading"`. Customer still gets a signal; the SDK doesn't crunch gigabytes.

**False-positive hazards.**
- Legitimate one-shot mass-tool-import calls (e.g., a manual tool-registry dump). Currently fires; V1 adds session-context awareness so a single mass-import call after a quiet session is treated differently from sustained mass-import on every turn.
- A single big tool with a huge JSON schema (e.g., one `submit` tool with 200 enum values). Same fix applies — prune the schema — but V1 may surface this as a different signal.

**Example.** Claude Desktop with filesystem + browser + slack + jira + linear + sentry + supabase MCP servers all attached. Each user message ships 58 tool definitions (~55K tokens). Even before the user finishes their question, ~72% of context is gone.

**Suggested action.** `reduce_tool_count_or_use_lazy_tool_loading`. Either: prune the connected MCP servers per-task; split the agent into multiple narrower agents; or move to a lazy-tool-loading host that only injects definitions for tools the model actually intends to call (the MCP spec for this is in flight as of May 2026).

## 8. Retrieval thrash

**Definition.** A retrieval-shaped tool is called repeatedly with overlapping queries within a window — the agent is asking essentially the same retrieval question multiple ways instead of caching, widening, or deduping.

**Signal.** A tool-loop variant scoped to retrieval-shaped tool names. Same TF-IDF char-n-gram cosine similarity as `tool_loop`, but with looser thresholds calibrated to retrieval's natural query overlap.

**Default thresholds.**
- `min_calls`: 3
- `window_seconds`: 120 (longer than tool_loop's 60s — retrieval pipelines are slower per turn)
- `cosine_threshold`: 0.65 (looser than tool_loop's 0.70 — retrieval queries overlap naturally)
- `retrieval_tool_patterns`: `('search', 'retrieve', 'query', 'lookup', 'find_documents', 'vector_search', 'similarity_search', 'rag_*', '*_search', '*_query', '*_retrieve')` — substring match for entries without `*`, glob match for entries containing `*`. Customer-configurable.

** safety knobs.** Same redaction + cap controls as `tool_loop`: `include_raw_args` (default `false`), `max_arg_bytes` (64KB), `max_total_corpus_bytes` (1MB). Retrieval queries often contain user PII, so the privacy-by-default redaction is especially important here.

**False-positive hazards.**
- Legitimate query-refinement loops where the agent intentionally widens or narrows. Mitigation: 0.65 is loose enough to catch real thrash but the threshold is configurable per-customer.
- Multi-tenant workflows where the same tool name handles distinct queries per turn. Mitigation: V1 adds tenant-scoped session keys.
- Tools whose names match a retrieval pattern but aren't actually retrieval (e.g., a `query_status` tool). Mitigation: per-customer `retrieval_tool_patterns` override.

**Example.**
```
vector_search("kubernetes pod stuck pending")        ─┐
vector_search("k8s pod pending state troubleshoot")   ├─ similarity 0.71, 4 calls in 90s → fires
vector_search("kubernetes scheduler pod pending")    ─┤
vector_search("pod stuck pending kubernetes debug")  ─┘
```

**Suggested action.** `cache_retrieval_results_or_widen_initial_query_or_dedupe`. Concretely: add an LRU cache keyed on the embedded query vector; or have the agent issue one widened initial query and re-rank locally; or dedupe by chunk-hash before sending to the LLM.

## V1-only anti-patterns (deferred)

These need LLM-as-judge to detect reliably and are NOT in V0:

- **Semantic loop**: agent says effectively the same thing in different words across turns.
- **Hallucinated tool retry**: agent calls a tool that returned an error, hallucinates the response, continues without re-trying.
- **Verbose chain-of-thought leak**: agent reasons in expensive tokens when concise output would do.
- **Premature compaction**: customer's compaction step kicks in too aggressively, costing more in re-summarization than the saved tokens.

LLM-as-judge takes the gray-zone V0 firings and the above anti-patterns as inputs.

## Severity levels

Every leak event carries a `confidence` score 0.0–1.0. Default routing:

| Confidence | Default action |
|---|---|
| < 0.5 | discarded (not emitted) |
| 0.5–0.75 | logged only, even in `alert` mode |
| 0.75–0.9 | emitted in `alert` mode |
| ≥ 0.9 | emitted in all modes; eligible for `block` |

Customers tune this in `Sentinel(config={...})`. The defaults err toward **fewer false positives at the cost of some false negatives** — the cardinal rule of detection products.
