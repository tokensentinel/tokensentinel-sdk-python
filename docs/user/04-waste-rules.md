# Leak rules

TokenSentinel V0 ships eight deterministic waste detection rules. Each is a pure function of the in-process per-session ring buffer plus your config — no I/O, no network calls, sub-millisecond p95 per rule.

This page is the user-facing reference: what each rule detects, when it fires, default thresholds, how to tune, and a sample event payload. For the design rationale and false-positive analysis, see [`docs/04_leak_taxonomy.md`](../04_leak_taxonomy.md) at the repo root.

## Quick index

| Rule | Default confidence | Catches |
|---|---|---|
| [`tool_loop`](#1-tool_loop) | 0.6–0.99 | Same tool, near-identical args, ≥3 calls in 60s. |
| [`context_bloat`](#2-context_bloat) | 0.55–0.95 | Prompt tokens per turn rising past `slope_threshold`. |
| [`embedding_waste`](#3-embedding_waste) | 0.99 | Same embedding input embedded twice in a session (exact hash). |
| [`zombie`](#4-zombie) | 0.75 | Agent silent for ≥5 minutes while still firing API calls. |
| [`model_misroute`](#5-model_misroute) | 0.7 | Classification-shaped prompt sent to a frontier model. |
| [`retry_storm`](#6-retry_storm) | 0.9 | Same call retried ≥5 times in 30s, no parameter change. |
| [`tool_definition_bloat`](#7-tool_definition_bloat) | 0.85 / 0.95 | A single request ships ≥30 tool defs or ≥30KB of tool JSON. |
| [`retrieval_thrash`](#8-retrieval_thrash) | 0.55–0.95 | Retrieval tool called repeatedly with overlapping queries. |

All rules emit a [`LeakEvent`](./07-api-reference.md#leakevent) with the same shape. The `evidence` dict is rule-specific.

Tune any rule's thresholds via the project `config` dict. The pattern is:

```python
Sentinel(
    project="my-agent",
    config={
        "<rule_name>.<config_key>": value,
    },
)
```

For example, `"tool_loop.cosine_threshold": 0.80`.

To disable a rule, omit it from the `rules=` list:

```python
Sentinel(project="my-agent", rules=["embedding_waste", "retry_storm"])  # only these two
```

---

## 1. `tool_loop`

**What it detects.** An agent calling the same tool repeatedly with semantically similar arguments — a strong signal that the agent has not learned from prior responses.

**When it fires.** ≥`min_calls` invocations of the same tool name within `window_seconds`, where the mean pairwise similarity of the argument JSON strings is ≥`cosine_threshold`. Similarity is computed with TF-IDF char-n-gram cosine by default — pure Python, no model dependency, calibrated to catch paraphrases like `"web of life"` vs `"\\"Web of Life\\""` while letting genuinely different queries pass.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `tool_loop.window_seconds` | `60` | Look-back window for grouping calls. |
| `tool_loop.min_calls` | `3` | Minimum same-tool invocations before evaluating similarity. |
| `tool_loop.cosine_threshold` | `0.70` (TF-IDF) / `0.85` (Jaccard) | Mean pairwise similarity that triggers firing. |
| `tool_loop.similarity_metric` | `"tfidf_charngram"` | Either `"tfidf_charngram"` (default) or `"jaccard"`. |
| `tool_loop.charngram_size` | `4` | Character n-gram size, clamped to 3–5. |

**How to tune.**

```python
Sentinel(
    project="my-agent",
    config={
        "tool_loop.cosine_threshold": 0.80,   # stricter — fewer fires, fewer false positives
        "tool_loop.min_calls": 5,             # require more repetition before firing
        "tool_loop.window_seconds": 120,      # widen look-back for slower agents
    },
)
```

If you have an agent that legitimately calls a polling tool (`check_status` until ready) and `tool_loop` is firing on it, the cleanest fix is to disable `tool_loop` for that project entirely and rely on `retry_storm` (which is exact-hash, not similarity-based) or wait until V1 lands per-tool allow-lists.

**Sample event.**

```python
LeakEvent(
    type="tool_loop",
    confidence=0.84,
    project="my-agent",
    session_id="user-42-task-17",
    rule="v0.tool_loop",
    evidence={
        "tool": "web_search",
        "call_count": 3,
        "window_seconds": 60,
        "mean_similarity": 0.91,
        "sample_args": [
            {"query": "web of life game"},
            {"query": "Web of Life game"},
            {"query": '"Web of Life" game'},
        ],
    },
    estimated_burn=0.0324,
    suggested_action="pause_for_human_review",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Polling-heavy agents (every 10 seconds, same tool, same args). Multi-armed exploration agents that intentionally vary the same query. Paged-call agents (page 1, page 2, page 3 with otherwise-identical args).

---

## 2. `context_bloat`

**What it detects.** Prompt-token count per turn rising over time without progress — the agent is carrying forward irrelevant history each turn.

**When it fires.** A linear regression on the prompt-token counts of the last `lookback_turns` calls in a session yields a positive slope greater than `slope_threshold` (tokens per turn). Requires at least `min_turns` calls in the session before evaluating.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `context_bloat.lookback_turns` | `10` | How many recent calls to fit the slope on. |
| `context_bloat.slope_threshold` | `1500` | Tokens-per-turn slope that triggers firing. |
| `context_bloat.min_turns` | `5` | Minimum session length before evaluating. |

**How to tune.**

```python
Sentinel(
    project="my-agent",
    config={
        "context_bloat.slope_threshold": 3000,  # only fire on aggressive growth
        "context_bloat.min_turns": 10,          # don't evaluate short sessions
    },
)
```

Agents doing genuine multi-step research will have growing context that is not waste. Raise `slope_threshold` if your traffic skews this way, or disable the rule entirely for projects you know have long-context legitimate usage.

**Sample event.**

```python
LeakEvent(
    type="context_bloat",
    confidence=0.70,
    project="my-agent",
    session_id="research-task-9",
    rule="v0.context_bloat",
    evidence={
        "tokens_per_turn_slope": 2280.4,
        "first_turn_tokens": 3120,
        "last_turn_tokens": 19840,
        "turns_evaluated": 8,
    },
    estimated_burn=0.1026,
    suggested_action="truncate_or_summarize_history",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Long-running research agents where the growing context is the work product. Agents with explicit summarization/compaction layers that legitimately ratchet context up between compactions.

---

## 3. `embedding_waste`

**What it detects.** The same embedding input is embedded multiple times within a session.

**When it fires.** Two or more calls with `method == "embeddings.create"` in the same session whose hashed `raw_request["input"]` is identical. Match is exact SHA-256, not semantic — so this rule has effectively zero false positives.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| (none) | — | This rule has no tunable thresholds — it fires on exact-hash duplicates. |

**How to tune.** This rule has no thresholds. If you have a workflow that intentionally re-embeds, disable the rule:

```python
Sentinel(project="my-agent", rules=["tool_loop", "context_bloat", "retry_storm", ...])  # omit embedding_waste
```

Most teams catch their first embedding-waste leak within minutes of installing TokenSentinel — that is by design. The remediation is almost always a small LRU cache keyed on the embedding input.

**Sample event.**

```python
LeakEvent(
    type="embedding_waste",
    confidence=0.99,
    project="rag-pipeline",
    session_id="ingest-batch-2026-05-07",
    rule="v0.embedding_waste",
    evidence={
        "duplicate_count": 3,
        "input_hash": "9f8c2a1e4b3d7f60",
        "model": "text-embedding-3-small",
        "wasted_tokens": 24,
    },
    estimated_burn=0.0000,
    suggested_action="add_embedding_cache",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

Note: at `text-embedding-3-small` pricing the dollar value rounds to zero per duplicate; the value here is mostly the *signal*. In a chunky ingest pipeline with thousands of duplicates, the rounded-up burn becomes meaningful.

**When to disable.** Workflows that intentionally re-embed (rare). Otherwise leave it on — it is the highest-precision rule in the catalog.

---

## 4. `zombie`

**What it detects.** An agent run is still firing API calls but has produced no user-facing output for an unusually long time — it is stuck.

**When it fires.** All of:

- The session has at least `min_recent_calls` calls.
- The most recent user-facing output (a non-tool-call response with text) was more than `threshold_minutes` minutes ago.
- At least `min_recent_calls` calls happened in the last `threshold_minutes` window.

A "user-facing output" is a `CallRecord` with `user_facing_output=True`, which the wrappers set when the response contains text and no tool calls.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `zombie.threshold_minutes` | `5` | How long without user-facing output before firing. |
| `zombie.min_recent_calls` | `5` | Minimum calls in the threshold window. |

**How to tune.**

```python
Sentinel(
    project="my-agent",
    config={
        "zombie.threshold_minutes": 10,    # wait longer before flagging
        "zombie.min_recent_calls": 10,     # need more calls to count as "still firing"
    },
)
```

For long-running legitimate background agents (overnight research), raise `threshold_minutes` substantially or disable the rule for those projects.

**Sample event.**

```python
LeakEvent(
    type="zombie",
    confidence=0.75,
    project="coding-agent",
    session_id="vscode-task-31",
    rule="v0.zombie",
    evidence={
        "minutes_since_user_facing_output": 12.4,
        "recent_calls": 38,
    },
    estimated_burn=0.1900,
    suggested_action="kill_session_or_request_user_input",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Background agents that legitimately run for hours without user-visible output. Agents whose entire output is tool calls (e.g., a structured-only agent that never returns text). For the latter, the cleanest fix is to set `user_facing_output=True` on whatever you consider a "completion" call before passing it to `Sentinel.record_call` — but most users will just disable the rule.

---

## 5. `model_misroute`

**What it detects.** A small, classification-shaped prompt sent to a frontier model when a small model would produce equivalent results far cheaper.

**When it fires.** All of:

- Prompt token count below `max_prompt_tokens` (default 500).
- Completion token count below `max_completion_tokens` (default 50).
- The flattened messages text contains a classification keyword: `classify`, `yes or no`, `true or false`, `rate from 1`, `rate this on a scale`, `is this`, `categorize`, `label this`, `which category`.
- Model is a frontier model (Claude Opus, Claude Sonnet, GPT-5, GPT-4-turbo, GPT-4o, Gemini 2.5/2.0 Pro, DeepSeek-Chat/Reasoner, Command-R+/A, Mistral Large) — and is *not* an explicit cheap variant (`gpt-5-mini`, `gpt-5-nano`, `gpt-4o-mini`).

The emitted event includes a `recommended_alternative` field that names the cheaper model the rule would route this call to.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `model_misroute.max_prompt_tokens` | `500` | Below this, the prompt is "small enough to be a classify". |
| `model_misroute.max_completion_tokens` | `50` | Below this, the response is "small enough to be a classify". |

**How to tune.**

```python
Sentinel(
    project="my-agent",
    config={
        "model_misroute.max_prompt_tokens": 1000,   # widen — accept more shapes as "classification-shaped"
        "model_misroute.max_completion_tokens": 100,
    },
)
```

The keyword list is currently not configurable via `config`. If you need different keywords (e.g., domain-specific classification phrases), the cleanest path is to disable this rule and run your own pre-call classifier check.

**Sample event.**

```python
LeakEvent(
    type="model_misroute",
    confidence=0.70,
    project="my-agent",
    session_id="user-task-44",
    rule="v0.model_misroute",
    evidence={
        "model": "claude-sonnet-4-6",
        "prompt_tokens": 24,
        "completion_tokens": 6,
        "matched_keywords": ["classify"],
        "recommended_alternative": "claude-haiku-4-5",
    },
    estimated_burn=0.0015,
    suggested_action="route_to_claude-haiku-4-5",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Domains where small-prompt classification genuinely needs frontier-model judgment (medical / legal nuance). Per-route nuance — you want frontier-model classification on user-uploaded content but not on internal pipeline prompts. For both cases the right tool is per-project disable rather than try to thread the rule.

---

## 6. `retry_storm`

**What it detects.** The same call repeated many times in a short window without any parameter change — usually an upstream error not being handled.

**When it fires.** ≥`min_retries` calls within `window_seconds` whose `request_hash` is identical. The hash covers `(model, messages, tools, max_tokens)` (Anthropic / OpenAI / Gemini) or `(modelId, messages, toolConfig, inferenceConfig)` (Bedrock) — anything that would normally vary call-to-call. So a duplicate hash means *literally the same call*.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `retry_storm.window_seconds` | `30` | Look-back window for counting hash repeats. |
| `retry_storm.min_retries` | `5` | Number of identical calls required to fire. |

**How to tune.**

```python
Sentinel(
    project="my-agent",
    config={
        "retry_storm.min_retries": 10,    # quieter — only fire on egregious storms
        "retry_storm.window_seconds": 60, # widen window
    },
)
```

If you have a legitimate retry layer with exponential backoff doing 5+ retries in 30s, raise `min_retries`. The default catches the failure mode where someone's HTTP layer is doing tight retries with no jitter.

**Sample event.**

```python
LeakEvent(
    type="retry_storm",
    confidence=0.90,
    project="my-agent",
    session_id="webhook-handler-12",
    rule="v0.retry_storm",
    evidence={
        "request_hash": "a8b3f1c2d4e5f607",
        "retry_count": 6,
        "window_seconds": 30,
        "model": "claude-sonnet-4-6",
    },
    estimated_burn=0.0540,
    suggested_action="add_backoff_or_check_upstream_health",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Rare — exact-hash repeat is almost never legitimate. If you have a pipeline that intentionally repeats identical calls (load testing, idempotency probes), disable for that project.

---

## 7. `tool_definition_bloat`

**What it detects.** A single request is shipping an oversized block of tool definitions, burning context on every turn before the model has done any work. The canonical example is an MCP host with many servers attached: 12 servers × ~5 tools each = 58 tools, ~55K tokens of definitions injected on every user turn.

**When it fires.** On any single call, the request's `tools` array exceeds `tool_count_threshold` tools *or* `tool_definition_bytes_threshold` bytes when serialized to JSON. Bedrock's `toolConfig.tools` is also recognized.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `tool_definition_bloat.tool_count_threshold` | `30` | Number of tools above which the rule fires. |
| `tool_definition_bloat.tool_definition_bytes_threshold` | `30000` | Byte size of serialized tool defs above which the rule fires. |

Confidence ladder:
- `0.85` if either threshold is tripped.
- `0.95` if **both** thresholds are tripped by ≥50% (e.g., 45 tools at 45KB).

**How to tune.**

```python
Sentinel(
    project="mcp-host",
    config={
        "tool_definition_bloat.tool_count_threshold": 50,         # higher floor
        "tool_definition_bloat.tool_definition_bytes_threshold": 60000,
    },
)
```

If you are a legitimate large-tool-array workload, raise both thresholds rather than disabling — the rule's reported `top_tools_by_size` evidence is genuinely useful even when you don't act on it.

**Sample event.**

```python
LeakEvent(
    type="tool_definition_bloat",
    confidence=0.95,
    project="mcp-host",
    session_id="user-session-77",
    rule="v0.tool_definition_bloat",
    evidence={
        "tool_count": 58,
        "definition_bytes": 56240,
        "estimated_tokens": 14060,
        "top_tools_by_size": [
            {"name": "filesystem_read", "bytes": 4200},
            {"name": "browser_action", "bytes": 3940},
            {"name": "jira_create_issue", "bytes": 3680},
            {"name": "supabase_query", "bytes": 3420},
            {"name": "linear_create_issue", "bytes": 3110},
        ],
        "count_threshold": 30,
        "bytes_threshold": 30000,
        "recent_call_count": 12,
    },
    estimated_burn=1.5181,
    suggested_action="reduce_tool_count_or_use_lazy_tool_loading",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** A genuinely tool-heavy agent where the count is intentional (rare). Otherwise leave it on — even the false positives produce useful diagnostic output (`top_tools_by_size`).

---

## 8. `retrieval_thrash`

**What it detects.** A retrieval-shaped tool is called repeatedly with overlapping queries — the agent is asking essentially the same retrieval question multiple ways instead of caching, widening, or deduping.

**When it fires.** Same shape as `tool_loop`, but scoped to retrieval-shaped tool names and with looser thresholds. Tool names are matched against `retrieval_tool_patterns`: substring match for entries without `*`, glob match for entries containing `*`.

**Default thresholds.**

| Config key | Default | Description |
|---|---|---|
| `retrieval_thrash.window_seconds` | `120` | Look-back window — wider than `tool_loop` because retrieval pipelines are slower per turn. |
| `retrieval_thrash.min_calls` | `3` | Minimum calls before evaluating. |
| `retrieval_thrash.cosine_threshold` | `0.65` | Mean similarity — looser than `tool_loop`'s 0.70 because retrieval queries naturally overlap. |
| `retrieval_thrash.similarity_metric` | `"tfidf_charngram"` | Either `"tfidf_charngram"` or `"jaccard"`. |
| `retrieval_thrash.charngram_size` | `4` | Char n-gram size, clamped to 3–5. |
| `retrieval_thrash.retrieval_tool_patterns` | see below | Tool-name patterns considered "retrieval-shaped". |

Default `retrieval_tool_patterns`:
```python
("search", "retrieve", "query", "lookup", "find_documents",
 "vector_search", "similarity_search",
 "rag_*", "*_search", "*_query", "*_retrieve")
```

**How to tune.**

```python
Sentinel(
    project="rag-pipeline",
    config={
        "retrieval_thrash.cosine_threshold": 0.75,   # stricter
        "retrieval_thrash.retrieval_tool_patterns": ("vector_search", "kb_lookup"),  # only your retrievers
    },
)
```

If `retrieval_thrash` is firing on a tool named `query_status` (which is not actually retrieval), the cleanest fix is a custom `retrieval_tool_patterns` list that only includes your real retrievers.

**Sample event.**

```python
LeakEvent(
    type="retrieval_thrash",
    confidence=0.79,
    project="rag-pipeline",
    session_id="user-search-22",
    rule="v0.retrieval_thrash",
    evidence={
        "tool": "vector_search",
        "call_count": 4,
        "window_seconds": 120,
        "mean_similarity": 0.71,
        "sample_args": [
            {"query": "kubernetes pod stuck pending"},
            {"query": "k8s pod pending state troubleshoot"},
            {"query": "kubernetes scheduler pod pending"},
        ],
        "matched_pattern": "vector_search",
    },
    estimated_burn=0.0420,
    suggested_action="cache_retrieval_results_or_widen_initial_query_or_dedupe",
    raised_at=datetime(2026, 5, 7, 14, 22, 31, tzinfo=timezone.utc),
)
```

**When to disable.** Workloads where overlap between retrieval queries is intended (multi-tenant retrieval where the same tool resolves different tenants per call). Use a custom `retrieval_tool_patterns` to scope, or disable for that project.

---

## Tuning workflow

1. **Start with all defaults.** Run for at least a week in `log` mode with all rules enabled.
2. **Identify the noisy ones.** A rule firing more than ~5 times per 1000 calls warrants investigation.
3. **Decide: tune or disable?**
   - If most firings *are* leaks but a few aren't: tune thresholds (`cosine_threshold`, `min_calls`, `slope_threshold`, etc.).
   - If most firings *aren't* leaks: disable the rule entirely for that project.
4. **Promote one rule at a time to `block` mode.** Start with the highest-precision rules (`embedding_waste`, `retry_storm`).
5. **Iterate.** Re-tune as your traffic shape changes.

For per-rule pricing of the wrong-action cost:

| Rule | Cost of false positive | Cost of false negative |
|---|---|---|
| `tool_loop` | medium — agent paused unnecessarily | high — runaway loop spending |
| `context_bloat` | low — alert noise | high — quadratic token growth |
| `embedding_waste` | near-zero — exact match | medium — duplicate embed costs |
| `zombie` | medium — kills a slow but legit run | high — multi-hour stuck agent |
| `model_misroute` | low — over-cautious routing | medium — frontier-model bill |
| `retry_storm` | near-zero — exact-hash match | high — 100+ retries in seconds |
| `tool_definition_bloat` | low — tool list audit suggested | high — 70%+ context burned |
| `retrieval_thrash` | medium — caching suggested | medium — redundant retrieval |

This asymmetry is why the defaults err toward fewer false positives at the cost of some false negatives. Detection products that cry wolf get muted.

## What's next (V1 roadmap)

V1 will add an **LLM-as-judge** pass: a cheap model (Haiku) reads the gray-zone V0 firings (confidence 0.5–0.75) and ratifies or vetoes them. Haiku polices Opus. This dramatically reduces false positives on heuristic rules (`tool_loop`, `model_misroute`, `context_bloat`) without raising thresholds.

V1 will also add:

- Semantic similarity for `tool_loop` (sentence-transformers, optional via `[embeddings]` extra).
- Per-rule mode (e.g., `block` only on `embedding_waste`).
- Polling-tool allow-lists for `tool_loop`.
- Context-token-entropy refinement for `context_bloat`.

Until V1 lands, tune thresholds and use rule disable lists to manage noise.
