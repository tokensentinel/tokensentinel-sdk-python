# Troubleshooting

> **Documented for Python SDK `token-sentinel` 1.0.3.**


Practical debugging by symptom. If something is missing, open a GitHub issue with a minimal repro and `token_sentinel.__version__`.

## A rule didn't fire when I expected it to

### 1. Same `session_id` across multi-turn calls?

Most multi-turn rules need several calls in **one** session. By default each call gets a fresh UUID.

```python
session = "user-42-task-17"
client.messages.create(..., _sentinel_session_id=session)
client.messages.create(..., _sentinel_session_id=session)
```

```python
print(len(sentinel.tracer.session("user-42-task-17")))
```

If this stays `1` after three calls, the session id is not being threaded through.

### 2. Is the rule enabled?

```python
print([r.name for r in sentinel._rules])
# Expected for rules="all" (15 names):
# tool_loop, context_bloat, embedding_waste, zombie, model_misroute,
# retry_storm, tool_definition_bloat, retrieval_thrash,
# vision_re_upload, vision_high_detail_misroute, vision_cost_concentration,
# audio_multichannel_doubling, voice_switching_loop, rerank_thrash, repair_loop
```

Unknown names in `rules=[...]` produce a `UserWarning` and are dropped.

### 3. Is `min_confidence` filtering it out?

| Rule | Typical confidence |
|---|---|
| `tool_loop` | 0.6–0.99 |
| `context_bloat` | 0.55–0.95 |
| `embedding_waste` | 0.99 |
| `zombie` | 0.75 |
| `model_misroute` | 0.70 |
| `retry_storm` | 0.90 |
| `tool_definition_bloat` | 0.85–0.95 |
| `retrieval_thrash` | 0.55–0.95 |
| `vision_re_upload` | 0.65–0.99 |
| `vision_high_detail_misroute` | 0.75 |
| `vision_cost_concentration` | 0.65 |
| `audio_multichannel_doubling` | 0.75–0.85 |
| `voice_switching_loop` | 0.7–0.9 |
| `rerank_thrash` | 0.75–0.9 |
| `repair_loop` | 0.65–0.9 |

### 4. Preconditions met?

Examples:

- `tool_loop` — ≥3 same tool name in 60s, mean arg similarity ≥ 0.70 (TF-IDF)
- `context_bloat` — ≥5 turns, slope ≥ 1500 tokens/turn over lookback
- `retry_storm` — ≥5 identical `request_hash` in 30s
- `embedding_waste` — ≥2 embeds with same `raw_request["input"]` and method ending in `embeddings.create` **or** exactly `embed`
- `zombie` — needs a **prior** `user_facing_output=True` anchor; pure tool-only sessions never fire today
- `vision_*` / `audio_*` / `voice_*` / `rerank_*` — provider-specific; wrong provider → never fire
- `repair_loop` — needs full `messages` in `raw_request` (Cohere chat redacts them)

### 5. Provider-shaped fields present?

If you hand-build `CallRecord`s:

- `embedding_waste` → `raw_request["input"]`
- `model_misroute` / `repair_loop` → `raw_request["messages"]`
- `tool_definition_bloat` → `raw_request["tools"]` or `toolConfig.tools`
- Vision rules → base64 image blocks in messages/contents
- Audio multichannel → Deepgram `usage_extra.model_specific_meta`

## False positives

Tighten thresholds, disable the rule for that project, or raise `min_confidence`. See [Leak rules](./04-waste-rules.md).

Retrieval tools may fire **both** `tool_loop` and `retrieval_thrash` — disable one if that is noisy.

## Streaming / tokens

### OpenAI streaming **is** instrumented

`stream=True` is wrapped by a stream proxy that records a `CallRecord` when the stream finishes.

If tokens are zero:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},
)
```

Without `include_usage`, expect `raw_response_meta["usage_unavailable"] = True`.

A one-time `RuntimeWarning` about streaming only appears if the proxy **cannot** be constructed (defensive fallback). Normal streaming should not warn.

### Anthropic / Gemini / Bedrock streaming

Records finalize on context-manager exit or stream exhaustion. Prefer `with stream:` so `mode="block"` can raise cleanly. Abandoned iterators cleaned up by GC may suppress `LeakDetected` (handlers still ran).

## Async

Wrappers detect async via `inspect.iscoroutinefunction`. Use `AsyncAnthropic` / `AsyncOpenAI` / Gemini `client.aio.models.*` as appropriate.

## `LeakDetected` swallowed

- Handler exceptions never become block — only Sentinel's raise does.
- Abandoned streams (GC path) may suppress the raise after handlers ran.
- Policy exceptions (`BudgetExceeded`, etc.) only fire when cloud policy is configured.

Verify block with synthetic `record_call` + `retry_storm` as in older guides, or:

```python
from token_sentinel import Sentinel, LeakDetected, CallRecord
from datetime import datetime, timezone
import hashlib

sentinel = Sentinel(project="check", mode="block", config={"retry_storm.min_retries": 2})
h = hashlib.sha256(b"same").hexdigest()
for _ in range(2):
    try:
        sentinel.record_call(CallRecord(
            session_id="check",
            timestamp=datetime.now(timezone.utc),
            provider="test", model="test", method="test",
            prompt_tokens=10, completion_tokens=5, latency_ms=1,
            request_hash=h,
        ))
    except LeakDetected as exc:
        print("ok", exc.event.type)
        break
```

## Memory growth

Defaults: **200** records per session, **1000** sessions (LRU eviction).

```python
Sentinel(project="my-agent", max_records_per_session=50, max_sessions=200)
```

Clear finished work:

```python
sentinel.tracer.clear("user-42-task-17")
sentinel.tracer.clear()  # all
```

Prefer stable `_sentinel_session_id` values you can clear, not unlimited unique UUIDs without `max_sessions`.

## High overhead

Budget target: ≤50ms p95 across rules. `tool_loop` / `retrieval_thrash` cost grows with pairwise comparisons. Lower session depth, clear sessions, or disable those rules.

## `Unsupported client type`

1. Pass an **instance**, not a module.
2. Install the matching extra and upgrade the provider SDK ([Installation](./01-installation.md)).
3. Official module prefixes only — custom subclasses in your package may not match; wrap the official client.
4. Cohere: use `ClientV2` / `AsyncClientV2`, not legacy `Client`.
5. Bedrock: use `bedrock-runtime` Converse APIs.

## Cloud / policy surprises

| Symptom | Likely cause |
|---|---|
| Events leave the process in `mode="log"` | `cloud_endpoint` + `api_key` set — shipping is not alert-only |
| `BudgetExceeded` after the model already ran | Policy runs in `record_call` after the provider call; cannot un-bill current call |
| No policy exceptions ever | Policy poller needs `api_key` + `policy_endpoint` or `cloud_endpoint` |

Cloud features are **paid closed-source** products. OSS SDK works fully without them.

## Other

1. [API reference](./07-api-reference.md)
2. [Leak rules](./04-waste-rules.md)
3. [Modes](./03-modes.md)
4. GitHub Issues with version + provider SDK versions + minimal repro
