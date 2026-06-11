# Troubleshooting

Practical debugging guide for the issues you'll actually hit. Organized by symptom.

If you find a bug not in this list, open a GitHub issue with a minimal repro.

## A rule didn't fire when I expected it to

Walk through this checklist in order. Each step is a thing that genuinely catches it for some users.

### 1. Is the same `session_id` being used across calls?

Most rules need ≥3 calls in the same session before they evaluate (`tool_loop`, `retry_storm`, `context_bloat`, `retrieval_thrash`, `zombie`). By default each call gets its own UUID `session_id`, so a fresh session means rules never have enough data.

Pass a stable `_sentinel_session_id`:

```python
session = "user-42-task-17"
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
```

To verify the session id is being honored, inspect the tracer:

```python
session_records = sentinel.tracer.session("user-42-task-17")
print(f"calls in session: {len(session_records)}")
```

If `len(session_records) == 1` after three calls, your `_sentinel_session_id` is not being threaded through correctly.

### 2. Is the rule actually enabled?

If you set `rules=` to a subset, anything not in the list is disabled:

```python
# Only retry_storm is enabled — tool_loop will never fire.
sentinel = Sentinel(project="my-agent", rules=["retry_storm"])
```

Check `[r.name for r in sentinel._rules]`:

```python
print([r.name for r in sentinel._rules])
# ['tool_loop', 'context_bloat', 'embedding_waste', 'zombie', 'model_misroute',
#  'retry_storm', 'tool_definition_bloat', 'retrieval_thrash']
```

### 3. Is `min_confidence` filtering it out?

If you raised `min_confidence` above the rule's typical confidence, events are dropped silently before reaching handlers. For example, `model_misroute` emits at `0.7` — if you set `min_confidence=0.8`, you'll never see one.

Match your floor to the rules you care about:

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

### 4. Are the rule's threshold conditions actually met?

The most common reason a rule doesn't fire is that the data doesn't satisfy the rule's preconditions. For example:

- `tool_loop` needs ≥3 calls *with the same tool name* in 60s with mean similarity ≥0.70. If your tool calls have different names, or the args are too dissimilar, it won't fire.
- `context_bloat` needs ≥5 calls in the session and a slope ≥1500 tokens/turn over the last 10 turns.
- `retry_storm` needs ≥5 calls with the *same `request_hash`* in 30s. Hash covers `(model, messages, tools, max_tokens)` — any change makes them count as different calls.
- `embedding_waste` needs ≥2 calls with `method=="embeddings.create"` and identical `raw_request['input']`.

Inspect the captured `CallRecord`s directly:

```python
records = sentinel.tracer.session("user-42-task-17")
for r in records:
    print(r.method, r.tool_calls, r.request_hash[:16])
```

If the records look right but the rule isn't firing, lower the threshold temporarily to confirm the rule path works:

```python
sentinel = Sentinel(
    project="my-agent",
    config={"tool_loop.cosine_threshold": 0.0, "tool_loop.min_calls": 2},
)
# If this fires, the path works — adjust thresholds back up.
```

### 5. Is your provider's call shape what the rule expects?

Some rules read fields the wrappers populate. If you're using a custom integration that builds `CallRecord` directly via `Sentinel.record_call`, make sure those fields are populated:

- `embedding_waste` reads `raw_request["input"]`.
- `model_misroute` reads `raw_request["messages"]` (and supports list-of-string and list-of-dict-with-content).
- `tool_definition_bloat` reads `raw_request["tools"]` or `raw_request["toolConfig"]["tools"]`.

If you're using a native wrapper (Anthropic / OpenAI / Gemini / Bedrock), these are populated correctly. If you've customized something, verify by printing the record:

```python
print(records[-1].raw_request)
```

## A rule fired but it shouldn't have (false positive)

The defaults are calibrated for typical traffic. Every project has edge cases.

### 1. Tune the threshold

Most false positives can be killed by tightening the rule's threshold. Start by raising the threshold by ~10% and observe:

```python
Sentinel(
    project="my-agent",
    config={
        "tool_loop.cosine_threshold": 0.80,    # was 0.70
        "tool_loop.min_calls": 5,              # was 3
        "context_bloat.slope_threshold": 3000, # was 1500
        "retry_storm.min_retries": 10,         # was 5
    },
)
```

For `tool_definition_bloat`, raise both thresholds:

```python
config={
    "tool_definition_bloat.tool_count_threshold": 50,
    "tool_definition_bloat.tool_definition_bytes_threshold": 60000,
}
```

### 2. Disable the rule for that project

If a rule is fundamentally wrong for your traffic shape (e.g., `zombie` on a deliberately long-running background agent), disable it:

```python
Sentinel(
    project="my-agent",
    rules=[
        "tool_loop",
        "context_bloat",
        "embedding_waste",
        "model_misroute",
        "retry_storm",
        "tool_definition_bloat",
        "retrieval_thrash",
        # zombie omitted
    ],
)
```

### 3. Increase `min_confidence`

If only the lowest-confidence firings are wrong, raise the project floor:

```python
Sentinel(project="my-agent", min_confidence=0.7)
```

### 4. Use rule-specific configuration

Some rules support custom inputs:

- `retrieval_thrash.retrieval_tool_patterns` — change which tool names count as retrieval.
- `tool_loop.similarity_metric` — switch from `tfidf_charngram` to `jaccard` if your args have unusual character distributions.

See [Leak rules](./04-waste-rules.md) for the full per-rule config.

## Streaming isn't capturing tokens

### OpenAI streaming is currently passthrough

The OpenAI wrapper detects `stream=True` and passes the call through without instrumentation. Anthropic, Gemini, and Bedrock streaming are all fully instrumented today; OpenAI stream instrumentation is tracked for an upcoming release.

```python
# This call won't produce a CallRecord:
stream = client.chat.completions.create(model="gpt-5", messages=[...], stream=True)
```

Workarounds:

1. **Use non-streaming where possible** — `stream=False` (or omit `stream=`) and the wrapper instruments fully.
2. **Track the upcoming stream-instrumentation release** in the changelog and upgrade when it ships.

If you are on `mode='block'` and use OpenAI streaming, the wrapper emits a `RuntimeWarning` once per Sentinel so you know detection is silently bypassed for the streaming path. Suppress with `warnings.filterwarnings(...)` if intentional.

### Anthropic / Gemini / Bedrock streaming works

If you're using Anthropic `messages.stream`, Gemini `generate_content_stream`, or Bedrock `converse_stream` and the records aren't appearing:

- For Anthropic: the record is built when the stream context manager *exits*. If you break out of iteration without exiting the context manager, the record won't fire. Always use `with client.messages.stream(...) as stream:`.
- For Gemini: the record is built when the iterator is exhausted or garbage-collected. If you break out of iteration early, the record fires on GC — which may be after your test exits. Iterate to completion to be sure.
- For Bedrock: the record fires on stream exhaustion, `close()`, or `__del__`. Same caveat as Gemini.

### Token counts are zero on streaming

Token counts come from the stream's terminal events:

- Anthropic: `message_start` / `message_delta.usage` / `message_stop`.
- Gemini: each chunk's `usage_metadata` (cumulative).
- Bedrock: the `metadata` event near stream end.

If you see `prompt_tokens=0, completion_tokens=0` on a streamed call, either the SDK didn't emit the terminal usage event (rare, usually a network truncation) or you broke out of iteration before reaching the end. Iterate to completion.

## Async wrapper isn't working

The wrappers detect async vs sync via `inspect.iscoroutinefunction(client.<method>)`. If you have a custom async client subclass that doesn't expose its methods as coroutine functions (e.g., they're wrapped in a non-coroutine descriptor), the dispatcher will pick the sync path and fail.

Check:

```python
import inspect
print(inspect.iscoroutinefunction(client.messages.create))  # should be True for AsyncAnthropic
```

If it's False on what should be an async client:

1. You're not using `AsyncAnthropic` / `AsyncOpenAI` — you're using the sync class.
2. You're using a custom subclass that breaks the coroutine-function pattern. Either fix the subclass or wrap the underlying SDK client directly.

For Gemini async, the wrapper instruments `client.aio.models.generate_content` and `client.aio.models.generate_content_stream`. The sync surface (`client.models.*`) is also instrumented separately — you can use both.

## `LeakDetected` is being swallowed

The two-level safety boundary in the wrapper:

- Record-building errors are swallowed (instrumentation must never break user code).
- Rule and handler exceptions are swallowed (a buggy rule can't kill the agent).
- `LeakDetected` is the *one* exception that always propagates — it's the entire point of `block` mode.

There is one specific edge case where `block` mode is best-effort rather than guaranteed: **abandoned streams**. For Bedrock and Gemini streams that get cleaned up by Python's garbage collector (because the user broke iteration without `with` or `close()`), Python silently swallows exceptions raised from `__del__` / generator-close paths. The leak is still detected and your handlers are still called, but `LeakDetected` is suppressed with a `RuntimeWarning`. Use `with stream:` to get guaranteed propagation.

If you're seeing `LeakDetected` swallowed on a non-streaming call, that's a regression — open an issue with a minimal repro.

To verify the fix is in your install:

```python
from token_sentinel import Sentinel, LeakDetected, CallRecord
from datetime import datetime, timezone
import hashlib

sentinel = Sentinel(project="check", mode="block", config={"retry_storm.min_retries": 2})
same_hash = hashlib.sha256(b"same").hexdigest()

for i in range(2):
    try:
        sentinel.record_call(CallRecord(
            session_id="check",
            timestamp=datetime.now(timezone.utc),
            provider="test",
            model="test",
            method="test",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=10,
            request_hash=same_hash,
        ))
    except LeakDetected as exc:
        print(f"BLOCK fired correctly: {exc.event.type}")
        break
```

You should see `BLOCK fired correctly: retry_storm` on the second call.

## Memory is growing

The tracer keeps a per-session ring buffer of `CallRecord`s, capped at 200 records per session by default. If you have:

- A long-lived process with many distinct `session_id` values, each gets its own ring buffer.
- Sessions that never get explicitly cleared.

Memory will grow linearly with the number of distinct sessions, capped per-session.

### Options

**1. Clear sessions you're done with.** This is the simplest fix:

```python
sentinel.tracer.clear("user-42-task-17")  # clear one session
sentinel.tracer.clear()                   # clear ALL sessions
```

Call this in your agent's "task complete" hook.

**2. Reduce the per-session cap.** The default is 200 records per session. Lower it if you don't need that much history for rule evaluation:

```python
from token_sentinel.tracer import Tracer

sentinel = Sentinel(project="my-agent")
sentinel.tracer = Tracer(max_records_per_session=50)
```

(Replacing `sentinel.tracer` after construction is internal — the cleaner public version of this API is on the V1 roadmap.)

**3. Use ephemeral session ids and accept GC.** If you don't pass `_sentinel_session_id`, each call gets a fresh UUID. The tracer holds these forever (no time-based eviction in V0). This is the worst pattern for memory — always pass a stable session id and clear it when done.

## Latency overhead is too high (>50ms)

The V0 budget is ≤50ms p95 across all rules. If you're seeing more:

### 1. Profile rules

The most expensive rule is `tool_loop` because it computes pairwise TF-IDF cosine similarity. It's still sub-millisecond on small corpora, but if you have an agent that's making 100+ same-tool calls in a 60-second window, the n² pair count adds up.

Check the session size:

```python
records = sentinel.tracer.session("...")
print(f"records in session: {len(records)}")
print(f"tool_calls in latest record: {len(records[-1].tool_calls)}")
```

If the session has 200+ records (the per-session cap) and many of them have many tool calls, `tool_loop` and `retrieval_thrash` will both pay similarity-computation costs each call.

### 2. Reduce session retention

Lower `max_records_per_session` or clear sessions sooner — see "Memory is growing" above.

### 3. Disable the most expensive rules

If you genuinely don't need `tool_loop` or `retrieval_thrash`, disable them:

```python
Sentinel(
    project="my-agent",
    rules=[
        "context_bloat",
        "embedding_waste",
        "zombie",
        "model_misroute",
        "retry_storm",
        "tool_definition_bloat",
        # tool_loop and retrieval_thrash omitted
    ],
)
```

### 4. Profile a single call

```python
import time
start = time.perf_counter()
client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[...])
print(f"{(time.perf_counter() - start) * 1000:.1f}ms")
```

Compare against the same call without TokenSentinel (use the raw `anthropic.Anthropic()` instead of `sentinel.wrap(...)`). The delta is the instrumentation overhead. If it's > 50ms p95, file an issue with the session size and rule configuration.

## "Unsupported client type" on `wrap()`

```
TypeError: Unsupported client type: <ClassName> from <module>. Sentinel supports
Anthropic, OpenAI (+ compatible), Google Gemini, Google Vertex AI, and AWS
Bedrock clients.
```

The dispatcher checks `type(client).__module__`. Most often:

1. **You're passing the module, not an instance.** `sentinel.wrap(anthropic)` is wrong; `sentinel.wrap(anthropic.Anthropic())` is right.
2. **Your SDK is too old.** Upgrade to the version listed in [Installation](./01-installation.md).
3. **You're passing a custom subclass** in your own module. The dispatcher matches the official module prefixes — your subclass's `__module__` won't match. Either upstream the subclass to the official SDK or wrap the official client directly.
4. **For Bedrock specifically**: the dispatcher checks for `"bedrock" in cls_name.lower()` *or* the boto3 client's `service_model.service_name`. If you're using a non-standard botocore version that doesn't expose `service_model`, the check falls back to the class name — which should still work for `bedrock-runtime` clients.

## Other questions

If your issue isn't in this list:

1. Check [API reference](./07-api-reference.md) for the exact behavior contract.
2. Check [Leak rules](./04-waste-rules.md) for the rule's preconditions.
3. Check [Modes](./03-modes.md) if it's about `log` / `alert` / `block` behavior.
4. Open a GitHub issue with a minimal repro. Include your TokenSentinel version (`python -c "import token_sentinel; print(token_sentinel.__version__)"`), your provider SDK version, and the exact code that produced the unexpected behavior.
