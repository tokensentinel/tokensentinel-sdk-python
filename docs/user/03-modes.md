# Modes

> **Documented for Python SDK `token-sentinel` 1.0.3.**


TokenSentinel has three modes that control what happens when a rule fires: `log`, `alert`, and `block`. Pick the one that matches how much you trust the rules to be right.

```python
sentinel = Sentinel(project="my-agent", mode="log")    # default
sentinel = Sentinel(project="my-agent", mode="alert")
sentinel = Sentinel(project="my-agent", mode="block")
```

| Mode | What it does | When to use |
|---|---|---|
| `log` | Calls your registered `on_leak` / `on_waste` handlers. Never raises for rules. | Day-one production. Always. |
| `alert` | Same handler behavior as `log`. Historically named for “cloud-oriented” deployments; **does not by itself enable cloud**. | Same as `log` unless your team uses the name as a convention. |
| `block` | Same as `log` plus raises `LeakDetected` / `WasteDetected` after handlers. | Once a rule has been trustworthy in `log` and you have a recovery path. |

Handlers always run when a rule fires (subject to `min_confidence` and dedup). Cloud delivery is a **separate axis** — see below.

### Cloud delivery is orthogonal to mode

If you set **both** `cloud_endpoint` and `api_key`, the SDK ships events to TokenSentinel Cloud in **`log`, `alert`, and `block`**. Mode is stamped on each event for cloud analytics (e.g. savings weighting). If cloud is not configured, `alert` behaves like `log` locally.

Cloud is **optional, closed-source, and paid** (Team / Pro / Enterprise). The Apache-2.0 SDK never phones home without those two kwargs.

## `log` mode (default — safe for prod from day 1)

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")

@sentinel.on_leak
def handle(event):
    # ship to your logger / queue / metrics system
    print(f"LEAK [{event.type}] {event.confidence:.2f} burn=${event.estimated_burn:.4f}")

client = sentinel.wrap(anthropic.Anthropic())
client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[...])
```

`log` is the default for a reason: the worst case is your handler runs and emits noise. The wrapped call always returns successfully. There is no path where TokenSentinel breaks your agent in `log` mode — handler exceptions are caught and swallowed, the underlying call's response is always returned.

This is what you should ship first. Always. Even if you ultimately want `block`.

## `alert` mode

```python
sentinel = Sentinel(
    project="my-agent",
    mode="alert",
    # Optional paid cloud — works in log/alert/block when both are set:
    cloud_endpoint="https://api.tokensentinel.dev",
    api_key="ts_live_...",
)

@sentinel.on_leak
def handle(event):
    print(f"LEAK [{event.type}] {event.confidence:.2f}")
```

Locally, `alert` is the same as `log`: handlers run, no raise. Use it if your org’s runbooks already say “alert mode” for cloud-connected agents.

With cloud configured (any mode), the SDK fire-and-forgets batched POSTs to `{cloud_endpoint}/v1/events`. Network failures never block the agent. The hosted dashboard (webhooks, retention, Intervention Pack, Pro judge/composites) is closed-source — see [FAQ](./09-faq.md#open-source-sdk-vs-paid-cloud) and [tokensentinel.dev](https://tokensentinel.dev).

## `block` mode (hard-stop)

```python
from token_sentinel import Sentinel, LeakDetected
import anthropic

sentinel = Sentinel(project="my-agent", mode="block")
client = sentinel.wrap(anthropic.Anthropic())

try:
    response = client.messages.create(model="claude-sonnet-4-6", messages=[...])
except LeakDetected as exc:
    # exc.event is the LeakEvent that caused the block
    print(f"Blocked: {exc.event.type} confidence={exc.event.confidence:.2f}")
    # Take whatever recovery action makes sense — kill the session, alert ops,
    # surface to the user, etc.
```

`block` raises `LeakDetected` *after* the underlying LLM call has already returned. The exception propagates up to your code, which gets to decide what to do.

The exception carries the offending `LeakEvent` on `exc.event`, so you can inspect `type`, `confidence`, `evidence`, etc. before deciding to retry, fall back, or fail the request entirely.

### Important caveats for `block`

- **The LLM call already happened.** `block` cannot prevent the *current* call from being billed — rules run after the provider returns. The savings come from stopping the *next* turns in a wasteful loop (and from your handler / exception recovery).
- **Handler exceptions are still caught.** If your handler raises, that does not block the agent — `LeakDetected` comes from `Sentinel`, not from your handler.
- **Streaming finalizes at stream close.** Anthropic / OpenAI / Gemini / Bedrock streaming build the `CallRecord` when the stream finishes. Prefer `with stream:` so raises propagate; GC cleanup of abandoned streams may suppress the exception after handlers ran.

## Policy plane (paid cloud, mode-independent)

When cloud policy is configured (`cloud_endpoint`/`policy_endpoint` + `api_key`), `record_call` may raise `BudgetExceeded`, `VelocityExceeded`, or `KillSwitchActive` **regardless of mode**. Those exceptions subclass `LeakDetected`.

They still run **after** the wrapper’s provider call (same boundary as rules). Configure policy only if your paid cloud tier enables the Intervention Pack and you accept post-call halt semantics.

## Per-rule confidence floors

Beyond modes, you can filter individual rules by confidence with the project-level `min_confidence`:

```python
sentinel = Sentinel(project="my-agent", mode="log", min_confidence=0.7)
```

Any event below `min_confidence` is silently dropped before reaching handlers. Default is `0.5`. Use this to silence noisy rules without removing them entirely.

For per-rule tuning (different thresholds per rule), use the `config` dict:

```python
sentinel = Sentinel(
    project="my-agent",
    mode="log",
    config={
        # Tune individual rules
        "tool_loop.cosine_threshold": 0.80,    # default 0.70 — stricter
        "tool_loop.min_calls": 5,              # default 3 — fewer fires
        "retry_storm.min_retries": 10,         # default 5 — quieter
        "context_bloat.slope_threshold": 3000, # default 1500 — only fire on big jumps
    },
)
```

Each rule's config keys are documented in [Leak rules](./04-waste-rules.md).

## Selecting a subset of rules

By default Sentinel runs all fifteen rules. To run only a specific subset, pass `rules=`:

```python
sentinel = Sentinel(
    project="my-agent",
    mode="log",
    rules=["embedding_waste", "retry_storm"],  # only these two
)
```

The string in the list is the rule's `name` attribute. Valid names:

```
tool_loop, context_bloat, embedding_waste, zombie, model_misroute,
retry_storm, tool_definition_bloat, retrieval_thrash,
vision_re_upload, vision_high_detail_misroute, vision_cost_concentration,
audio_multichannel_doubling, voice_switching_loop, rerank_thrash, repair_loop
```

Default is `"all"`.

Disabling rules is the right tool for "I don't care about this leak class" (e.g., a stateless inference API that can't have `context_bloat`). For "I want this rule but tuned looser", use `config` instead.

## Recommended graduation pattern

Don't go straight to `block`. The right sequence for a new project:

1. **Week 1: `log` everywhere.** Wire up handlers that ship events to your logging stack. Watch what fires. Note any handler-side false positives.
2. **Week 2-4: tune.** Adjust per-rule `config` keys on rules that are noisy for your traffic. The defaults err toward fewer false positives, but every project's ground truth is different.
3. **Month 2: `block` on the highest-confidence rules.** Start with `embedding_waste` (0.99 confidence, exact-hash match — never wrong) and `retry_storm` (0.9 confidence, deterministic hash repeat — almost never wrong). Leave `tool_loop` and `model_misroute` in `log` longer because they are heuristic.
4. **Ongoing: per-rule mode.** TokenSentinel doesn't currently support per-rule mode — it's all-or-nothing on `block`. If you need finer control, run two `Sentinel` instances on the same client (one in `log`, one in `block` with `rules=` filter) and chain them.

## When NOT to use `block`

- **Your agent doesn't have a recovery path.** `block` raises `LeakDetected`; your code must catch it and do something sensible. If you'll just propagate the exception to the user, you are turning a quiet leak into a loud failure.
- **You haven't tuned thresholds yet.** Defaults are good but not optimal for every workload. Run `log` first, eyeball false positives, then promote.
- **The rule is heuristic and the cost of a wrong block is high.** `model_misroute` is a useful signal but not always right; blocking it would prevent some legitimate frontier-model classification work.

## Handler requirements

**Handlers run synchronously, inline with the wrapped LLM call.** A handler that takes 5s adds 5s of latency to every leak-firing LLM call your agent makes. This isn't a TokenSentinel problem — it's the handler's. Two rules:

1. **Don't do network I/O in a handler.** Don't post to Slack, write to Datadog, or call an HTTP webhook directly. Queue the event to a background worker (`queue.Queue`, `concurrent.futures`, etc.) and let the worker do the network call.
2. **Don't sleep, lock contention, or `await` in a handler.** Handlers are sync; an `await` inside a handler that blocks on the running event loop will deadlock asyncio agents.

A safe handler pattern:

```python
import queue
_event_q: queue.Queue = queue.Queue(maxsize=1000)

@sentinel.on_leak
def fast_handler(event):
    try:
        _event_q.put_nowait(event)
    except queue.Full:
        pass  # drop on overflow rather than block

# in a separate worker thread / asyncio task:
def drain_events():
    while True:
        event = _event_q.get()
        slack.post_message(event=event)  # actual network call here, off the hot path
```

This matters most in `block` mode where the `LeakDetected` raise unwinds through the wrapped call — slow handlers extend that unwind in front of the user's exception handler.

## Streaming + block mode (best-effort on abandoned streams)

For streamed responses, the `CallRecord` is built and rules evaluated when the stream **finalizes**. If the user breaks out of the iteration loop without using `with` or calling `close()`, the stream gets cleaned up via Python's garbage collector — and Python silently swallows exceptions raised during GC.

What this means in practice:

- **`with stream:` exits normally** → `LeakDetected` propagates to the surrounding `except` block. Block mode works as expected.
- **Full iteration to completion** → same. Block mode works.
- **`break` out of the loop without `with` / `close()`** → the leak is still detected and your handlers are still called (the `record_call` happens), but `LeakDetected` is suppressed with a `RuntimeWarning`. Block mode is best-effort here because Python won't propagate the exception from a GC path.
- **Process exit (Ctrl-C, kill -9)** → in-flight streams may not finalize at all. The last batch of events is best-effort.

If you depend on block mode for streams, use `with`:

```python
with client.messages.stream(...) as stream:
    for event in stream:
        ...
# On block mode, LeakDetected raises here on __exit__ as expected.
```
- **You only have one agent run.** `block` shines when bad behavior is repeating and you want to halt the *loop*. For a single bad call, `log` and a post-hoc remediation in your handler is usually more valuable.

## Mode and rule confidence interaction

Mode controls *what happens when a rule fires*. `min_confidence` and per-rule thresholds control *whether a rule fires at all*. They are independent.

A rule that emits at confidence 0.6 will:

| `min_confidence` | mode | Outcome |
|---|---|---|
| 0.5 | log | handler called; no exception; cloud only if configured |
| 0.5 | alert | same as log locally; cloud only if configured |
| 0.5 | block | handler called; `LeakDetected` raised; cloud only if configured |
| 0.7 | any | event dropped silently before reaching the handler |

Use `min_confidence` to silence rules. Use `mode="block"` to halt agents on rules you trust.

Now read [Leak rules](./04-waste-rules.md) for what each rule actually detects and how to tune it.
