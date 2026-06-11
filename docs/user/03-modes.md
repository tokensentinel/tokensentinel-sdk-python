# Modes

TokenSentinel has three modes that control what happens when a rule fires: `log`, `alert`, and `block`. Pick the one that matches how much you trust the rules to be right.

```python
sentinel = Sentinel(project="my-agent", mode="log")    # default
sentinel = Sentinel(project="my-agent", mode="alert")
sentinel = Sentinel(project="my-agent", mode="block")
```

| Mode | What it does | When to use |
|---|---|---|
| `log` | Calls your registered `on_leak` handlers. Never raises. | Day-one production. Always. |
| `alert` | Same as `log` plus emits to the cloud dashboard and webhooks (cloud feature, optional). | Once you have handlers shipping events somewhere durable and want richer dashboards. |
| `block` | Same as `log` plus raises `LeakDetected` to halt the wrapped call's caller. | Once a specific rule has been firing cleanly in `log` for a while and you trust it to short-circuit. |

The modes are **strictly additive in severity** — anything that happens in `log` also happens in `alert` and `block`. You always get your handlers called.

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

## `alert` mode (cloud-augmented logging)

```python
sentinel = Sentinel(
    project="my-agent",
    mode="alert",
    cloud_endpoint="https://api.tokensentinel.dev",
    api_key="ts_live_...",
)

@sentinel.on_leak
def handle(event):
    print(f"LEAK [{event.type}] {event.confidence:.2f}")
```

`alert` adds two things on top of `log`:

1. Events are also POSTed to the configured `cloud_endpoint`, batched and fire-and-forget (network failures don't block your agent).
2. The cloud dashboard runs its own webhook routing — Slack, PagerDuty, Linear, custom HTTPS, etc. — keyed off rule type and severity.

`alert` is functionally a superset of `log`: your handlers still fire identically. If you don't configure `cloud_endpoint` and `api_key`, `alert` behaves exactly like `log` (the cloud sink degrades silently).

The cloud dashboard is closed-source and optional. The SDK works perfectly without it; you just won't get the hosted dashboards / retention / team features. See the [FAQ](./09-faq.md) for the OSS-vs-cloud split.

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

- **The LLM call already happened.** `block` cannot prevent the *current* call from being billed — by the time the rule has enough signal to fire, the call's response is already on the way back. `block` halts the *next* call in a degenerate loop. That is the actual savings.
- **Handler exceptions are still caught.** If your handler raises, that does not block the agent — the `LeakDetected` exception comes from `Sentinel`, not from your handler.
- **Streaming calls block at the stream's close.** For Anthropic / Gemini / Bedrock streaming, the rule fires when the stream context manager exits. `LeakDetected` is raised from `__exit__` / `__aexit__`. You should design your stream consumer to handle this.

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

By default Sentinel runs all eight rules. To run only a specific subset, pass `rules=`:

```python
sentinel = Sentinel(
    project="my-agent",
    mode="log",
    rules=["embedding_waste", "retry_storm"],  # only these two
)
```

The string in the list is the rule's `name` attribute. Valid names: `tool_loop`, `context_bloat`, `embedding_waste`, `zombie`, `model_misroute`, `retry_storm`, `tool_definition_bloat`, `retrieval_thrash`. Default is `"all"`.

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
| 0.5 | log | handler called; no exception |
| 0.5 | alert | handler called; cloud sink notified |
| 0.5 | block | handler called; `LeakDetected` raised |
| 0.7 | any | event dropped silently before reaching the handler |

Use `min_confidence` to silence rules. Use `mode="block"` to halt agents on rules you trust.

Now read [Leak rules](./04-waste-rules.md) for what each rule actually detects and how to tune it.
