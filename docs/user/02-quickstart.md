# Quickstart

Five-minute end-to-end tutorial. By the end you will have a working TokenSentinel install, a wrapped client, a registered leak handler, and a leak event in your terminal.

We use Anthropic for the example because it is the most-deployed agent provider as of mid-2026. Substitute another provider freely — the API surface is identical (see [Providers](./05-providers.md)).

## Prerequisites

- Python 3.10 or later.
- An Anthropic API key in `ANTHROPIC_API_KEY`.

## Step 1 — install

```bash
pip install token-sentinel[anthropic]
```

## Step 2 — wrap the client

Create `quickstart.py`:

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="quickstart", mode="log")
client = sentinel.wrap(anthropic.Anthropic())
```

Two important things happened:

1. `Sentinel(project=...)` constructed the in-process detector. `mode="log"` is the safe default — Sentinel will only call your handler when a leak fires, never raise an exception.
2. `sentinel.wrap(client)` mutated the client in place to instrument `messages.create` and `messages.stream`. The returned object is the same `client` you passed in, so all your IDE type hints continue to work. Use it exactly as you would a normal `anthropic.Anthropic`.

## Step 3 — register a handler

```python
@sentinel.on_leak
def handle(event):
    print(f"\n>>> LEAK <<<")
    print(f"  type:        {event.type}")
    print(f"  confidence:  {event.confidence:.2f}")
    print(f"  rule:        {event.rule}")
    print(f"  burn_est:    ${event.estimated_burn:.4f}")
    print(f"  evidence:    {event.evidence}")
    print(f"  suggestion:  {event.suggested_action}")
```

`@sentinel.on_leak` registers a callable that fires for every leak above the project's `min_confidence` threshold (default `0.5`). You can register multiple handlers — they run in registration order, and exceptions in one handler do not block others.

## Step 4 — trigger a leak

The simplest leak to trigger with one real call is `model_misroute`: it fires when a classification-shaped prompt is sent to a frontier model. Add this to `quickstart.py`:

```python
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[
        {"role": "user", "content": "Classify this as positive or negative: 'I love this movie'"},
    ],
)
```

Run it:

```bash
python quickstart.py
```

You should see something like:

```
>>> LEAK <<<
  type:        model_misroute
  confidence:  0.70
  rule:        v0.model_misroute
  burn_est:    $0.0050
  evidence:    {'model': 'claude-sonnet-4-6', 'prompt_tokens': 24, 'completion_tokens': 6,
                'matched_keywords': ['classify'], 'recommended_alternative': 'claude-haiku-4-5'}
  suggestion:  route_to_claude-haiku-4-5
```

That fired because:

- The prompt was small (under 500 tokens) — heuristic for "this is a classification task".
- The expected output was small (under 50 tokens) — same heuristic.
- The prompt contained the keyword `"classify"` — confirms classification shape.
- The model was `claude-sonnet-4-6` — a frontier model, where Haiku would do.

The `recommended_alternative` field tells you which cheaper model to route to. The `estimated_burn` is the dollar cost of *this single call*; in production where this fires hundreds of times a day, multiply accordingly.

## Step 5 — make sense of the event fields

Every leak emits a `LeakEvent` dataclass with these fields. You will see the same shape for every rule.

| Field | Type | What it is |
|---|---|---|
| `type` | str | Leak class — one of `tool_loop`, `context_bloat`, `embedding_waste`, `zombie`, `model_misroute`, `retry_storm`, `tool_definition_bloat`, `retrieval_thrash`. |
| `confidence` | float | 0.0–1.0. Below `min_confidence` (default 0.5) the event is dropped before reaching your handler. |
| `project` | str | Whatever you passed to `Sentinel(project=...)`. Useful when one process runs multiple Sentinels. |
| `session_id` | str | Identifies a single agent run. Defaults to per-call UUID; pass a stable ID to group calls (see below). |
| `rule` | str | Which rule fired, prefixed with the rules-engine version (e.g. `v0.model_misroute`). |
| `evidence` | dict | Rule-specific payload. Keys are documented per rule in [Leak rules](./04-waste-rules.md). Always include enough detail to reproduce the firing. |
| `estimated_burn` | float | Rough dollar figure for the wasted spend this leak represents. Treat it as a sort key, not an invoice. |
| `suggested_action` | str | Machine-readable hint for what to do — `route_to_claude-haiku-4-5`, `add_embedding_cache`, `pause_for_human_review`, etc. |
| `raised_at` | datetime | UTC timestamp the event was emitted. |

The full type definition lives in [API reference](./07-api-reference.md).

## Stable session IDs

By default each call gets its own UUID `session_id`. That is fine for one-shot calls but defeats the purpose of any rule that looks for *patterns* (`tool_loop`, `retry_storm`, `context_bloat`, `zombie`, `embedding_waste`, `retrieval_thrash`).

To group calls into a single agent run, pass `_sentinel_session_id` as a kwarg on the wrapped client call:

```python
session = "user-42-task-17"
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
client.messages.create(model="...", messages=[...], _sentinel_session_id=session)
```

The `_sentinel_session_id` kwarg is intercepted by the wrapper before the call goes out, so the underlying SDK never sees it. Use any string — a UUID, a request ID, a user ID, whatever uniquely identifies "one logical agent run".

## What just happened, end to end

```
your code            wrapped anthropic.Anthropic       Sentinel
─────────            ───────────────────────────       ────────
client.messages
  .create(...)  ──▶  intercept kwargs (pop session_id)
                     time the underlying call
                     call original_create(...)
                     build CallRecord from response
                     hand record to sentinel  ──────▶  tracer.record(call)
                                                       run all 8 rules
                                                       for any rule that fired
                                                       above min_confidence:
                                                         dispatch LeakEvent
                                                         to your handlers  ──▶  print(event)
                     return response  ◀───────────────  (mode=log: just return)
                                                        (mode=block: raise LeakDetected)
your code   ◀──────  Message object (unchanged)
```

Total overhead: under 10ms p95 per call for the V0 rule set. The rules are pure functions of the per-session ring buffer — no I/O, no network calls.

## What to do next

You have a working integration. Now:

- **Tune for your project.** Read [Modes](./03-modes.md) and decide whether `log` is enough for your team or you want `block`.
- **Understand which rules apply to you.** Each rule has different thresholds and false-positive characteristics. See [Leak rules](./04-waste-rules.md).
- **Wire it into your stack.** [Integrations](./06-integrations.md) covers MCP, RAG, LangChain, LangGraph, CrewAI, AutoGen, and Pydantic AI.
- **Reference the full surface.** [API reference](./07-api-reference.md) documents every kwarg.
- **Find help when something goes wrong.** [Troubleshooting](./08-troubleshooting.md).

If you want to see all eight rules fire on synthetic data without making real API calls, run `examples/tool_loop_demo.py` from the repo. It uses `Sentinel.record_call` directly to inject hand-crafted `CallRecord` objects.
