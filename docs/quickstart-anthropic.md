# Quickstart — Anthropic

A 5-minute end-to-end: install, wrap, see a leak fire.

## 1. Install

```bash
pip install token-sentinel[anthropic]
```

The `[anthropic]` extra pulls in `anthropic>=0.39.0`. Python 3.10+.

## 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The Anthropic SDK reads `ANTHROPIC_API_KEY` automatically. You can also pass `api_key=` directly to the constructor.

## 3. Wrap your client

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")

@sentinel.on_leak
def handle(event):
    print(
        f"LEAK type={event.type} "
        f"confidence={event.confidence:.2f} "
        f"burn=${event.estimated_burn:.4f} "
        f"action={event.suggested_action} "
        f"evidence_keys={list(event.evidence.keys())}"
    )

client = sentinel.wrap(anthropic.Anthropic())
```

`sentinel.wrap` mutates the client in place — `client.messages.create` and `client.messages.stream` are now instrumented. The returned object is the same `anthropic.Anthropic` instance, so all your IDE type hints continue to work.

## 4. Trigger a leak

The simplest signal to fire reliably with one real call is `model_misroute`: a classification-shaped prompt aimed at a frontier model.

```python
SESSION = "demo-session-1"

client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[{
        "role": "user",
        "content": "Classify this as positive or negative: 'I love this movie'",
    }],
    _sentinel_session_id=SESSION,
)
```

The `_sentinel_session_id` kwarg is intercepted by the wrapper before the call goes out, so the underlying SDK never sees it. Pass any stable string to group calls into one logical agent run.

The rule fires because:
- The prompt is small (under 500 tokens) — heuristic for "this is a classification task".
- The expected output is small (`max_tokens=10`).
- The prompt contains the keyword `classify`.
- The model is a frontier model where Haiku would do.

## 5. See it land in your handler

You should see something like:

```
LEAK type=model_misroute confidence=0.70 burn=$0.0050 action=route_to_claude-haiku-4-5 evidence_keys=['model', 'prompt_tokens', 'completion_tokens', 'matched_keywords', 'recommended_alternative']
```

The handler receives a `LeakEvent` dataclass with these fields:

| Field | Type | What it is |
|---|---|---|
| `type` | str | One of the eight V0 rule types (or composite types fired cloud-side) |
| `confidence` | float | 0.0-1.0; below `min_confidence` (default 0.5) the event is dropped |
| `project` | str | What you passed to `Sentinel(project=...)` |
| `session_id` | str | Identifies a single agent run |
| `rule` | str | Which rule fired (e.g. `v0.model_misroute`) |
| `evidence` | dict | Rule-specific payload — keys documented per rule |
| `estimated_burn` | float | Rough dollar figure for the wasted spend this leak represents |
| `suggested_action` | str | Machine-readable hint (`route_to_claude-haiku-4-5`, `add_embedding_cache`, …) |
| `raised_at` | datetime | UTC timestamp |
| `metadata` | dict | Cloud-side judge verdict trail when ratification fires (Pro tier) |

## Streaming

`messages.stream` is a context manager. Use it with `with` to guarantee finalisation:

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": "Write a haiku about token leaks."}],
    _sentinel_session_id=SESSION,
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta.text, end="", flush=True)
print()
```

The wrapper observes every event in flight and finalises a `CallRecord` from `MessageStop` plus `stream.get_final_message()` on `__exit__`. Token usage is aggregated from `message_delta.usage` deltas. Leak handlers fire on the way out of the `with` block.

## Async

`anthropic.AsyncAnthropic` is detected automatically — the same `sentinel.wrap` works:

```python
import asyncio
import anthropic
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent", mode="log")

async def main():
    client = sentinel.wrap(anthropic.AsyncAnthropic())
    await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Categorise this email as spam or ham"}],
    )

asyncio.run(main())
```

Async streaming mirrors the sync surface — replace `with` with `async with`:

```python
async with client.messages.stream(...) as stream:
    async for event in stream:
        ...
```

## Going to production

- Switch from `mode="log"` to `mode="alert"` to get cloud-side dashboards. Configure `cloud_endpoint` and `api_key`:
  ```python
  sentinel = Sentinel(
      project="my-agent",
      mode="alert",
      cloud_endpoint="https://api.tokensentinel.dev",
      api_key="tsk_...",
  )
  ```
- For hard intervention, set `mode="block"` to raise `LeakDetected` at the next call boundary. Wrap calls in `try / except LeakDetected as exc:` and inspect `exc.event`.
- Pair with the cloud (Team / Pro) — see [`docs/pricing.md`](pricing.md). Pro adds the Intervention Pack (budget cap per session, velocity cap, kill-switch) which raises `BudgetExceeded` / `VelocityExceeded` / `KillSwitchActive` regardless of mode.
- Long-running agents: call `sentinel.close(timeout=5.0)` before exit to flush the cloud sink and stop the policy poller daemon thread.

## Common issues

- **`Unsupported client type: Anthropic from anthropic.X`** — make sure you're on the official `anthropic` SDK 0.39+, not a fork. The dispatcher checks `type(client).__module__`.
- **Block mode on streamed calls aborted via GC** — `LeakDetected` is suppressed with a `RuntimeWarning` rather than vanishing silently. Always use `with stream:` (or fully iterate) to guarantee block-mode halts; abandoning the stream falls back to best-effort.
- **Streaming token counts look low** — Anthropic emits running totals on `message_delta.usage` and a final count on `message_stop.message.usage`. The wrapper takes `max()` across both so it never regresses, but if you see `prompt_tokens=0` post-stream it's a sign the underlying SDK didn't emit either event — usually a partial server response or a mocked client.
- **Want to call `sentinel.unregister(handler)`** — the symmetric counterpart to `on_leak` exists; returns `True` if the handler was found and removed.
