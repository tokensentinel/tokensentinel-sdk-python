# Quickstart — OpenAI

A 5-minute end-to-end: install, wrap, see a leak fire.

## 1. Install

```bash
pip install token-sentinel[openai]
```

The `[openai]` extra pulls in `openai>=1.50.0`. Python 3.10+. The same install also covers every OpenAI-compatible provider — see below.

## 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

The OpenAI SDK reads `OPENAI_API_KEY` automatically. You can also pass `api_key=` directly to the constructor.

## 3. Wrap your client

```python
from token_sentinel import Sentinel
import openai

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

client = sentinel.wrap(openai.OpenAI())
```

`sentinel.wrap` mutates the client in place — `client.chat.completions.create` and `client.embeddings.create` are now instrumented. The returned object is the same `openai.OpenAI` instance, so all your IDE type hints continue to work.

## 4. Trigger a leak

The simplest signal to fire reliably with one real call is `model_misroute`: a classification-shaped prompt aimed at a frontier model.

```python
SESSION = "demo-session-1"

client.chat.completions.create(
    model="gpt-5",
    max_tokens=10,
    messages=[{
        "role": "user",
        "content": "Classify this as positive or negative: 'I love this movie'",
    }],
    _sentinel_session_id=SESSION,
)
```

The `_sentinel_session_id` kwarg is intercepted by the wrapper (popped from `**kwargs`) before the call reaches the OpenAI SDK, so the underlying client never sees it. Pass any stable string to group calls into one logical agent run.

The rule fires because:
- The prompt is small (under 500 tokens) — heuristic for "this is a classification task".
- The expected output is small (`max_tokens=10`).
- The prompt contains the keyword `classify`.
- The model is a frontier model where a smaller model would do.

## 5. See it land in your handler

You should see something like:

```
LEAK type=model_misroute confidence=0.70 burn=$0.0050 action=route_to_gpt-4o-mini evidence_keys=['model', 'prompt_tokens', 'completion_tokens', 'matched_keywords', 'recommended_alternative']
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
| `suggested_action` | str | Machine-readable hint |
| `raised_at` | datetime | UTC timestamp |
| `metadata` | dict | Cloud-side judge verdict trail when ratification fires (Pro tier) |

## Streaming

OpenAI streaming is fully instrumented as of stable release. The wrapper returns a proxy iterator that siphons each `ChatCompletionChunk` into a usage accumulator and flushes a `CallRecord` on iteration end / close / GC.

Pass `stream_options={"include_usage": True}` if you want token counts in the final `CallRecord` — without it, OpenAI omits usage from streamed chunks and the wrapper sets `raw_response_meta["usage_unavailable"] = True`.

```python
stream = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Write a haiku about token leaks."}],
    stream=True,
    stream_options={"include_usage": True},
    _sentinel_session_id=SESSION,
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)
print()
```

`tool_calls` arrive as `ChoiceDeltaToolCall` deltas — the wrapper stitches them by `index` into the same shape the non-streaming path produces, so downstream rules (`tool_loop`, `retrieval_thrash`) see consistent records.

## Async

`openai.AsyncOpenAI` is detected automatically — the same `sentinel.wrap` works:

```python
import asyncio
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent", mode="log")

async def main():
    client = sentinel.wrap(openai.AsyncOpenAI())
    await client.chat.completions.create(
        model="gpt-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Categorise this email as spam or ham"}],
    )

asyncio.run(main())
```

Async streaming uses `async for`:

```python
stream = await client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "..."}],
    stream=True,
    stream_options={"include_usage": True},
)
async for chunk in stream:
    ...
```

## Transparent coverage of OpenAI-compatible providers

Every provider that exposes an OpenAI-compatible REST surface keeps `type(client).__module__ == "openai..."` and picks up `wrap_openai` for free. No provider-specific code path. Swap `base_url`:

| Provider | `base_url` | Notes |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | DeepSeek-V3 (`deepseek-chat`), R1 (`deepseek-reasoner`) |
| Together AI | `https://api.together.xyz/v1` | Llama, Qwen, DeepSeek, Mixtral hosted |
| Fireworks | `https://api.fireworks.ai/inference/v1` | Llama, Qwen, FireFunction |
| Groq | `https://api.groq.com/openai/v1` | Cheap, ultra-fast inference |
| OpenRouter | `https://openrouter.ai/api/v1` | Multi-model gateway |
| Anyscale | `https://api.endpoints.anyscale.com/v1` | Open models on Anyscale Endpoints |
| Mistral La Plateforme | `https://api.mistral.ai/v1` | Mistral Small / Large / Codestral |
| Perplexity | `https://api.perplexity.ai` | Sonar models with web grounding |
| vLLM | `http://your-vllm:8000/v1` | Native OpenAI-compat |
| Ollama | `http://localhost:11434/v1` | API key any non-empty string |
| text-generation-inference | (your TGI host)/v1 | HuggingFace TGI OpenAI-compat |
| LM Studio | `http://localhost:1234/v1` | Local LLM GUI |

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent", mode="log")

# DeepSeek
client = sentinel.wrap(openai.OpenAI(api_key="...", base_url="https://api.deepseek.com"))
client.chat.completions.create(model="deepseek-chat", messages=[...])

# Self-hosted vLLM
client = sentinel.wrap(openai.OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1"))
client.chat.completions.create(model="llama-3.3-70b", messages=[...])
```

Self-hosted footnote: waste signals (token-count + call-pattern based) remain accurate, but the dollar-burn estimate assumes priced public-API usage — treat it as a quality signal, not a billing signal.

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
- Pair with the cloud (Team / Pro) — see [`docs/pricing.md`](pricing.md). Pro adds the Intervention Pack which raises `BudgetExceeded` / `VelocityExceeded` / `KillSwitchActive` regardless of mode.
- Long-running agents: call `sentinel.close(timeout=5.0)` before exit to flush the cloud sink and stop the policy poller daemon thread.

## Common issues

- **Custom `base_url` and the wrapper still works** — that's the design. Any OpenAI-compatible provider (DeepSeek, Together, Groq, vLLM, Ollama, …) is transparently covered. The CallRecord shows `provider="openai"` because the wrapper can't distinguish them by SDK type alone; the model name lands in `record.model`.
- **Streamed CallRecords have `prompt_tokens=0`** — pass `stream_options={"include_usage": True}` so OpenAI emits a final usage block. Without it `raw_response_meta["usage_unavailable"]` is set to `True` and you'll see the gap in dashboards.
- **Block mode on streamed calls aborted via GC / generator close** — `LeakDetected` is suppressed with a `RuntimeWarning` rather than vanishing silently. Always use `with stream:` (or fully iterate) to guarantee block-mode halts; abandoning the stream falls back to best-effort.
- **Mocked clients in tests fail to wrap** — the dispatcher validates that `client.chat.completions.create` and `client.embeddings.create` are reachable. A partially-built mock will raise `TypeError` with the original `AttributeError` chained via `__cause__`.
