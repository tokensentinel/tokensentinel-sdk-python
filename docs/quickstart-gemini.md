# Quickstart — Google Gemini

A 5-minute end-to-end: install, wrap, see a leak fire.

## 1. Install

```bash
pip install token-sentinel[gemini]
```

The `[gemini]` extra pulls in `google-genai>=1.0.0` — the modern unified SDK that covers both the direct Gemini API and Vertex AI. (The legacy `google-generativeai` and `vertexai` packages are deprecated; this wrapper does not target them.) Python 3.10+.

## 2. Set your API key

For the direct Gemini API:

```bash
export GOOGLE_API_KEY="..."
```

The `google.genai.Client(api_key=...)` constructor reads `GOOGLE_API_KEY` automatically, or accepts the key as a kwarg. For Vertex AI, see the **Vertex AI** section below — you authenticate via Application Default Credentials, not a static API key.

## 3. Wrap your client

```python
from token_sentinel import Sentinel
from google import genai

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

client = sentinel.wrap(genai.Client())
```

`sentinel.wrap` mutates the client in place — both the sync surface (`client.models.generate_content`, `client.models.generate_content_stream`) and the async surface (`client.aio.models.*`) are now instrumented. The returned object is the same `genai.Client` instance, so all your IDE type hints continue to work.

## 4. Trigger a leak

The simplest signal to fire reliably with one real call is `model_misroute`: a classification-shaped prompt aimed at a frontier model.

```python
SESSION = "demo-session-1"

client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Classify this as positive or negative: 'I love this movie'",
    _sentinel_session_id=SESSION,
)
```

The `_sentinel_session_id` kwarg is intercepted by the wrapper before the call goes out, so the underlying SDK never sees it. Pass any stable string to group calls into one logical agent run.

The rule fires because:
- The prompt is small (under 500 tokens).
- The output is short.
- The prompt contains the keyword `classify`.
- The model is `gemini-2.5-pro`, a frontier model where `gemini-2.5-flash` would do.

## 5. See it land in your handler

You should see something like:

```
LEAK type=model_misroute confidence=0.70 burn=$0.0050 action=route_to_gemini-2.5-flash evidence_keys=['model', 'prompt_tokens', 'completion_tokens', 'matched_keywords', 'recommended_alternative']
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

`generate_content_stream` returns a plain iterator (no context manager) — the wrapper proxies it and finalises the `CallRecord` when iteration ends:

```python
stream = client.models.generate_content_stream(
    model="gemini-2.5-pro",
    contents="Write a haiku about token leaks.",
    _sentinel_session_id=SESSION,
)
for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
```

Token usage is read from `chunk.usage_metadata` (cumulative, per the docs). The wrapper takes `max()` across chunks so it never regresses on a final-chunk quirk. `function_call` parts are accumulated into the same `tool_calls` shape the non-streaming path produces.

If you abandon the stream early (break out, garbage-collect mid-iter), `LeakDetected` from block mode is suppressed with a `RuntimeWarning` rather than vanishing silently. Fully iterate (or use a `with` block via the iterator's `close()`) for guaranteed block-mode halts.

## Async

The async surface lives at `client.aio.models.*` and is wrapped automatically — the same `sentinel.wrap` covers both:

```python
import asyncio
from google import genai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent", mode="log")

async def main():
    client = sentinel.wrap(genai.Client())
    await client.aio.models.generate_content(
        model="gemini-2.5-pro",
        contents="Categorise this email as spam or ham",
    )

asyncio.run(main())
```

Async streaming uses `async for`. `client.aio.models.generate_content_stream` is a coroutine that returns an async iterator, so you `await` once and then iterate:

```python
stream = await client.aio.models.generate_content_stream(
    model="gemini-2.5-pro",
    contents="...",
)
async for chunk in stream:
    ...
```

## Vertex AI

The same wrapper transparently covers Vertex AI — pass `vertexai=True` to the constructor:

```python
import google.auth
from google import genai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent", mode="log")
client = sentinel.wrap(
    genai.Client(
        vertexai=True,
        project="my-gcp-project",
        location="us-central1",
    )
)
client.models.generate_content(model="gemini-2.5-pro", contents="...")
```

Authentication uses Google Cloud Application Default Credentials. Run `gcloud auth application-default login` once locally, or attach a service-account in production (Workload Identity, GKE node SA, Cloud Run, etc.).

The dispatcher routes both backends through the same code path because `type(client).__module__` is `google.genai.client` either way. Streaming, async, async-streaming all work identically. The legacy `vertexai` SDK (`from vertexai.generative_models import GenerativeModel`) is **not** instrumented — migrate to `google-genai` with `vertexai=True`.

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

- **`Unsupported client type: Client from google.genai...`** — the SDK is correctly named `google-genai`, **not** the legacy `google-generativeai`. Run `pip show google-genai` to confirm. The wrapper detects on the `google.genai` / `google_genai` module prefix.
- **Wrapping the legacy `vertexai` SDK silently does nothing** — the legacy `from vertexai.generative_models import GenerativeModel` is deprecated and not wrapped. Migrate to `genai.Client(vertexai=True, project=..., location=...)`. The dispatcher will raise `TypeError: Unsupported client type` on a legacy client.
- **`tool_loop` rule fires unexpectedly on `function_call` parts** — Gemini's `function_call.args` are exposed as a dict-like; the wrapper coerces to a plain dict for stable hashing across runs. If you see a tool-loop signal you don't expect, inspect `event.evidence["redacted_args_summary"]` — the redaction layer ships sorted key names + per-key value lengths + a SHA-256 prefix, never the raw args.
- **Vertex AI 401 / 403 errors** — Application Default Credentials are not configured. Run `gcloud auth application-default login` (local), set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json` (CI), or attach a service account (managed environments).
