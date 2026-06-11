# Providers

TokenSentinel ships first-class wrappers for the four major LLM platforms and works transparently with any OpenAI-compatible endpoint. This page is the canonical reference for what's supported.

## Choose your provider

```
                       ┌──────────────────────────────┐
                       │  Which SDK does your code    │
                       │  import to call the model?   │
                       └─────────────┬────────────────┘
                                     │
       ┌─────────────┬───────────────┼─────────────────┬────────────────┐
       │             │               │                 │                │
       ▼             ▼               ▼                 ▼                ▼
  ┌─────────┐  ┌─────────┐  ┌──────────────┐   ┌────────────┐    ┌──────────┐
  │anthropic│  │ openai  │  │ google.genai │   │   boto3    │    │  other   │
  └────┬────┘  └────┬────┘  └──────┬───────┘   └─────┬──────┘    └────┬─────┘
       │            │              │                 │                │
       │            │              │                 │                ▼
       │            │              │                 │      ┌─────────────────┐
       │            │              │                 │      │ Does it support │
       │            │              │                 │      │ OpenAI-compat   │
       │            │              │                 │      │ endpoints?      │
       │            │              │                 │      │ (vLLM, Ollama,  │
       │            │              │                 │      │  DeepSeek, etc) │
       │            │              │                 │      └────────┬────────┘
       │            │              │                 │               │
       │            │              │                 │       ┌───────┴───────┐
       │            │              │                 │     yes              no
       │            │              │                 │       │                │
       │            │              │                 │       ▼                ▼
       │            │              │                 │  use openai     not supported
       │            │              │                 │  with custom    (request a wrapper)
       │            │              │                 │  base_url
       │            │              │                 │
       ▼            ▼              ▼                 ▼
  [anthropic]   [openai]       [gemini]          [bedrock]
```

| Your SDK / endpoint | Install extra | Wrapper | Streaming | Async |
|---|---|---|---|---|
| `anthropic.Anthropic` / `AsyncAnthropic` | `[anthropic]` | `wrap_anthropic` | yes | yes |
| `openai.OpenAI` / `AsyncOpenAI` | `[openai]` | `wrap_openai` | passthrough (instrumentation tracked) | yes |
| `google.genai.Client` | `[gemini]` | `wrap_gemini` | yes | yes |
| `boto3.client("bedrock-runtime")` | `[bedrock]` | `wrap_bedrock` | yes | sync only |

How dispatch works: `Sentinel.wrap(client)` looks at `type(client).__module__` and routes to the right wrapper. OpenAI-compatible providers are accessed through the official `openai` SDK with a custom `base_url`, so they keep `__module__ == "openai..."` and pick up `wrap_openai` for free — no provider-specific code path needed.

---

## Native providers

### Anthropic

```python
import anthropic
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(anthropic.Anthropic())
client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[...])
```

Covers: `messages.create` (sync + async), `messages.stream` (sync + async). Token counts come from the response's `usage` for non-streaming and from accumulated `message_delta.usage` events plus the final `message_stop` for streaming.

### OpenAI

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(openai.OpenAI())
client.chat.completions.create(model="gpt-5", messages=[...])
client.embeddings.create(model="text-embedding-3-small", input="...")
```

Covers: `chat.completions.create` (sync + async, non-streaming), `embeddings.create` (sync + async). Streaming via `stream=True` is currently passthrough — no record is emitted for streaming calls; full stream instrumentation is tracked for an upcoming release. In `mode='block'`, streaming calls emit a `RuntimeWarning` once per Sentinel so you know detection is bypassed for that path.

### Google Gemini

```python
from google import genai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(genai.Client(api_key="..."))
client.models.generate_content(model="gemini-2.5-pro", contents="...")
```

Covers: `models.generate_content` and `models.generate_content_stream` on both the sync `client.models.*` surface and the async `client.aio.models.*` surface.

For Google Vertex AI, use the same `google.genai` SDK with `vertexai=True`:

```python
client = sentinel.wrap(genai.Client(vertexai=True, project="my-gcp-project", location="us-central1"))
```

The dispatch is identical because the module prefix is `google.genai` either way.

### AWS Bedrock

```python
import boto3
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))
client.converse(modelId="anthropic.claude-sonnet-4-5-v2:0", messages=[...])
```

Covers: `converse` (non-streaming) and `converse_stream` (streaming). Sync only — `aioboto3` is third-party and is not currently instrumented. The lower-level `invoke_model` / `invoke_model_with_response_stream` paths are not instrumented today; use `converse` / `converse_stream` (which all current Bedrock-supported models accept) to get full coverage.

---

## OpenAI-compatible providers

These providers all expose the OpenAI Chat Completions / Embeddings API shape. Instantiate the official `openai` SDK with a custom `base_url`, and TokenSentinel's `wrap_openai` instruments it identically to OpenAI.

| Provider | `base_url` | Notes |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | DeepSeek-V3 (`deepseek-chat`), R1 (`deepseek-reasoner`). |
| Together AI | `https://api.together.xyz/v1` | Llama, Qwen, DeepSeek, Mixtral hosted. |
| Fireworks | `https://api.fireworks.ai/inference/v1` | Llama, Qwen, FireFunction. |
| Groq | `https://api.groq.com/openai/v1` | Cheap, ultra-fast inference. |
| OpenRouter | `https://openrouter.ai/api/v1` | Multi-model gateway. |
| Anyscale | `https://api.endpoints.anyscale.com/v1` | Open models on Anyscale Endpoints. |
| Mistral La Plateforme | `https://api.mistral.ai/v1` | Mistral Small / Large / Codestral. |
| Perplexity | `https://api.perplexity.ai` | Sonar models with web grounding. |

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(
    openai.OpenAI(api_key="...", base_url="https://api.deepseek.com"),
)
client.chat.completions.create(model="deepseek-chat", messages=[...])
```

The same four lines work for any row in the table — change `api_key` and `base_url`. The emitted `CallRecord` will show `provider="openai"` because the wrapper can't distinguish DeepSeek from OpenAI from the SDK type alone, but the actual model lands in `record.model` (`deepseek-chat`, `llama-3.3-70b`, `mixtral-8x7b`, etc.).

---

## Self-hosted

Open-source LLM servers that ship an OpenAI-compatible endpoint also pick up `wrap_openai` for free. Point `base_url` at your server.

| Stack | Setup |
|---|---|
| vLLM | Native OpenAI-compat. `base_url="http://your-vllm:8000/v1"`. |
| Ollama | `base_url="http://localhost:11434/v1"`. API key any non-empty string. |
| text-generation-inference | HuggingFace TGI exposes an OpenAI-compat endpoint. |
| LM Studio | Local LLM GUI with a built-in OpenAI-compat server. |
| LocalAI | OpenAI-compat for whatever model you've loaded. |

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(
    openai.OpenAI(api_key="not-needed", base_url="http://localhost:11434/v1"),
)
client.chat.completions.create(model="llama3.3:70b", messages=[...])
```

### Self-hosted cost-counter caveat

TokenSentinel's waste signals (tool-loop, context bloat, embedding waste, zombie agent, model misroute, retry storm, tool-definition bloat, retrieval thrash) are computed from token counts and call patterns and remain fully accurate against self-hosted endpoints.

The dollar-burn estimate (`LeakEvent.estimated_burn`) assumes priced public-API usage and **is not meaningful for self-hosted deployments**. For vLLM / Ollama / TGI / LM Studio, treat the burn estimate as zero and read waste signals as **quality signals** about your agent's behavior, not as cost figures. If you want a meaningful internal cost, plug your own GPU-amortized $/token into the event before logging it.

---

## What does NOT work transparently

The OpenAI-compatible path covers Chat Completions and Embeddings. Provider-specific endpoints that don't mirror the OpenAI surface need a dedicated wrapper:

- Anthropic native `messages.create` — handled by `wrap_anthropic`.
- Google Gemini `generate_content` — handled by `wrap_gemini`.
- AWS Bedrock `converse` / `invoke_model` — handled by `wrap_bedrock`.
- Cohere native `chat` / `embed` — Cohere's OpenAI-compat surface is incomplete as of May 2026. A native wrapper is deferred until demand justifies it; open an issue if you want this.
- Reranker / safety / classification endpoints across providers — not currently instrumented.

If you need a wrapper that doesn't exist yet, open an issue — we triage these against the wedge of "how much of the agent traffic in the wild flows through this surface".

---

## Multi-provider in one app

A single `Sentinel` instance handles wrapped clients from any number of providers. The tracer keys events by `session_id`, so cross-provider sessions work transparently:

```python
import anthropic, openai
import boto3
from token_sentinel import Sentinel

sentinel = Sentinel(project="multi-provider-agent")

anthropic_client = sentinel.wrap(anthropic.Anthropic())
openai_client = sentinel.wrap(openai.OpenAI())
bedrock_client = sentinel.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))

# All three feed the same tracer / rules engine:
session = "user-task-1"
anthropic_client.messages.create(model="claude-sonnet-4-6", _sentinel_session_id=session, ...)
openai_client.chat.completions.create(model="gpt-5", extra_body={"_sentinel_session_id": session}, ...)
```

Note: for OpenAI-compatible providers, `_sentinel_session_id` is intercepted as a top-level kwarg before it reaches the SDK — pass it the same way as Anthropic.

---

## Verifying your provider

The fastest way to confirm a provider is correctly wrapped is to instantiate the client, wrap it, and inspect the wrapped methods:

```python
import anthropic
from token_sentinel import Sentinel

client = anthropic.Anthropic(api_key="test")
sentinel = Sentinel(project="check")
wrapped = sentinel.wrap(client)

# The wrapped method should have the original wrapped under functools.wraps
print(wrapped.messages.create.__wrapped__)  # <bound method Messages.create of ...>
```

If the wrap call raises `TypeError: Unsupported client type: ... from ...`, the dispatcher didn't recognize the client. Most often this is because:

- The SDK was installed but not the right version. Upgrade to the version listed in [Installation](./01-installation.md).
- You are passing a subclass (e.g., a custom enterprise wrapper). The dispatcher checks `type(client).__module__` — subclasses in your own modules won't match.
- You are passing the module instead of an instance: `sentinel.wrap(anthropic)` instead of `sentinel.wrap(anthropic.Anthropic())`.

Now read [Integrations](./06-integrations.md) for how to use these clients inside MCP, RAG, and orchestration frameworks.
