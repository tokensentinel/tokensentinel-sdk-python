# Providers

TokenSentinel ships first-class wrappers for nine native provider families and works transparently with any OpenAI-compatible chat/embeddings endpoint. This page is the canonical matrix for the open-source SDK.

## Choose your provider

| Your SDK / endpoint | Install extra | Streaming | Async | Notes |
|---|---|---|---|---|
| `anthropic.Anthropic` / `AsyncAnthropic` | `[anthropic]` | yes | yes | `messages.create`, `messages.stream` |
| `openai.OpenAI` / `AsyncOpenAI` | `[openai]` | yes | yes | Chat, embeddings, Whisper STT |
| OpenAI-compatible `base_url` (DeepSeek, Together, Groq, vLLM, Ollama, …) | `[openai]` | yes | yes | Same wrapper; `provider` field still `"openai"` |
| `google.genai.Client` (API or Vertex) | `[gemini]` | yes | yes | Vertex: `vertexai=True` |
| `boto3.client("bedrock-runtime")` | `[bedrock]` | yes | sync only | `converse` / `converse_stream` (not `invoke_model`) |
| `voyageai.Client` / `AsyncClient` | `[voyage]` | n/a | yes | `embed`, `rerank` |
| `cohere.ClientV2` / `AsyncClientV2` | `[cohere]` | limited | yes | `chat`, `embed`, `rerank` — not legacy `cohere.Client` |
| `replicate.Client` | `[replicate]` | n/a | n/a | Non-token pricing via `usage_extra` |
| `deepgram.DeepgramClient` / `AsyncDeepgramClient` | `[deepgram]` | live WS | yes | STT; feeds `audio_multichannel_doubling` |
| `elevenlabs.ElevenLabs` / `AsyncElevenLabs` | `[elevenlabs]` | yes | yes | TTS; feeds `voice_switching_loop` |

How dispatch works: `Sentinel.wrap(client)` inspects `type(client).__module__` (and class name where needed) and routes to the right wrapper. OpenAI-compatible hosts keep `__module__` under `openai...`, so they pick up `wrap_openai` with no special code path.

**Timing (all providers).** Wrappers call the real provider first, then build a `CallRecord` and run rules. Detection and `mode="block"` act **after** the current call is billed; they protect the rest of the session loop.

---

## Native LLM platforms

### Anthropic

```python
import anthropic
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(anthropic.Anthropic())
client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[...])
```

Covers: `messages.create` and `messages.stream` (sync + async). Token counts come from response `usage` or accumulated stream events.

### OpenAI (and OpenAI-compatible)

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(openai.OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
client.embeddings.create(model="text-embedding-3-small", input="...")
```

Covers:

- `chat.completions.create` — sync + async, **including `stream=True`**
- `embeddings.create` — sync + async
- `audio.transcriptions.create` / `audio.translations.create` — Whisper STT (per-second `usage_extra`)

#### Streaming notes (OpenAI)

Streaming is fully instrumented. The wrapper returns a proxy iterator that accumulates chunks and flushes a `CallRecord` when the stream ends (iteration complete, `close()`, or context exit).

1. **Token usage on streams** appears only if you pass  
   `stream_options={"include_usage": True}`. Without it, records may have  
   `prompt_tokens=0`, `completion_tokens=0` and  
   `raw_response_meta["usage_unavailable"] = True`.
2. **`mode="block"`** raises `LeakDetected` when the stream finalizes (same post-call semantics as non-streaming). Prefer `with` / full iteration so exceptions are not suppressed on GC cleanup.
3. A **defensive** `RuntimeWarning` appears only if the stream proxy cannot be constructed (rare; e.g. a non-iterable mock). That is not the normal path.

#### OpenAI-compatible hosts

| Provider | Example `base_url` |
|---|---|
| DeepSeek | `https://api.deepseek.com` |
| Together AI | `https://api.together.xyz/v1` |
| Fireworks | `https://api.fireworks.ai/inference/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Anyscale | `https://api.endpoints.anyscale.com/v1` |
| Mistral La Plateforme | `https://api.mistral.ai/v1` |
| Perplexity | `https://api.perplexity.ai` |
| vLLM / Ollama / TGI / LM Studio / LocalAI | your server `/v1` |

```python
client = sentinel.wrap(
    openai.OpenAI(api_key="...", base_url="https://api.deepseek.com"),
)
```

`CallRecord.provider` stays `"openai"`; the real model id is in `record.model`. For self-hosted, treat `estimated_burn` as a quality signal, not a billing number.

### Google Gemini (and Vertex)

```python
from google import genai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(genai.Client(api_key="..."))
# Vertex:
# client = sentinel.wrap(genai.Client(vertexai=True, project="...", location="us-central1"))
client.models.generate_content(model="gemini-2.5-pro", contents="...")
```

Covers sync + async generate and stream. Surfaces `prompt_tokens_details` for the Gemini-only `vision_cost_concentration` rule when the API provides them.

### AWS Bedrock

```python
import boto3
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))
client.converse(modelId="anthropic.claude-sonnet-4-5-v2:0", messages=[...])
```

Covers `converse` and `converse_stream` (sync). `aioboto3` and low-level `invoke_model*` are not instrumented — use Converse.

---

## Embeddings, rerank, media

### Voyage AI

```python
import voyageai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(voyageai.Client())
client.embed(texts=["..."], model="voyage-3")
```

Feeds `embedding_waste` via `method="embed"` and `raw_request["input"]`.

### Cohere V2

```python
import cohere
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(cohere.ClientV2())
client.chat(model="command-r-plus", messages=[...])
client.embed(texts=["..."], model="embed-english-v3.0")
client.rerank(model="rerank-english-v3.0", query="...", documents=[...])
```

- **Legacy `cohere.Client` (V1) is not wrapped** — use `ClientV2` / `AsyncClientV2`.
- Chat `raw_request` **redacts message bodies** (privacy). Rules that need full text (`repair_loop`, `model_misroute` keywords) will not fire on Cohere chat; `retry_storm` and `rerank_thrash` still work.
- Rerank thrashing is detected by `rerank_thrash`.

### Replicate

```python
import replicate
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(replicate.Client())
```

Uses non-token `usage_extra` dimensions (per-image / per-second / …). Token-slope rules may not apply; burn estimates follow the non-token path.

### Deepgram (STT)

```python
from deepgram import DeepgramClient
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(DeepgramClient())
```

Populates multichannel metadata used by `audio_multichannel_doubling`.

### ElevenLabs (TTS)

```python
from elevenlabs.client import ElevenLabs
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(ElevenLabs())
```

Populates `voice_id` + `text_hash` for `voice_switching_loop`.

---

## Multi-provider in one app

One `Sentinel` can wrap many clients. Share a stable `_sentinel_session_id` so multi-turn rules see a single session:

```python
import anthropic
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="multi-provider-agent")
anthropic_client = sentinel.wrap(anthropic.Anthropic())
openai_client = sentinel.wrap(openai.OpenAI())

session = "user-task-1"
anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[...],
    _sentinel_session_id=session,
)
openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    _sentinel_session_id=session,
)
```

`_sentinel_session_id` is always a **top-level kwarg** stripped before the provider SDK sees it (including OpenAI-compatible clients).

Optional chargeback tags:

```python
growth = sentinel.session(session_id="user-task-1", tags={"team": "growth", "feature": "chat"})
# Pass growth.session_id as _sentinel_session_id on wrapped calls.
# Tags auto-apply when you use Session.record_call(); wrappers currently
# stamp session_id only — set CallRecord.tags yourself if you build records by hand.
```

---

## What is not instrumented

- Legacy Vertex `vertexai` SDK (use `google.genai` with `vertexai=True`)
- Cohere V1 `Client`
- Bedrock `invoke_model` / `invoke_model_with_response_stream`
- Provider-specific endpoints that are not on the surfaces above

For custom stacks, build a `CallRecord` and call `sentinel.record_call(...)` (see [API reference](./07-api-reference.md)).

## Optional cloud (paid)

Wrapping and rules never require cloud. If you configure `cloud_endpoint` + `api_key`, events are batched to the closed-source TokenSentinel Cloud for dashboards, retention, webhooks, and (on Team+) policy enforcement. Details and pricing live on [tokensentinel.dev](https://tokensentinel.dev) — not in this open-source package.

Next: [Integrations](./06-integrations.md) for MCP, RAG, LangChain, and OTel.
