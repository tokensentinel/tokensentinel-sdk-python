# Provider Matrix

> **Documented for Python SDK `token-sentinel` 1.0.2.**

Canonical “what’s supported” reference for the open-source SDK. For narrative install + examples, prefer [docs/user/05-providers.md](user/05-providers.md).

## How dispatch works

`Sentinel.wrap(client)` inspects `type(client).__module__` (and class name where needed) and routes to a wrapper. OpenAI-compatible hosts use the official `openai` SDK with a custom `base_url`, so they keep an `openai…` module prefix and pick up `wrap_openai` for free.

## Native wrappers

| Provider | SDK | Streaming | Async | Notes |
|---|---|---|---|---|
| Anthropic | `anthropic` | yes | yes | `messages.create` + `messages.stream` |
| OpenAI | `openai` | yes | yes | Chat, embeddings, Whisper STT. Streams need `stream_options.include_usage` for tokens |
| Google Gemini | `google-genai` | yes | yes | Vertex via `vertexai=True` |
| AWS Bedrock | `boto3` | yes | sync only | `converse` / `converse_stream` |
| Voyage AI | `voyageai` | n/a | yes | `embed`, `rerank` |
| Cohere V2 | `cohere` | limited | yes | `ClientV2` / `AsyncClientV2` only |
| Replicate | `replicate` | n/a | n/a | Non-token `usage_extra` |
| Deepgram | `deepgram-sdk` | live WS | yes | STT; multichannel rule |
| ElevenLabs | `elevenlabs` | yes | yes | TTS; voice-switching rule |

## OpenAI-compatible (via `wrap_openai`)

DeepSeek · Together · Fireworks · Groq · OpenRouter · Anyscale · Mistral · Perplexity · vLLM · Ollama · TGI · LM Studio · LocalAI

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(
    openai.OpenAI(api_key="...", base_url="https://api.deepseek.com"),
)
```

`CallRecord.provider` is `"openai"`; the real model string is in `record.model`.

## Self-hosted

Same OpenAI-compatible path. Waste **signals** are real; `estimated_burn` is not a GPU invoice — treat it as a relative quality metric.

## Not instrumented

- Legacy Vertex `vertexai` SDK (use `google.genai` + `vertexai=True`)
- Cohere V1 `Client`
- Bedrock `invoke_model*`

Custom stacks: build `CallRecord` and call `Sentinel.record_call`.
