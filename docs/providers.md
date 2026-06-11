# Provider Matrix

TokenSentinel ships first-class wrappers for the four major LLM platforms and works **transparently** through the OpenAI wrapper for anything that exposes an OpenAI-compatible REST surface. This page is the canonical "what's supported" reference for V0.

## How dispatch works

`Sentinel.wrap(client)` looks at `type(client).__module__` and routes to the right wrapper. Because OpenAI-compatible providers are accessed through the official `openai` SDK with a custom `base_url`, they all keep `__module__ == "openai.lib..."` and pick up the OpenAI wrapper for free — no provider-specific code path required.

## Native wrappers

| Provider      | SDK            | Wrapper           | Streaming        | Async         | Notes                                    |
| ------------- | -------------- | ----------------- | ---------------- | ------------- | ---------------------------------------- |
| Anthropic     | `anthropic`    | `wrap_anthropic`  | Yes              | Yes           | `messages.create` + `messages.stream`    |
| OpenAI        | `openai`       | `wrap_openai`     | (passthrough)    | Yes           | `chat.completions` + `embeddings`. Streaming passthrough today; full stream instrumentation tracked for an upcoming release. |
| Google Gemini | `google-genai` | `wrap_gemini`     | Yes              | Yes           | Covers Vertex AI via `vertexai=True`     |
| AWS Bedrock   | `boto3`        | `wrap_bedrock`    | Yes              | (sync only)   | `converse` + `converse_stream`           |

```python
import anthropic, openai, boto3
from google import genai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
anthropic_client = sentinel.wrap(anthropic.Anthropic())
openai_client    = sentinel.wrap(openai.OpenAI())
gemini_client    = sentinel.wrap(genai.Client())
bedrock_client   = sentinel.wrap(boto3.client("bedrock-runtime"))
```

Streaming for OpenAI is currently passthrough (no record emitted) — full stream instrumentation is tracked for an upcoming release. Anthropic, Gemini, and Bedrock streaming are all fully instrumented today. Bedrock async support waits on stable `aioboto3` adoption.

## OpenAI-compatible (transparent through `wrap_openai`)

These providers all expose the OpenAI Chat Completions / Embeddings API shape. You instantiate the official `openai` SDK with a custom `base_url`, and `Sentinel.wrap` routes to `wrap_openai` because `type(client).__module__` still starts with `openai`. No additional configuration on the TokenSentinel side.

| Provider               | `base_url`                                | Notes                                  |
| ---------------------- | ----------------------------------------- | -------------------------------------- |
| DeepSeek               | `https://api.deepseek.com`                | DeepSeek-V3 (`deepseek-chat`), R1 (`deepseek-reasoner`) |
| Together AI            | `https://api.together.xyz/v1`             | Llama, Qwen, DeepSeek, Mixtral hosted  |
| Fireworks              | `https://api.fireworks.ai/inference/v1`   | Llama, Qwen, FireFunction              |
| Groq                   | `https://api.groq.com/openai/v1`          | Cheap, ultra-fast inference            |
| OpenRouter             | `https://openrouter.ai/api/v1`            | Multi-model gateway across providers   |
| Anyscale               | `https://api.endpoints.anyscale.com/v1`   | Open models on Anyscale Endpoints      |
| Mistral La Plateforme  | `https://api.mistral.ai/v1`               | Mistral Small / Large / Codestral      |
| Perplexity             | `https://api.perplexity.ai`               | Sonar models with web grounding        |

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(
    openai.OpenAI(api_key="...", base_url="https://api.deepseek.com"),
)
```

The same four lines work for any row in the table — swap `api_key` and `base_url`. CallRecords show `provider="openai"` because the wrapper can't tell DeepSeek from OpenAI from the SDK type alone; the actual model lands in `record.model` (e.g., `deepseek-chat`, `llama-3.3-70b`, `mixtral-8x7b`).

## Self-hosted (transparent through `wrap_openai`)

Open-source LLM servers that ship an OpenAI-compatible endpoint also pick up the wrapper for free. Point the `base_url` at your server.

| Stack                       | Notes                                                            |
| --------------------------- | ---------------------------------------------------------------- |
| vLLM                        | Native OpenAI-compat. Set `base_url='http://your-vllm:8000/v1'`. |
| Ollama                      | Set `base_url='http://localhost:11434/v1'`.                      |
| text-generation-inference   | HuggingFace TGI exposes an OpenAI-compat endpoint.               |
| LM Studio                   | Local LLM GUI with a built-in OpenAI-compat server.              |

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(
    openai.OpenAI(api_key="not-needed", base_url="http://localhost:11434/v1"),
)
```

> **Self-hosted footnote — cost-counter caveat.** TokenSentinel's waste signals (tool-loop, context bloat, embedding waste, zombie agent, model misroute, retry storm, tool-definition bloat, retrieval thrash) are computed from token counts and call patterns and remain fully accurate against self-hosted endpoints. The dollar-burn estimate (`LeakEvent.estimated_burn`) assumes priced public-API usage and is **not meaningful for self-hosted deployments**. For vLLM / Ollama / TGI / LM Studio, treat the burn estimate as zero and read the waste signals as **quality signals** about your agent's behavior, not as cost figures.

## What does **not** work transparently

The OpenAI-compatible path covers the Chat Completions and Embeddings shapes. Provider-specific endpoints that do **not** mirror the OpenAI surface are out of scope for `wrap_openai` and need a dedicated wrapper:

- Anthropic's native `messages.create` (handled by `wrap_anthropic`).
- Google Gemini's `generate_content` (handled by `wrap_gemini`).
- AWS Bedrock's `converse` / `invoke_model` (handled by `wrap_bedrock`).
- Cohere's native `chat` / `embed` endpoints (Cohere's OpenAI-compat surface is incomplete as of May 2026; native wrapper deferred until demand justifies it — open an issue if you want this).
- Reranker / safety / classification endpoints across providers.

## Verifying coverage in tests

`tests/test_openai_compat.py` parameterises a handful of OpenAI-compatible providers (DeepSeek, Together, Fireworks, Groq) and asserts `wrap_openai` instruments each one identically — same `CallRecord` shape, same provider="openai", same per-call hashing. The test deliberately uses mock clients with the right `base_url` configured; no real API calls.
