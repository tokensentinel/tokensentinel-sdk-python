# Installation

> **Documented for Python SDK `token-sentinel` 1.0.3** (PyPI package name: `token-sentinel`).

TokenSentinel is published to PyPI as `token-sentinel`. The **core package has zero runtime dependencies** — provider SDKs and optional features are pulled in via extras so you only install what you use.

## Python version

TokenSentinel requires **Python 3.10 or later**. It is tested on 3.10, 3.11, and 3.12.

## Core install

```bash
pip install token-sentinel
```

The core package gives you `Sentinel`, the 15 in-process waste rules, the tracer, model-aware burn helpers (`estimate_usd`, `ModelRate`, …), and the public types (`CallRecord`, `LeakEvent` / `WasteEvent`, `LeakDetected` / `WasteDetected`, plus the optional policy exceptions). It will not instrument any provider client on its own — for that you need at least one provider extra below.

## Provider extras

Install the extras for the providers you call. You can combine them.

| Extra | Pulls in | Use when |
|---|---|---|
| `[anthropic]` | `anthropic>=0.39.0` | `anthropic.Anthropic` / `AsyncAnthropic` |
| `[openai]` | `openai>=1.50.0` | `openai.OpenAI` / `AsyncOpenAI`, plus any OpenAI-compatible `base_url` (DeepSeek, Together, Groq, vLLM, Ollama, …). Also covers Whisper STT via `audio.transcriptions` / `audio.translations` |
| `[gemini]` | `google-genai>=1.0.0` | `google.genai.Client` (API key or Vertex via `vertexai=True`) |
| `[bedrock]` | `boto3>=1.35.0` | `boto3.client("bedrock-runtime")` |
| `[voyage]` | `voyageai>=0.2` | `voyageai.Client` / `AsyncClient` (embed + rerank) |
| `[cohere]` | `cohere>=5.0` | `cohere.ClientV2` / `AsyncClientV2` (chat + embed + rerank) |
| `[replicate]` | `replicate>=0.40` | `replicate.Client` (image/video; non-token pricing) |
| `[deepgram]` | `deepgram-sdk>=3.0` | `DeepgramClient` / `AsyncDeepgramClient` (STT) |
| `[elevenlabs]` | `elevenlabs>=1.0` | `ElevenLabs` / `AsyncElevenLabs` (TTS) |

```bash
# Single provider
pip install token-sentinel[anthropic]

# Multi-provider
pip install token-sentinel[anthropic,openai,bedrock]

# Embeddings + STT + TTS stack
pip install token-sentinel[openai,voyage,deepgram,elevenlabs]
```

The `[openai]` extra is enough for every OpenAI-compatible chat/embeddings endpoint — DeepSeek, Together, Fireworks, Groq, OpenRouter, Anyscale, Mistral, Perplexity, vLLM, Ollama, TGI, LM Studio — because they use the official `openai` SDK with a custom `base_url`. See [Providers](./05-providers.md).

## Optional feature extras

| Extra | Pulls in | Use when |
|---|---|---|
| `[tiktoken]` | `tiktoken` | Optional local token counts when a provider omits usage (e.g. OpenAI streams without `stream_options.include_usage`). Improves burn estimates only — not required for rules |
| `[embeddings]` | `sentence-transformers`, `numpy` | Reserved for future semantic similarity on `tool_loop` / `retrieval_thrash`. **Not required today** — rules use pure-Python TF-IDF char-n-grams |
| `[vision-perceptual]` | `imagehash`, `Pillow` | Perceptual-hash fallback for `vision_re_upload` (catches re-uploads after resize/recompress) |
| `[audio-metadata]` | `mutagen` | Better audio-duration probing for Whisper streaming paths |
| `[langchain]` | `langchain-core` | `TokenSentinelCallbackHandler` enricher |
| `[otel]` | OpenTelemetry API/SDK | `TokenSentinelSpanProcessor` for CrewAI / AutoGen / Pydantic AI (and any `gen_ai.*` spans) |
| `[all]` | Everything above (including `tiktoken`) | Convenience meta-extra for multi-provider apps and full feature set |
| `[dev]` | `pytest`, `ruff`, `mypy`, … | Contributing to the SDK itself |

```bash
pip install token-sentinel[tiktoken]
pip install token-sentinel[vision-perceptual]
pip install token-sentinel[langchain]
pip install token-sentinel[all]
```

## Verifying your install

```bash
python -c "from token_sentinel import Sentinel, __version__; print(__version__, Sentinel.__module__)"
```

Expected output starts with `1.0.3` (or your installed version) and `token_sentinel.sentinel`.

To verify a provider extra and wrapper dispatch (no real network call required for wrap itself):

```bash
# Anthropic
python -c "import anthropic; from token_sentinel import Sentinel; Sentinel(project='check').wrap(anthropic.Anthropic(api_key='test')); print('anthropic ok')"

# OpenAI (+ OpenAI-compatible)
python -c "import openai; from token_sentinel import Sentinel; Sentinel(project='check').wrap(openai.OpenAI(api_key='test')); print('openai ok')"

# Gemini
python -c "from google import genai; from token_sentinel import Sentinel; Sentinel(project='check').wrap(genai.Client(api_key='test')); print('gemini ok')"

# Bedrock — needs AWS credentials configured but wrap itself does not call the API
python -c "import boto3; from token_sentinel import Sentinel; Sentinel(project='check').wrap(boto3.client('bedrock-runtime', region_name='us-east-1')); print('bedrock ok')"
```

Each line should print `<provider> ok`. If you see `Unsupported client type`, the dispatcher did not recognize the client — usually an old SDK version or a custom subclass in your own module. See [Troubleshooting](./08-troubleshooting.md).

## Open source vs paid cloud

| Path | What you get | Cost |
|---|---|---|
| **SDK only** (this package) | All 15 rules, all wrappers, `log` / `alert` / `block`, zero phoning home | Free (Apache-2.0) |
| **TokenSentinel Cloud** (Proprietary) | Hosted dashboards, retention, webhooks, Intervention Pack (budgets / velocity / kill-switch), Pro judge + drift + composites | Paid tiers — see [tokensentinel.dev](https://tokensentinel.dev) |

Nothing leaves the process unless you set **both** `cloud_endpoint=` and `api_key=` on `Sentinel(...)`. The SDK is fully functional without cloud.

## Upgrading

```bash
pip install --upgrade token-sentinel
```

The public API (`Sentinel`, `wrap`, `on_leak` / `on_waste`, `record_call`, `session`, `LeakEvent` / `WasteEvent`, `CallRecord`, `LeakDetected` / `WasteDetected`, and the policy exceptions) follows semantic versioning. Modules under `token_sentinel.tracer`, `token_sentinel.rules`, and `token_sentinel.wrappers` are internal — pin a minor version if you import them directly.

## Troubleshooting installation

**`No matching distribution found for token-sentinel`** — upgrade pip or check you are on Python ≥ 3.10.

**`No matching distribution found for token-sentinel[anthropic]`** — extras syntax needs `pip>=20.3`.

**`Sentinel.wrap(...)` raises `Unsupported client type`** — install the matching provider extra, upgrade the provider SDK to the version in the table, and pass a fully constructed client instance (not the module).

Once install works, go to [Quickstart](./02-quickstart.md).
