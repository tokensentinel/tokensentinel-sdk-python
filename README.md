# token-sentinel

Predictive token-waste detection for AI agents.

A Python SDK that catches token waste *mid-run* — while a session is still active — and gives your app a callback to log, alert, or hard-stop the agent before the *next* call goes out. Detection runs after each provider response returns (that call is already billed); intervention saves subsequent turns. Apache-2.0 licensed, zero-dependency core. Pair the SDK with the optional TokenSentinel Cloud for hosted dashboards, budget enforcement, drift detection, and judge ratification on Pro.

Existing observability tools (Langfuse, LangSmith, Helicone, Datadog LLM) tell you what your bill was. TokenSentinel tells you which agent is leaking *right now*.

## What it catches

Fifteen deterministic rules, all in-process, sub-millisecond per rule:

| Leak / Waste | Signal |
|---|---|
| **Tool-loop** | Same tool, ≥3 cosine-similar calls in a window |
| **Context bloat** | Prompt-tokens-per-turn slope rising past threshold |
| **Embedding waste** | Same embedding lookup repeated within session |
| **Zombie agent** | No user-facing output for N min, calls still firing |
| **Model misroute** | Classification-shaped prompt sent to a frontier model |
| **Retry storm** | Same call retried >N times without parameter change |
| **Tool-definition bloat** | A single request ships ≥30 tool defs or ≥30KB of tool JSON (the MCP problem) |
| **Retrieval thrash** | Retrieval tool called repeatedly with overlapping queries (the RAG problem) |
| **Vision re-upload** | Same image (SHA-256 or perceptual hash) uploaded repeatedly across turns |
| **Vision detail misroute** | High-detail vision flag on low-detail-suitable images (e.g. icons, low-res) |
| **Vision concentration** | Visual tokens heavily concentrated in a single/few outlier sessions |
| **Audio channel doubling** | Stereo/multichannel audio transcription when mono-channel would suffice |
| **Voice switching loop** | Rapid shifting of ElevenLabs voice IDs on identical text payloads |
| **Rerank thrash** | Cohere rerank API requests repeated for identical search lists |
| **Repair loop** | Conversational loop with repeated user corrections and similar agent regenerations |

**Composite signals (Pro tier, cloud-side)**

| Composite | Fires when |
|---|---|
| **lost_agent** | `tool_loop` + `context_bloat` + `model_misroute` all hit on the same session inside a 30s window |
| **runaway_retrieval** | `retrieval_thrash` + `embedding_waste` co-fire while the per-turn token slope is still climbing |
| **zombie_loop** | `zombie` + `retry_storm` co-fire on a session with no user-facing output |

## Supported providers

Native wrappers — `pip install token-sentinel[<provider>]`:

| Provider | SDK | Streaming | Async |
|---|---|---|---|
| Anthropic | `anthropic` | yes | yes |
| OpenAI | `openai` | yes¹ | yes |
| Google Gemini | `google-genai` | yes | yes |
| AWS Bedrock | `boto3` | yes | sync only |

¹ OpenAI streaming is fully instrumented; pass `stream_options={"include_usage": True}` for token counts on streams.

Transparent through the OpenAI wrapper (just set `base_url`):

DeepSeek · Together AI · Fireworks · Groq · OpenRouter · Anyscale · Mistral La Plateforme · Perplexity · vLLM · Ollama · text-generation-inference · LM Studio

Google Vertex AI is reached via the same Gemini wrapper by passing `vertexai=True` to `genai.Client(...)`.

See [docs/providers.md](docs/providers.md) for the full matrix and per-provider snippets.

## Quick start

```bash
pip install token-sentinel[anthropic]
```

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")  # log | alert | block

@sentinel.on_leak
def handle(event):
    print(f"LEAK [{event.type}] confidence={event.confidence:.2f} burn=${event.estimated_burn:.4f}")

client = sentinel.wrap(anthropic.Anthropic())
# use the client normally — Sentinel watches in-process
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Switch providers by installing the right extra and changing one line:

```python
# DeepSeek (or any OpenAI-compatible endpoint)
import openai
client = sentinel.wrap(openai.OpenAI(base_url="https://api.deepseek.com"))

# Google Gemini
from google import genai
client = sentinel.wrap(genai.Client())

# AWS Bedrock
import boto3
client = sentinel.wrap(boto3.client("bedrock-runtime"))
```

Per-provider deep dives — install, wrap, leak, stream, async, production:

- [Anthropic quickstart](docs/quickstart-anthropic.md)
- [OpenAI quickstart](docs/quickstart-openai.md) (covers DeepSeek, Together, Groq, vLLM, Ollama, …)
- [Gemini quickstart](docs/quickstart-gemini.md) (covers Vertex AI)
- [Bedrock quickstart](docs/quickstart-bedrock.md)

## Modes

| Mode | Behavior |
|---|---|
| `log` | Emit events to your handler. Default. Safe for production from day one. |
| `alert` | Same handler behavior as `log` (cloud is separate — see below). |
| `block` | Raise `LeakDetected` after the provider returns (halts the agent loop). Opt-in. |

Cloud event shipping requires `cloud_endpoint` **and** `api_key` and works in **any** mode. Without those kwargs, nothing leaves the process.

## Works with MCP, RAG, and orchestration frameworks

TokenSentinel instruments at the LLM-client layer, so it transparently catches traffic from MCP hosts, RAG pipelines, and orchestration frameworks (LangChain, LangGraph, CrewAI, AutoGen, Pydantic AI). See [docs/integration-patterns.md](docs/integration-patterns.md).

## Cloud (optional, paid, closed source)

This repository is the **Apache-2.0 SDK only**. The commercial product is separate:

| Free (this package) | Paid TokenSentinel Cloud |
|---|---|
| 15 rules, 9 providers, log/alert/block | Hosted dashboard, retention, webhooks |
| Zero network by default | Intervention Pack: budgets, velocity, kill-switch |
| Self-host friendly | Pro: judge, drift, composites, RBAC, chargeback |

Configure cloud via `cloud_endpoint=` and `api_key=` on `Sentinel(...)`. Tier comparison and pricing: [tokensentinel.dev](https://tokensentinel.dev). Full user guide: [docs/user/README.md](docs/user/README.md).

## Migrate from Helicone / Langfuse / LangSmith

The `tokensentinel-migrate` companion package replays your existing trace history through the rules and backfills events into your TokenSentinel cloud project. See the [tokensentinel-migrate package on PyPI](https://pypi.org/project/tokensentinel-migrate/).

```bash
pip install tokensentinel-migrate
python -m tokensentinel_migrate helicone --helicone-api-key sk-... --tokensentinel-endpoint https://... --tokensentinel-api-key tsk_... --project my-agent --since 2026-04-09 --dry-run
```

> **Self-hosted note.** vLLM / Ollama / TGI all expose OpenAI-compatible endpoints, so TokenSentinel works against them out of the box. Leak signals are real, but the dollar burn estimate assumes priced API usage — for self-hosted, treat the burn estimate as a quality signal, not a billing signal.

## Status

**Stable Release** — 15 deterministic rules, 9 native providers (Anthropic, OpenAI, Gemini, Bedrock, Voyage, Cohere, Replicate, Deepgram, ElevenLabs), streaming + async, and full integration with the optional TokenSentinel Cloud policy engine.

**Tests**: 912 SDK tests passing. Codebase is clean of ruff, mypy, and typecheck warnings.

The public API surface (`Sentinel`, `wrap`, `on_leak` / `on_waste`, `record_call`, `session`, `LeakEvent` / `WasteEvent`, `CallRecord`, `LeakDetected` / `WasteDetected`, plus policy exceptions `BudgetExceeded`, `VelocityExceeded`, `KillSwitchActive`) is stable and follows semver — pin deliberately (e.g. `token-sentinel>=1.0,<2`).

## Architecture

- **SDK (this package)** — Python wrapper around all major LLM clients. Apache-2.0 licensed.
- **Optional cloud dashboard** — closed-source, hosted at `api.tokensentinel.dev`. Provides retention, dashboards, the Intervention Pack policy plane, the LLM-as-judge ratification pipeline, drift / stability scoring, RBAC + audit, and multi-environment routing. The SDK works perfectly without it; nothing phones home unless you explicitly configure `cloud_endpoint` and `api_key`.

rule detection runs entirely in-process. The composite rules and judge ratification run cloud-side on top of the same `LeakEvent` stream. Cloud is opt-in for retention, dashboards, team features, and the chargeback attribution coming in V2.

## Docs

User-facing docs (published with the OSS SDK):

- [User Guide](docs/user/) — installation, quickstart, modes, leak rules, providers, integrations, API reference
- [Architecture](docs/architecture.md) — how the wrapper, tracer, and rules engine fit together
- [Leak taxonomy](docs/waste-taxonomy.md) — the rules in detail with thresholds and false-positive hazards
- [Providers](docs/providers.md) — full matrix of supported providers
- [Integration patterns](docs/integration-patterns.md) — MCP, RAG, LangChain, LangGraph, CrewAI, AutoGen, Pydantic AI
- [Changelog](CHANGELOG.md)

## Contact & Support

For support, feedback, or inquiries, please contact shakyasmreta@gmail.com or visit our official website at [tokensentinel.dev](https://tokensentinel.dev).

## License

Apache-2.0 — see [LICENSE](LICENSE). The patent grant in Apache-2.0 is the right OSS contract for an SDK that runs inline against enterprise customers' production AI calls.
