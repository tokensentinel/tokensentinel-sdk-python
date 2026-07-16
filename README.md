# token-sentinel

Predictive **token-waste detection** for AI agents — a Python SDK that runs
**in-process**, watches LLM calls after each provider response, and fires a
typed callback so you can log, alert, or hard-stop the next turn.

Observability tools show the bill after the fact. TokenSentinel names the waste
pattern **while the session is still running**. Detection is post-call (the
turn that just finished is already billed); intervention saves **subsequent**
turns.

**Docs:** [https://docs.tokensentinel.dev](https://docs.tokensentinel.dev)

## Install

```bash
pip install token-sentinel[anthropic]   # or [openai], [gemini], …
# optional: local token estimates when usage is missing
pip install token-sentinel[tiktoken]
```

## Quick start

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")  # log | alert | block

@sentinel.on_waste  # or @sentinel.on_leak — same callback
def handle(event):
    print(f"{event.type} conf={event.confidence:.2f} burn≈${event.estimated_burn:.4f}")

client = sentinel.wrap(anthropic.Anthropic())
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Tune rule thresholds

Every rule exposes flat config keys `"rule_name.param"`. Example — require five
similar tool calls (default is **3**) before `tool_loop` fires:

```python
sentinel = Sentinel(
    project="my-agent",
    mode="log",
    config={
        "tool_loop.min_calls": 5,
        "tool_loop.cosine_threshold": 0.80,
        "tool_loop.window_seconds": 120,
    },
)
```

Full parameter tables: [docs/user/04-waste-rules.md](docs/user/04-waste-rules.md).

## What it catches (15 rules)

| Rule | Signal |
|---|---|
| **tool_loop** | Same tool, ≥N cosine-similar calls in a window |
| **context_bloat** | Prompt-tokens-per-turn slope rising |
| **embedding_waste** | Same embedding lookup repeated in session |
| **zombie** | Calls continue with no user-facing output |
| **model_misroute** | Classification-shaped prompt on a frontier model |
| **retry_storm** | Same call retried many times unchanged |
| **tool_definition_bloat** | Huge tool JSON / MCP tool lists |
| **retrieval_thrash** | Overlapping retrieval queries |
| **vision_re_upload** | Same image re-uploaded across turns |
| **vision_high_detail_misroute** | High-detail flag on low-detail work |
| **vision_cost_concentration** | Vision spend concentrated in few sessions |
| **audio_multichannel_doubling** | Multichannel STT billing trap |
| **voice_switching_loop** | Same text, many voice IDs |
| **rerank_thrash** | Duplicate Cohere rerank requests |
| **repair_loop** | Correction churn + similar regenerations |

## Providers

Native wrappers (`pip install token-sentinel[<extra>]`):

| Provider | Extra | Streaming |
|---|---|---|
| Anthropic | `anthropic` | yes |
| OpenAI (+ Whisper) | `openai` | yes |
| Google Gemini / Vertex | `gemini` | yes |
| AWS Bedrock | `bedrock` | yes |
| Voyage, Cohere, Replicate, Deepgram, ElevenLabs | matching extra | varies |

**OpenAI-compatible hosts** (same wrapper, set `base_url`): DeepSeek, Together,
Fireworks, Groq, OpenRouter, Mistral, Perplexity, vLLM, Ollama, TGI, LM Studio,
xAI Grok (`https://api.x.ai/v1`), etc.

```python
import openai
client = sentinel.wrap(
    openai.OpenAI(api_key="…", base_url="https://api.deepseek.com")
)
```

Pass a stable `_sentinel_session_id=...` on each call (or use
`Sentinel.session(...)`) so multi-turn rules see history.

## Modes

| Mode | Behavior |
|---|---|
| `log` | Emit events to your handler. Default. Safe for prod day one. |
| `alert` | Same local behavior as `log` (mode stamp for optional cloud). |
| `block` | Raise `LeakDetected` / `WasteDetected` **after** the provider returns. |

## Cost estimates (`estimated_burn`)

Burn figures are **approximate FinOps signals**, not invoices:

- **Model-aware rates** for major Anthropic / OpenAI / Gemini / DeepSeek / Cohere / Mistral families (input vs output priced separately).
- **Prompt-cache reads** (OpenAI `cached_tokens`, Anthropic `cache_read_input_tokens`) discounted when present on the usage payload.
- **Unknown models** fall back to a flat average (~$9e-6 / token).
- Optional **`tiktoken`** fills missing token counts when usage was omitted.

Override rates with `Sentinel(pricing_table={...})` or
`from token_sentinel import estimate_usd, ModelRate, default_pricing_table`.

## Frameworks

Instruments at the LLM-client layer, so MCP hosts, RAG pipelines, LangChain /
LangGraph / CrewAI / AutoGen / Pydantic AI work when traffic bottoms out in a
wrapped client. Enrichers: `pip install token-sentinel[langchain]` or `[otel]`.

## Optional cloud

Nothing phones home unless you pass both `cloud_endpoint=` and `api_key=` on
`Sentinel(...)`. For hosted dashboard, Composite Signals and policy features — see [tokensentinel.dev](https://tokensentinel.dev). This package is
**Apache-2.0** and fully usable offline.

## Status

**1.0.3** — 15 rules, model-aware burn estimates, cache-aware usage, optional
tiktoken fallback, OSS-focused docs.

Public API (`Sentinel`, `wrap`, `on_leak` / `on_waste`, `record_call`, `session`,
`mark_long_running`, events/exceptions, pricing helpers) follows semver —
pin deliberately (`token-sentinel>=1.0,<2`).

## More docs

- [User guide](docs/user/) — install, modes, rules, providers, API reference  
- [Architecture](docs/architecture.md) · [Waste taxonomy](docs/waste-taxonomy.md)  
- [CHANGELOG](CHANGELOG.md)

## Contact

[support@tokensentinel.dev](mailto:support@tokensentinel.dev) ·
[hello@tokensentinel.dev](mailto:hello@tokensentinel.dev) ·
[tokensentinel.dev](https://tokensentinel.dev)

## License

Apache-2.0 — see [LICENSE](LICENSE).
