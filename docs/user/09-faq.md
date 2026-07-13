# FAQ

> **Documented for Python SDK `token-sentinel` 1.0.2.**

## Open source SDK vs paid cloud

| | **Open-source SDK** (`token-sentinel`) | **TokenSentinel Cloud** (separate product) |
|---|---|---|
| License | Apache-2.0 | proprietary |
| What it is | In-process rules + provider wrappers | Hosted API, dashboard, policy, Pro features |
| Cost | Free forever | Paid tiers (Team / Pro / Enterprise) |
| Network | Nothing leaves the process by default | Only if you set `cloud_endpoint` + `api_key` |
| Where docs live | This repo (`docs/user/`) | [tokensentinel.dev](https://tokensentinel.dev) / product site |

**Strategy in one line:** detection logic is open so teams can audit and self-host the agent-side library; dashboards, multi-tenant policy, retention, and calibration are the commercial cloud.

### What the free SDK includes

- All **15** deterministic waste rules (chat/agent, vision, audio, voice, rerank, repair)
- Native wrappers: Anthropic, OpenAI (+ compatible), Gemini/Vertex, Bedrock, Voyage, Cohere V2, Replicate, Deepgram, ElevenLabs
- Modes `log` / `alert` / `block`
- Optional LangChain + OpenTelemetry enrichers
- Zero-dependency core; failure-isolated instrumentation

### What paid cloud adds (not in this package)

Typical Team+ / Pro surfaces (subject to current product pricing):

- Event ingestion, retention, hosted dashboard, webhooks (email / Slack / PagerDuty / HTTP)
- **Intervention Pack:** session USD budgets, tokens/min velocity caps, operator kill-switch (SDK polls policy when configured)
- **Pro:** LLM-as-judge on gray-zone confidences, drift / stability index, trace consolidation, composite signals (`lost_agent`, `runaway_retrieval`, `zombie_loop`), RBAC / audit, chargeback by tags

Pricing and the full feature matrix are product-site concerns — not re-litigated in this OSS tree. Start at [tokensentinel.dev](https://tokensentinel.dev).

```python
# OSS-only (default)
Sentinel(project="my-agent", mode="log")

# Opt into paid cloud (any mode ships events when both are set)
Sentinel(
    project="my-agent",
    mode="log",
    cloud_endpoint="https://api.tokensentinel.dev",
    api_key="ts_live_...",
)
```

## Why not OpenTelemetry alone?

OTel spans are observational and usually fire after the call. TokenSentinel sits on the client call path so your handler (or `block` exception) can stop **the next** call. We also ship an **OTel span processor enricher** for frameworks that only emit `gen_ai.*` spans — that path is still post-call observation, not a substitute for wrap when you need tight control.

## How is this different from Langfuse / LangSmith / Helicone / Datadog LLM?

Those products excel at traces, analytics, and cost reporting **after** the fact. TokenSentinel is in-process **detection** (and optional mid-session halt) for waste patterns. They compose: wrap with TokenSentinel and keep your observability stack.

## Does it work with self-hosted LLMs?

Yes — OpenAI-compatible servers (vLLM, Ollama, TGI, LM Studio, LocalAI). Point `openai.OpenAI(base_url=...)` and wrap. Waste **signals** still apply; `estimated_burn` is not a real invoice for GPU-amortized tokens.

## Timing: does it stop spend on the current call?

No. Wrappers call the provider first, then rules / `block` / policy. The current request is already billed. Value is stopping runaway **loops** and giving operators a signal in real time. Documented in [Modes](./03-modes.md).

## Is there a TypeScript or Go SDK?

Not yet. Python is first. Open an issue if you need another language for a production use case.

## How do I contribute?

- Bugs / features: GitHub Issues on this repository
- PRs: welcome; non-trivial changes should start with an issue
- Tests: `pytest tests/ -q`
- Lint: `ruff check token_sentinel tests examples`
- Types: `mypy token_sentinel`

New providers: follow `wrappers/anthropic.py` patterns. New rules: deterministic, no I/O, pure function of the session buffer.

## License

Apache-2.0 for the SDK. Use it in commercial products. Cloud remains proprietary.

## Why “TokenSentinel”?

“Sentinel” = watching the call path. Marketing speaks “token waste” (cost); the Python API keeps `LeakEvent` / `on_leak` for compatibility and adds `WasteEvent` / `on_waste` aliases.

## Does `alert` mode send data to the cloud?

Only if you configure `cloud_endpoint` and `api_key`. Those flags enable the sink for **all** modes, including `log`. Without them, `alert` is local-only like `log`.

## Policy exceptions without `mode="block"`?

Yes. `BudgetExceeded`, `VelocityExceeded`, and `KillSwitchActive` are raised from the policy client when cloud policy is active, independent of mode. They require a paid cloud policy plane.

## Version

```bash
python -c "import token_sentinel; print(token_sentinel.__version__)"
```
