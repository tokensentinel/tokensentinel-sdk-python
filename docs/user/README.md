# TokenSentinel User Guide

> **Python SDK version: `token-sentinel` 1.0.1**  
> Install: `pip install "token-sentinel>=1.0.1,<2"` · Check: `python -c "import token_sentinel; print(token_sentinel.__version__)"`  
> Changelog: [CHANGELOG.md](https://github.com/tokensentinel/tokensentinel-sdk-python/blob/main/CHANGELOG.md) on GitHub

TokenSentinel is an **open-source Python SDK** that detects token waste in AI agents while a session is still active, and gives your app a callback to log, alert, or hard-stop the agent before the *next* call goes out.

Detection runs **after** each provider response (that call is already billed). Intervention saves subsequent turns. Pair the free SDK with optional **TokenSentinel Cloud** (closed source, paid) for dashboards, policy enforcement, and Pro calibration — nothing phones home unless you set `cloud_endpoint` and `api_key`.

## Who this is for

- Backend engineers running agents in production who want waste signals *during* a run
- Platform teams that need guardrails without shipping a full observability stack first
- OSS and self-hosted users who want rules with zero cloud dependency

If you only want post-hoc cost analytics, tools like Langfuse / LangSmith / Helicone / Datadog LLM may fit better. TokenSentinel sits in the client call path.

## What ships in the free SDK

- **15** deterministic in-process rules (tool loops, context bloat, embedding waste, zombies, misroutes, retries, MCP tool-def bloat, RAG thrash, vision, audio multichannel, voice switching, rerank thrash, repair loops)
- **9** native provider families + OpenAI-compatible endpoints
- Modes: `log` | `alert` | `block`
- Optional LangChain callback + OpenTelemetry span enrichers

## 30-second quickstart

```bash
pip install token-sentinel[anthropic]
```

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")

@sentinel.on_leak  # or @sentinel.on_waste
def handle(event):
    print(f"[{event.type}] confidence={event.confidence:.2f} burn=${event.estimated_burn:.4f}")

client = sentinel.wrap(anthropic.Anthropic())
```

`mode="log"` is safe for production day one. See [Modes](./03-modes.md) before enabling `block`.

## Contents

1. [Installation](./01-installation.md) — extras, Python version, OSS vs cloud
2. [Quickstart](./02-quickstart.md) — end-to-end with a real client
3. [Modes](./03-modes.md) — `log` / `alert` / `block`, cloud axis, policy notes
4. [Waste rules](./04-waste-rules.md) — all 15 rules, thresholds, tuning
5. [Providers](./05-providers.md) — native + OpenAI-compatible matrix
6. [Integrations](./06-integrations.md) — MCP, RAG, LangChain, OTel, frameworks
7. [API reference](./07-api-reference.md) — public surface
8. [Troubleshooting](./08-troubleshooting.md) — common failures
9. [FAQ](./09-faq.md) — OSS vs paid cloud, comparisons, contributing

## Also in this repository

| Item | Location |
|---|---|
| Architecture | [`docs/architecture.md`](../architecture.md) |
| Waste taxonomy (design depth) | [`docs/waste-taxonomy.md`](../waste-taxonomy.md) |
| Examples | `examples/` |
| Changelog | `CHANGELOG.md` |
| License | `LICENSE` (Apache-2.0) |

## Conventions

- **Runnable samples** with real SDK shapes
- **Defaults favor fewer false positives** — tune thresholds up or disable rules per project
- **Cloud pricing and dashboard UX** live on the product site, not in this OSS guide

## Stability

Package version **1.0.1+**. Public API: `Sentinel`, `wrap`, `on_leak` / `on_waste`, `record_call`, `session`, `mark_long_running`, `close`, `CallRecord`, `LeakEvent` / `WasteEvent`, `LeakDetected` / `WasteDetected`, policy exceptions. Semver applies. Internal modules may change between minors.

## Getting help

- Bug reports / features: GitHub Issues on this repository
- Security: GitHub Security Advisory on the repository
