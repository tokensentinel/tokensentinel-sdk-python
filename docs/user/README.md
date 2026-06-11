# TokenSentinel User Guide

TokenSentinel catches token waste in AI agents *mid-run* — before your bill arrives — and gives your application a callback to log, alert, or hard-stop the agent.

It is a Python SDK that wraps the official LLM clients (Anthropic, OpenAI, Gemini, Bedrock, plus everything OpenAI-compatible). Eight deterministic rules watch the call stream for the patterns that account for most agent token waste in the wild: tool loops, context bloat, embedding waste, zombie agents, model misroutes, retry storms, tool-definition bloat, and retrieval thrash.

This guide is for developers integrating TokenSentinel into their application. It assumes you have a working LLM-calling Python codebase and want to understand how the SDK behaves before you ship it to production.

## Who this is for

- Backend engineers running agents in production who want to know which agent is leaking *right now*, not which agent leaked last month.
- Platform teams owning shared LLM infrastructure who need automated guardrails for tenant misuse.
- AI engineers tuning agents who want a fast, deterministic signal during development that an iteration loop is degenerating.

If you only want post-hoc cost analysis, a traditional observability tool (Langfuse, LangSmith, Helicone, Datadog LLM) covers that better. TokenSentinel sits in the call path so it can intervene before the meter spins.

## 30-second quickstart

```bash
pip install token-sentinel[anthropic]
```

```python
from token_sentinel import Sentinel
import anthropic

sentinel = Sentinel(project="my-agent", mode="log")

@sentinel.on_leak
def handle(event):
    print(f"LEAK [{event.type}] confidence={event.confidence:.2f} burn=${event.estimated_burn:.4f}")

client = sentinel.wrap(anthropic.Anthropic())
# Use `client` exactly as you would a normal anthropic.Anthropic.
```

That's the whole integration. `mode="log"` is safe for production from day one — Sentinel only emits to your handler. To halt agents on detection, switch to `mode="block"` (see [Modes](./03-modes.md)).

## Contents

1. [Installation](./01-installation.md) — pip extras, Python version, how to verify your install.
2. [Quickstart](./02-quickstart.md) — five-minute end-to-end tutorial with a real client and a synthetic leak.
3. [Modes](./03-modes.md) — `log` / `alert` / `block`, when to use each, the graduation pattern.
4. [Leak rules](./04-waste-rules.md) — what each of the eight rules detects, default thresholds, and how to tune.
5. [Providers](./05-providers.md) — Anthropic, OpenAI, Gemini, Bedrock, OpenAI-compatible, self-hosted. Pick yours.
6. [Integrations](./06-integrations.md) — MCP hosts, RAG pipelines, LangChain, LangGraph, CrewAI, AutoGen, Pydantic AI.
7. [API reference](./07-api-reference.md) — full kwarg-by-kwarg reference for `Sentinel`, `CallRecord`, `LeakEvent`.
8. [Troubleshooting](./08-troubleshooting.md) — what to check when a rule didn't fire, did fire incorrectly, or your overhead is high.
9. [FAQ](./09-faq.md) — pricing direction, cloud vs OSS, comparisons to other tools, contributing.

## Where things live

| Item | Location |
|---|---|
| SDK source | `token_sentinel/` |
| User docs (this guide) | `docs/user/` |
| Examples | `examples/` |
| Changelog | `CHANGELOG.md` |
| License | `LICENSE` (Apache-2.0) |

## Conventions in this guide

- **Code samples are runnable.** Every snippet uses real model names and real provider SDK shapes. If a snippet does not run for you, it is a documentation bug — please open an issue.
- **No marketing.** This guide is technical reference, not pitch material.
- **Defaults erred toward fewer false positives.** The eight rules are tuned to under-fire rather than over-fire. If you want stricter detection, lower thresholds. If you want quieter behavior, raise them or disable rules per-project.

## Stability

The SDK is in stable 1.0.0 release. The public surface — `Sentinel`, `Sentinel.wrap`, `Sentinel.on_leak`, `Sentinel.record_call`, `LeakEvent`, `CallRecord`, `LeakDetected` — is stable and we follow semver: breaking changes will get a major version bump. Internal modules (`token_sentinel.tracer`, `token_sentinel.rules.*`, `token_sentinel.cloud_client`, `token_sentinel.wrappers.*`) are not stable yet — pin to a minor version if you depend on them directly.

## Getting help

- Bug reports and feature requests: GitHub Issues
- Discussions: GitHub Discussions
- Security disclosures: open a GitHub Security Advisory on the repository for private disclosure.
