# FAQ

Frequently asked questions about TokenSentinel — what it is, what it isn't, how it differs from adjacent tools, and how the OSS / cloud split works.

## Why not OpenTelemetry?

OTel is read-only by design — spans are emitted *after* the call completes. The wedge TokenSentinel sits in is mid-run intervention: returning a callback (or in `block` mode, raising an exception) that the host app's flow control can act on *before* the next call goes out. By the time an OTel span fires, the meter has already spun.

We will *emit* OTel spans (planned) so you can fan TokenSentinel events into your existing observability stack. But we don't *ingest* via OTel — the data path needs to be in-process, in the call stack, with the ability to short-circuit.

If you want post-hoc cost analysis without intervention, OTel-based tools (Helicone, Langfuse, LangSmith, Datadog LLM) cover that better.

## How is this different from Langfuse / LangSmith / Helicone / Datadog LLM?

Different problem.

| Tool | Tells you |
|---|---|
| Langfuse / LangSmith | What your traces looked like. Great for prompt iteration and offline analysis. |
| Helicone | What your bill was, broken down by request. Excellent for cost reporting. |
| Datadog LLM | What your LLM traffic looks like alongside the rest of your infra. Good for ops dashboards. |
| TokenSentinel | Which agent is leaking *right now*, and gives your code a callback to intervene. |

The tools above are observability — read-only, after-the-fact, optimized for analysis. TokenSentinel is detection + intervention — in the call stack, real-time, optimized for stopping a runaway loop before it bills 1,000 calls.

You can run TokenSentinel alongside any of them. They don't conflict — TokenSentinel sits at the SDK wrap layer; observability tools usually sit at the HTTP / OTel layer.

## Does TokenSentinel work with self-hosted LLMs?

Yes. vLLM, Ollama, text-generation-inference, LM Studio, and LocalAI all expose OpenAI-compatible endpoints. Use the standard `openai` SDK with `base_url` pointing at your server, then wrap it:

```python
import openai
from token_sentinel import Sentinel

sentinel = Sentinel(project="self-hosted")
client = sentinel.wrap(openai.OpenAI(base_url="http://localhost:8000/v1", api_key="local"))
```

**Caveat about cost figures.** TokenSentinel's waste signals (tool loop, context bloat, embedding waste, zombie agent, model misroute, retry storm, tool definition bloat, retrieval thrash) work fully — they're computed from token counts and call patterns. The dollar `estimated_burn` figure assumes priced API usage and is not meaningful for self-hosted (your tokens are GPU-amortized, not API-priced). Treat the burn estimate as a relative-cost signal on self-hosted, not as an invoice. See [Providers — self-hosted](./05-providers.md#self-hosted-cost-counter-caveat).

## What does TokenSentinel cost?

The SDK is **Apache-2.0 licensed** and **free forever**. You get every rule, every wrapper, and every detection signal in `log` / `alert` / `block` modes, self-hosted, with no cloud dependency.

The optional hosted cloud dashboard is closed-source and currently in design-partner phase — not generally available yet. Pricing for the hosted tiers (and the enterprise tier with chargeback / SOC 2 / SSO) will be published when the cloud goes GA. We expect the SDK-only path to remain free indefinitely.

If you want to be a design partner for the cloud while it's in beta, open an issue or reach out via the project README.

## Is there a TypeScript or Go SDK?

Not yet. Python first. TypeScript and Go are on the V3 roadmap — once the V0/V1 detection is stable in production, we'll port.

If you want one urgently and have a real use case, open an issue — that's signal we use to prioritize.

In the meantime, if your stack is mostly TypeScript/Go but you have a Python sidecar that proxies LLM calls, you can run TokenSentinel in the sidecar.

## How do I contribute?

The SDK source is open — `token_sentinel/` in the repo.

- **Bug reports**: GitHub Issues with a minimal repro.
- **Feature requests**: GitHub Issues, tagged `enhancement`. Include the use case — we triage against "how much agent traffic in the wild does this affect".
- **Pull requests**: welcome. For non-trivial changes, open an issue first to discuss the approach. The test / lint / typecheck commands are: `pytest tests/ -q`, `ruff check token_sentinel tests examples`, and `mypy token_sentinel`.
- **New provider wrappers**: also welcome. Use `wrappers/anthropic.py` as the reference pattern (it's the best-documented). Include unit tests covering sync, async, streaming (where the provider supports it), and the two-level safety boundary (record-build errors swallowed, `LeakDetected` propagated).
- **New rules**: open an issue describing the leak class, the signal you'd use, and the false-positive surface. Rules need to be deterministic, sub-millisecond, and pure functions of the session buffer.

## What's the license?

The SDK is **Apache-2.0**. See `LICENSE` in the repo root. Use it freely in commercial projects, fork it, embed it, redistribute it. The patent grant in Apache-2.0 is the right OSS contract for an SDK that runs inline against your production AI calls.

## Is the cloud dashboard open-source?

**No.** The SDK in this repository is Apache-2.0 licensed and you can run it standalone forever. The hosted cloud dashboard (alerts, retention, team features, chargeback attribution) is closed-source and runs on our infrastructure.

This is a deliberate split. The detection logic (rules, wrappers, tracer) is the part the community benefits from auditing and contributing to — that's all open. The dashboard, the analytics infrastructure, the team management, and the multi-tenant data plane are the parts we charge for.

The cloud is also opt-in in every sense — the SDK works perfectly without it, and there's no telemetry phoning home unless you explicitly configure `cloud_endpoint`.

## Why call it "TokenSentinel"?

It watches token spend (the cost surface of LLMs) and stands sentinel over the call path (the in-process integration model). The name is descriptive, not aspirational.

## Does TokenSentinel send my prompts anywhere?

Not by default.

In `mode="log"` and `mode="block"`, TokenSentinel runs entirely in-process. No network calls outside of your normal LLM provider traffic. The rules engine reads the captured `CallRecord` from the in-memory ring buffer and emits events to your registered handlers. Nothing leaves the process.

In `mode="alert"`, if (and only if) you configured `cloud_endpoint` and `api_key`, the cloud sink fire-and-forget POSTs `LeakEvent` summaries (event type, confidence, evidence, project, session_id) to your configured endpoint. The `evidence` dict for some rules contains short snippets of arguments (e.g., `tool_loop`'s `sample_args` includes the first three tool-call argument dicts). If you have sensitive data there, you can scrub the event in your `on_leak` handler before it reaches the cloud sink — handlers are called *before* the cloud-sink dispatch.

The full request body (`raw_request`) is never sent to the cloud. Only the event evidence is.

## What's the V1 roadmap?

V1 adds an **LLM-as-judge** pass: a cheap model (Haiku) reads the gray-zone V0 firings (confidence 0.5–0.75) and ratifies or vetoes them. Haiku polices Opus. This dramatically reduces false positives on heuristic rules without raising thresholds.

V1 also adds:

- Optional sentence-transformer-based semantic similarity for `tool_loop` (via the `[embeddings]` extra).
- Per-rule mode (e.g., `block` only on `embedding_waste`, `log` everything else).
- Polling-tool allow-lists for `tool_loop`.
- Context-token-entropy refinement for `context_bloat`.

V1 is on the roadmap, not shipped. Until then, tune thresholds and use rule disable lists to manage noise. See [Leak rules — Tuning workflow](./04-waste-rules.md#tuning-workflow).

## How does the cloud-vs-OSS split work technically?

The SDK in this repo emits `LeakEvent` to your registered handlers. Always.

In `mode="alert"`, the SDK *additionally* batches events and POSTs them to the configured `cloud_endpoint`. The cloud accepts those events and stores them, indexes them, runs alert routing on them, and exposes dashboards. The cloud runs all closed-source code outside this repo.

If you don't configure `cloud_endpoint`, `mode="alert"` behaves identically to `mode="log"` — your handlers are called, no network traffic.

The cloud's API surface is small and stable: a single `/v1/events` endpoint that accepts a JSON array of `LeakEvent`-shaped objects. If you want to host your own sink (write events to your own database, route your own alerts), point `cloud_endpoint` at your endpoint and implement that contract.

## Is TokenSentinel production-ready?

The SDK is in **stable 1.0.0** release. The public surface is stable, the test suite is comprehensive (912 passing tests), and the fail-safe design ensures TokenSentinel never breaks your agent — instrumentation errors are caught, rule errors are caught, handler errors are caught, the optional cloud sink runs on a daemon thread that never blocks the agent and never raises into user code. The only exception that propagates is `LeakDetected` in `block` mode, which is the entire point.

The public API is stable and follows semver.

For production deployment, we recommend:

1. Start in `mode="log"` only. Always.
2. Run for a week. Identify any rules firing falsely on your traffic. Tune.
3. Promote one rule at a time to `block` mode, starting with the highest-precision (`embedding_waste`, `retry_storm`).
4. Don't promote heuristic rules (`tool_loop`, `model_misroute`, `context_bloat`) to `block` until LLM-as-judge ratification ships.

## How can I get help?

- GitHub Issues for bugs and feature requests.
- GitHub Discussions for questions.
- Security disclosures: open a GitHub Security Advisory on the repository for private disclosure.

We do not currently offer paid support. If you need it, open an issue describing your use case.
