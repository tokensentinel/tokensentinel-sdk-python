# Architecture

## Design principles

1. **In-process by default.** runs entirely client-side. Zero network calls just to do detection. Cloud is opt-in.
2. **Wrap, don't replace.** We wrap the official Anthropic, OpenAI, Google Gemini, and AWS Bedrock clients. Customers keep their existing SDK upgrade path.
3. **Be fast or be removed.** ≤50ms p95 overhead per LLM call. Budget-aware rule evaluation.
4. **Predict, don't decide.** Sentinel emits a *signal*. The host app decides what to do.

## Data flow

```
                                                     ┌──────────────────┐
                                                     │  on_leak handler │
                                                     │  (host app)      │
                                                     └────────▲─────────┘
                                                              │
  agent code                                                  │ LeakEvent
       │                                                      │
       │  client.messages.create(...)                         │
       ▼                                                      │
  ┌──────────────┐    ┌─────────────────┐    ┌────────────────┴─────────────┐
  │ wrapped      │───▶│ tracer (in-proc)│───▶│ rules engine (in-proc)       │
  │ Anthropic    │    │  capture I/O    │    │  8 rules + thresholds        │
  │ client       │    │  token counts   │    │  emits LeakEvent             │
  └──────────────┘    └────────┬────────┘    └─────────────┬────────────────┘
                               │                           │
                               │ optional async            │ optional async
                               ▼                           ▼
                       ┌──────────────────────────────────────────────┐
                       │ cloud sink (HTTP, batched, fire-and-forget)  │
                       │ for dashboards / retention / team features   │
                       └──────────────────────────────────────────────┘
```

## Components

### Wrapper layer (`token_sentinel/wrappers/`)

- `wrappers/anthropic.py` — wraps `anthropic.Anthropic` client by reassigning `client.messages.create` to an instrumented function that delegates to the original. Preserves type hints because we mutate the live instance, not a subclass.
- `wrappers/openai.py` — same pattern for the OpenAI v1 client.
- `wrappers/gemini.py` — same pattern for `google.genai.Client` (covers Vertex AI via `vertexai=True`).
- `wrappers/bedrock.py` — same pattern for boto3 `bedrock-runtime` clients.
- Streaming: passthrough generators that fan tokens to the tracer as they arrive. Final usage record reconciled at stream close.

### Tracer (`token_sentinel/tracer.py`)

- Captures: model, prompt tokens, completion tokens, tool calls, latency, parent session, request ID.
- Stores last N events per session in a bounded ring buffer (default N=200, configurable). Memory cap.
- Thread-safe via internal `Lock`. Async safe (asyncio runs in single thread per loop).

### Rules engine (`token_sentinel/rules/`)

- Each rule is a class subclassing `Rule` with `evaluate(session_buffer, project) -> Optional[LeakEvent]`.
- Rules run after every captured call. Total budget across all rules: 50ms.
- Rules are pure functions of the session buffer + global config. No I/O.

### Event bus (`Sentinel._run_handlers`)

- Sync callback (registered with `@sentinel.on_leak`). Multiple handlers run in registration order.
- Handlers run synchronously in the wrapped call's thread *after* the underlying API response has been received. For async work (network calls, queue dispatch), enqueue inside the handler and let a separate worker drain the queue — see [docs/user/03-modes.md](user/03-modes.md#handler-requirements) for the safe pattern.
- Handlers wrapped in try/except — a buggy handler can never kill the agent. `BaseException` (`KeyboardInterrupt`, `SystemExit`) still propagates.

### Cloud sink (optional, v0.4+)

- HTTP POST batched to a configurable `cloud_endpoint` (e.g. `https://api.tokensentinel.dev/v1/events`).
- Fire-and-forget with bounded retry (3 attempts, exponential backoff). Never blocks the agent. Never raises into user code.
- Stdlib-only HTTP via `urllib.request` — keeps the SDK's zero-runtime-deps core intact.
- Customers can self-host the sink — the wire contract is a single `POST /v1/events` with a JSON `{"project": str, "events": [LeakEvent-shaped, ...]}` body.

## State

Two layers of state by default:

1. **Per-session ring buffer.** In-memory, bounded (default 200 records per session, capped at 1000 sessions per Sentinel). Holds recent calls for rule evaluation.
2. **Cloud sink (opt-in).** When you configure `cloud_endpoint=` and `api_key=`, a daemon thread batches and POSTs `LeakEvent`s to that endpoint. Nothing else leaves the process.

Default: in-memory only. Customer opts in to cloud delivery explicitly.

## Type-hint preservation gotcha

Subclassing the OpenAI v1 client breaks IDE type hints because the v1 SDK is heavily generated. Monkey-patching the live instance preserves types but requires us to mutate `client.messages.create` carefully. We document this clearly and prefer mutation over subclassing.

For Anthropic: same approach — mutate the live instance's `messages.create`. The original is captured in a closure.

## Concurrency

- Single Sentinel instance per process is the recommended pattern.
- Multiple wrapped clients can share one Sentinel — they all feed the same tracer keyed by `session_id`.
- `session_id` defaults to a per-call UUID. Customer can pass a stable ID via `client.messages.create(..., _sentinel_session_id="abc")` to group calls into one session.

## Failure modes

- **Tracer raises an exception** → caught silently. Agent continues normally. Fail-safe.
- **A rule raises** → caught silently; that rule's signal is dropped for the call. Other rules continue.
- **Cloud sink unreachable** → events queued in memory up to a cap, then oldest-dropped with a `RuntimeWarning`. Never blocks the agent.
- **Embedding model not installed**  → `tool_loop` rule degrades gracefully to TF-IDF char-n-gram similarity (the default; lower precision but works without any model dependency).

## Why not OTel-first ingestion

OTel is read-only by design. The wedge — mid-run intervention — requires being in the data path so we can return a callback (or in `block` mode, raise an exception) that the host app's flow control can act on. OTel spans are emitted *after* the call completes; the meter has already spun.

We plan to *emit* OTel spans on the roadmap so customers can fan TokenSentinel events into their existing observability stack. We do not *ingest* via OTel.
