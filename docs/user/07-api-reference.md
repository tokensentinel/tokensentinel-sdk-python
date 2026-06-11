# API reference

Full reference for the TokenSentinel public surface. The public API is everything exported from `token_sentinel`:

```python
from token_sentinel import Sentinel, LeakEvent, CallRecord, LeakDetected
```

These four names are the entire stable surface. Everything under `token_sentinel.tracer` and `token_sentinel.rules.*` is internal — pin to a minor version if you import those.

## `Sentinel`

```python
class Sentinel:
    def __init__(
        self,
        *,
        project: str,
        mode: Literal["log", "alert", "block"] = "log",
        rules: list[str] | Literal["all"] = "all",
        config: dict[str, Any] | None = None,
        cloud_endpoint: str | None = None,
        api_key: str | None = None,
        min_confidence: float = 0.5,
        max_records_per_session: int = 200,
        max_sessions: int | None = 1000,
        dedup_window_seconds: float = 5.0,
        cloud_flush_interval_seconds: float = 5.0,
        cloud_batch_size: int = 50,
        cloud_queue_max: int = 1000,
    ) -> None: ...

    def close(self, timeout: float = 5.0) -> bool: ...
```

The main entry point. One `Sentinel` per logical project per process is the recommended pattern.

### Constructor parameters

All parameters are keyword-only.

#### `project: str`

Required. Identifier for this project, included on every emitted `LeakEvent.project`. Use it to disambiguate when one process runs multiple Sentinels (rare) or to route events downstream.

```python
Sentinel(project="checkout-agent")
```

#### `mode: Literal["log", "alert", "block"] = "log"`

Behavior when a rule fires. See [Modes](./03-modes.md) for the full discussion.

- `"log"` — call registered handlers; never raise.
- `"alert"` — call handlers; also emit to cloud sink if configured.
- `"block"` — call handlers; raise `LeakDetected` from the wrapped call.

```python
Sentinel(project="my-agent", mode="log")
```

#### `rules: list[str] | Literal["all"] = "all"`

Which rules to run. `"all"` runs every V0 rule. Pass a list of rule names to enable a subset:

```python
Sentinel(project="my-agent", rules=["embedding_waste", "retry_storm"])
```

Valid rule names: `tool_loop`, `context_bloat`, `embedding_waste`, `zombie`, `model_misroute`, `retry_storm`, `tool_definition_bloat`, `retrieval_thrash`. Names not in this list are silently ignored.

#### `config: dict[str, Any] | None = None`

Per-rule configuration. Keys follow the pattern `"<rule_name>.<config_key>"`:

```python
Sentinel(
    project="my-agent",
    config={
        "tool_loop.cosine_threshold": 0.80,
        "tool_loop.min_calls": 5,
        "context_bloat.slope_threshold": 3000,
        "retry_storm.min_retries": 10,
        "retrieval_thrash.cosine_threshold": 0.75,
    },
)
```

See [Leak rules](./04-waste-rules.md) for every rule's config keys and defaults.

Unknown keys are silently ignored — typos won't crash, but they also won't take effect. Verify by inspecting the rule's behavior or by reading rule source if you suspect a key isn't being applied.

#### `cloud_endpoint: str | None = None`

Optional URL for the cloud sink. When set, `mode="alert"` and `mode="block"` will fire-and-forget POST events to `{cloud_endpoint}/v1/events`. Network failures never block the agent.

```python
Sentinel(project="my-agent", mode="alert", cloud_endpoint="https://api.tokensentinel.dev")
```

The cloud is opt-in. The SDK is fully functional without it. See [FAQ](./09-faq.md#cloud-vs-oss) for the OSS-vs-cloud split.

#### `api_key: str | None = None`

API key for the cloud sink. Required when `cloud_endpoint` is set; ignored otherwise.

```python
Sentinel(project="my-agent", cloud_endpoint="https://cloud.example.com", api_key="ts_...")
```

#### `min_confidence: float = 0.5`

Project-wide confidence floor. Events with `confidence < min_confidence` are dropped silently before reaching handlers. Default `0.5`.

```python
Sentinel(project="my-agent", min_confidence=0.7)  # only high-confidence events
```

This is independent of `mode` — handlers see only events at or above `min_confidence`, regardless of whether you're in `log`, `alert`, or `block`.

### Methods

#### `Sentinel.wrap(client: T) -> T`

Wraps an LLM client in place and returns it. The returned object is the same `client` you passed in — wrappers mutate methods on the live instance, so all your IDE type hints continue to work.

```python
import anthropic
from token_sentinel import Sentinel

sentinel = Sentinel(project="my-agent")
client = sentinel.wrap(anthropic.Anthropic())
# `client` is the same anthropic.Anthropic, with messages.create / messages.stream instrumented.
```

**Supported client types** (dispatched by `type(client).__module__`):

- Anthropic: `anthropic.Anthropic`, `anthropic.AsyncAnthropic`.
- OpenAI: `openai.OpenAI`, `openai.AsyncOpenAI`. Also covers any OpenAI-compatible base_url (DeepSeek, Together, Fireworks, Groq, OpenRouter, Anyscale, Mistral, Perplexity, vLLM, Ollama, TGI, LM Studio).
- Google Gemini: `google.genai.Client` (direct API or Vertex backend).
- AWS Bedrock: `boto3.client("bedrock-runtime")`.

If the client type is not recognized, raises `TypeError`. If the SDK is on an unexpected version such that the dispatcher doesn't recognize the module prefix, raises `TypeError` — upgrade to the version listed in [Installation](./01-installation.md).

**Calling wrapped methods.** Every wrapped method accepts an extra `_sentinel_session_id` kwarg that is stripped before reaching the SDK:

```python
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[...],
    _sentinel_session_id="user-42-task-17",  # group calls into one logical session
)
```

Without `_sentinel_session_id`, each call gets a fresh UUID — useful only for one-shot calls. For any rule that detects patterns, set a stable session id.

#### `Sentinel.on_leak(handler: Callable[[LeakEvent], None]) -> Callable[[LeakEvent], None]`

Register a leak event handler. Returns the handler unchanged, so it works as a decorator:

```python
sentinel = Sentinel(project="my-agent")

@sentinel.on_leak
def handle(event: LeakEvent) -> None:
    print(f"LEAK [{event.type}] confidence={event.confidence:.2f}")
```

You can register multiple handlers; they run in registration order. Exceptions in one handler do not block others — they are caught and swallowed.

Handlers run synchronously in the wrapped call's thread *after* the underlying API response has been received. Keep them short — long-running handlers add to the wrapped call's perceived latency.

For async work, dispatch to a queue inside the handler:

```python
import asyncio
queue: asyncio.Queue[LeakEvent] = asyncio.Queue()

@sentinel.on_leak
def handle(event):
    queue.put_nowait(event)
```

#### `Sentinel.record_call(call: CallRecord) -> list[LeakEvent]`

Direct injection — records a `CallRecord` and runs all rules against the session buffer. Returns the list of `LeakEvent` that fired (after `min_confidence` filtering). Handlers are also dispatched.

This is the path used by every wrapper internally. You generally don't call it directly — but it's useful for:

- Synthetic test data (`examples/tool_loop_demo.py`).
- Custom providers not yet covered by a wrapper. Build a `CallRecord` from your provider's response and hand it to `record_call`.
- Replay testing — feed historical traces back through the rules engine.

```python
from datetime import datetime, timezone
from token_sentinel import CallRecord

call = CallRecord(
    session_id="my-session",
    timestamp=datetime.now(timezone.utc),
    provider="custom",
    model="my-model",
    method="generate",
    prompt_tokens=120,
    completion_tokens=40,
    latency_ms=320.5,
    request_hash="...",
)
events = sentinel.record_call(call)
```

In `mode="block"`, `record_call` may raise `LeakDetected` — either let it propagate (matching the wrapper's behavior) or catch it explicitly.

### Attributes

- `sentinel.project: str` — the project string passed in.
- `sentinel.mode: str` — the mode.
- `sentinel.config: dict[str, Any]` — the config dict.
- `sentinel.min_confidence: float` — the confidence floor.
- `sentinel.tracer: Tracer` — internal tracer (see below).

The tracer is an internal class — its `record`, `session`, `clear` methods exist but are not part of the stable API. We expose `sentinel.tracer.clear(session_id)` for tests and demos.

---

## `CallRecord`

```python
@dataclass
class CallRecord:
    session_id: str
    timestamp: datetime
    provider: str
    model: str
    method: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    request_hash: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    user_facing_output: bool = False
    raw_request: dict[str, Any] = field(default_factory=dict)
    raw_response_meta: dict[str, Any] = field(default_factory=dict)
```

A captured LLM API call. Wrappers build these from the SDK's request/response; rules read them. You only construct one directly when injecting test data via `Sentinel.record_call`.

| Field | Type | Description |
|---|---|---|
| `session_id` | str | Logical session this call belongs to. Defaults to a per-call UUID inside wrappers; pass `_sentinel_session_id` to override. |
| `timestamp` | datetime | UTC timestamp the call was made. |
| `provider` | str | `"anthropic"`, `"openai"`, `"gemini"`, or `"bedrock"`. OpenAI-compatible providers (DeepSeek, vLLM, etc.) come through as `"openai"`. |
| `model` | str | Model identifier from the request — `"claude-sonnet-4-6"`, `"gpt-5"`, `"gemini-2.5-pro"`, `"anthropic.claude-sonnet-4-5-v2:0"`, etc. |
| `method` | str | SDK method called: `"messages.create"`, `"messages.stream"`, `"chat.completions.create"`, `"embeddings.create"`, `"models.generate_content"`, `"models.generate_content_stream"`, `"converse"`, or `"converse_stream"`. |
| `prompt_tokens` | int | Input token count from the response's usage metadata. |
| `completion_tokens` | int | Output token count from the response's usage metadata. For embeddings, always 0. |
| `latency_ms` | float | Wall-clock latency of the underlying SDK call in milliseconds. |
| `request_hash` | str | SHA-256 hex digest of `(model, messages, tools, max_tokens)` (Anthropic / OpenAI / Gemini) or `(modelId, messages, toolConfig, inferenceConfig)` (Bedrock). Stable across retries. Used by `retry_storm`. |
| `tool_calls` | list[dict] | Tool calls in the response, normalized to `[{"name": str, "arguments": dict | str}, ...]` across all providers. Used by `tool_loop` and `retrieval_thrash`. |
| `user_facing_output` | bool | True iff the response contains text content and no tool calls. Used by `zombie`. |
| `raw_request` | dict | Provider-specific request shape: `{"messages": ..., "tools": ..., "max_tokens": ...}` for Anthropic/OpenAI; `{"model": ..., "contents": ..., "tools": ...}` for Gemini; `{"modelId": ..., "messages": ..., "toolConfig": ...}` for Bedrock; `{"input": ..., "model": ...}` for embeddings. Used by `tool_definition_bloat`, `embedding_waste`, `model_misroute`. |
| `raw_response_meta` | dict | Small response metadata. Anthropic: `{"stop_reason": ..., "streamed": bool}`. OpenAI: `{"finish_reason": ...}`. Gemini: `{"finish_reason": ...}`. Bedrock: `{"stopReason": ...}`. |

---

## `LeakEvent`

```python
@dataclass
class LeakEvent:
    type: str
    confidence: float
    project: str
    session_id: str
    rule: str
    evidence: dict[str, Any]
    estimated_burn: float
    suggested_action: str
    raised_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

A waste signal emitted by a rule. Passed to your `on_leak` handler and (in `block` mode) attached to the `LeakDetected` exception.

| Field | Type | Description |
|---|---|---|
| `type` | str | Leak class. One of `tool_loop`, `context_bloat`, `embedding_waste`, `zombie`, `model_misroute`, `retry_storm`, `tool_definition_bloat`, `retrieval_thrash`. |
| `confidence` | float | 0.0–1.0. Below `Sentinel.min_confidence` the event is dropped before reaching handlers. |
| `project` | str | The `project` string from the `Sentinel` that emitted this. |
| `session_id` | str | The session id this leak was detected in. Matches the originating `CallRecord.session_id`. |
| `rule` | str | Which rule fired, prefixed with the rules-engine version (e.g. `v0.tool_loop`). |
| `evidence` | dict | Rule-specific payload describing why the rule fired. Keys differ per rule — see [Leak rules](./04-waste-rules.md) for each rule's evidence schema. |
| `estimated_burn` | float | Rough USD figure for the wasted spend this leak represents. Treat as a sort key, not an invoice. For self-hosted endpoints this is not meaningful — see [Providers](./05-providers.md#self-hosted-cost-counter-caveat). |
| `suggested_action` | str | Machine-readable hint for what to do — `route_to_claude-haiku-4-5`, `add_embedding_cache`, `pause_for_human_review`, etc. |
| `raised_at` | datetime | UTC timestamp the event was emitted by the rule. |

`LeakEvent.__str__` returns a short summary:

```python
str(event)  # 'LeakEvent(type=tool_loop, confidence=0.84, burn=$0.0324, rule=v0.tool_loop)'
```

---

## `LeakDetected`

```python
class LeakDetected(Exception):
    event: LeakEvent
    def __init__(self, event: LeakEvent) -> None: ...
```

Raised by the wrapper when `Sentinel.mode == "block"` and a leak fires.

```python
from token_sentinel import Sentinel, LeakDetected
import anthropic

sentinel = Sentinel(project="my-agent", mode="block")
client = sentinel.wrap(anthropic.Anthropic())

try:
    response = client.messages.create(model="claude-sonnet-4-6", messages=[...])
except LeakDetected as exc:
    print(f"Blocked: {exc.event.type} ({exc.event.confidence:.2f})")
    # exc.event is the LeakEvent that caused the block
    # exc.event.evidence has the rule-specific details
```

The exception is raised *after* the underlying LLM call has already returned (the call's response is built into the rule input). Block mode halts the *next* call in a degenerate loop — see [Modes — `block` mode](./03-modes.md#block-mode-hard-stop) for caveats.

---

## Rule names and config keys (cheat sheet)

| Rule | Name | Config keys | Confidence |
|---|---|---|---|
| Tool loop | `tool_loop` | `window_seconds`, `min_calls`, `cosine_threshold`, `similarity_metric`, `charngram_size` | 0.6–0.99 |
| Context bloat | `context_bloat` | `lookback_turns`, `slope_threshold`, `min_turns` | 0.55–0.95 |
| Embedding waste | `embedding_waste` | (none) | 0.99 |
| Zombie agent | `zombie` | `threshold_minutes`, `min_recent_calls` | 0.75 |
| Model misroute | `model_misroute` | `max_prompt_tokens`, `max_completion_tokens` | 0.7 |
| Retry storm | `retry_storm` | `window_seconds`, `min_retries` | 0.9 |
| Tool definition bloat | `tool_definition_bloat` | `tool_count_threshold`, `tool_definition_bytes_threshold` | 0.85 / 0.95 |
| Retrieval thrash | `retrieval_thrash` | `window_seconds`, `min_calls`, `cosine_threshold`, `similarity_metric`, `charngram_size`, `retrieval_tool_patterns` | 0.55–0.95 |

Config key format in the `config` dict is `"<rule_name>.<key>"` — for example, `"tool_loop.cosine_threshold": 0.80`.

See [Leak rules](./04-waste-rules.md) for default values, semantics, and tuning examples.

---

## Module-level constants

These are not in the public `__all__` but are documented because users sometimes need them:

- `token_sentinel.__version__` — current version string (e.g., `"0.4.0"`).
- `token_sentinel.tracer.Tracer` — the in-process ring buffer. Internal class (subject to change). The `Sentinel` constructor exposes `max_records_per_session=` (default 200) and `max_sessions=` (default 1000) for the most common tuning needs — prefer those over instantiating `Tracer` directly.

The full list of valid rule-name strings:

```python
{
    "tool_loop",
    "context_bloat",
    "embedding_waste",
    "zombie",
    "model_misroute",
    "retry_storm",
    "tool_definition_bloat",
    "retrieval_thrash",
}
```

---

## Type hints

All public types are exported with full annotations and play well with mypy/pyright. The wrappers use `functools.wraps` and live-instance mutation, so calls to wrapped methods retain the original SDK's signatures and return types.

If you hit a type-checking issue with wrapped clients, it is almost certainly because:

1. You are casting to a narrower type after wrapping. Drop the cast.
2. Your IDE has cached an older stub. Restart the language server.

The wrapper internals use `# type: ignore[method-assign]` because mutating instance methods is a deliberate type violation that the SDKs themselves rely on. Your code that consumes wrapped clients does not need any `type: ignore`.

---

## Stability guarantees

**Stable** (semver-tracked):

- `Sentinel` class: constructor parameters, `wrap`, `on_leak`, `record_call`.
- `LeakEvent` dataclass: field names and types.
- `CallRecord` dataclass: field names and types.
- `LeakDetected` exception: structure and behavior.
- The set of rule names and their config keys.

**Not stable** (may change in minor versions):

- `token_sentinel.tracer.Tracer` internals.
- Individual rule classes under `token_sentinel.rules.*`.
- The exact numeric values of confidence scores within a rule (we tune them).
- The precise wording of `suggested_action` strings (the format is stable, the contents may evolve).
- Wrapper internals (`token_sentinel.wrappers.*`).

If you depend on something not on the stable list, pin to a minor version.
