# API reference

> **Documented for Python SDK `token-sentinel` 1.0.1.**  
> Runtime version: `import token_sentinel; token_sentinel.__version__`

Public surface exported from `token_sentinel` (stable, semver-tracked):

```python
from token_sentinel import (
    Sentinel,
    CallRecord,
    LeakEvent,
    WasteEvent,       # alias of LeakEvent
    LeakDetected,
    WasteDetected,    # alias of LeakDetected
    BudgetExceeded,
    VelocityExceeded,
    KillSwitchActive,
    __version__,
)
```

Everything under `token_sentinel.tracer`, `token_sentinel.rules.*`, and `token_sentinel.wrappers.*` is internal — pin a minor version if you import those.

`WasteEvent is LeakEvent` and `WasteDetected is LeakDetected` are true (transparent aliases). Prefer either naming style; both stay first-class.

---

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
        # Optional cloud policy plane (Intervention Pack — paid cloud).
        # Default: same URL as cloud_endpoint. Pass None to disable policy only.
        policy_endpoint: str | None = ...,
        policy_active_poll_seconds: float = 2.0,
        policy_idle_poll_seconds: float = 5.0,
        policy_failure_mode: Literal["open", "closed"] = "open",
        # Optional Pro judge hints (forwarded as headers to cloud)
        judge_threshold_low: float = 0.5,
        judge_threshold_high: float = 0.8,
        judge_calls_per_month_max: int = 1_800_000,
    ) -> None: ...

    def wrap(self, client: T) -> T: ...
    def on_leak(self, handler: Callable[[LeakEvent], None]) -> Callable[[LeakEvent], None]: ...
    on_waste = on_leak  # brand alias
    def unregister(self, handler: Callable[[LeakEvent], None]) -> bool: ...
    def record_call(self, call: CallRecord) -> list[LeakEvent]: ...
    def session(
        self,
        session_id: str | None = None,
        *,
        tags: dict[str, str] | None = None,
    ) -> Session: ...
    def mark_long_running(self, session_id: str) -> None: ...
    def unmark_long_running(self, session_id: str) -> None: ...
    def close(self, timeout: float = 5.0) -> bool: ...
```

One `Sentinel` per logical project per process is the recommended pattern.

### Constructor parameters

All parameters are keyword-only.

| Parameter | Default | Description |
|---|---|---|
| `project` | required | Included on every `LeakEvent.project` |
| `mode` | `"log"` | `log` / `alert` / `block` — see [Modes](./03-modes.md) |
| `rules` | `"all"` | `"all"` or list of rule name strings (15 names below) |
| `config` | `None` | Per-rule knobs as `"rule_name.key"` |
| `cloud_endpoint` | `None` | Optional cloud base URL (no trailing `/v1/...`) |
| `api_key` | `None` | Required with `cloud_endpoint` for cloud sink + default policy poller |
| `min_confidence` | `0.5` | Drop events with lower confidence before handlers (clamped to \[0, 1\]) |
| `max_records_per_session` | `200` | Ring buffer depth per session |
| `max_sessions` | `1000` | LRU cap on sessions (`None` = unlimited) |
| `dedup_window_seconds` | `5.0` | Suppress duplicate `(type, rule, evidence)` per session; `0` disables |
| `cloud_flush_interval_seconds` | `5.0` | Cloud batch flush interval |
| `cloud_batch_size` | `50` | Max events per POST |
| `cloud_queue_max` | `1000` | In-memory cloud queue; drops oldest on overflow |
| `policy_endpoint` | same as `cloud_endpoint` | Policy poll URL; pass `None` to disable policy while keeping event sink |
| `policy_active_poll_seconds` | `2.0` | Policy poll while sessions marked active |
| `policy_idle_poll_seconds` | `5.0` | Policy poll when idle |
| `policy_failure_mode` | `"open"` | `"open"` = no policy after TTL; `"closed"` = synthetic kill-switch |
| `judge_threshold_*` / `judge_calls_per_month_max` | see above | Forwarded to cloud as `X-Judge-*` headers (Pro-tier cloud feature) |

**Cloud sink activation:** both `cloud_endpoint` and `api_key` must be set. When they are, events are POSTed in **every** mode (`log` included) — mode is stamped on the wire for cloud analytics; it does not gate shipping.

**Unknown rule names** in `rules=[...]` emit a `UserWarning` and are ignored (not a hard error).

### Methods

#### `wrap(client) -> client`

Mutates the live client in place and returns it (IDE types preserved). Supported families: Anthropic, OpenAI (+ compatible), Gemini, Bedrock, Voyage, Cohere V2, Replicate, Deepgram, ElevenLabs. Raises `TypeError` if unsupported or accessors are missing.

Every instrumented method accepts `_sentinel_session_id=` (stripped before the provider SDK).

#### `on_leak` / `on_waste`

Register a sync handler. Multiple handlers run in registration order. Handler exceptions are swallowed. `BaseException` (`KeyboardInterrupt`, `SystemExit`) propagates.

#### `unregister(handler) -> bool`

Remove a previously registered handler. Returns whether it was found.

#### `record_call(call) -> list[LeakEvent]`

Record a call, enforce optional cloud policy, run rules, dispatch handlers (and cloud), optionally raise in `block` mode. Used by all wrappers.

Order inside `record_call`:

1. Policy checks (if configured) — may raise `KillSwitchActive` / `BudgetExceeded` / `VelocityExceeded` **regardless of mode**
2. Append to tracer
3. Evaluate rules → confidence clamp / `min_confidence` / tag stamp
4. Dedup
5. Handlers + cloud enqueue
6. If `mode=="block"` and any events → `LeakDetected` (highest confidence)

#### `session(session_id=None, *, tags=None) -> Session`

Open a logical session with optional chargeback tags.

Allowed tag keys: `team`, `feature`, `customer`, `environment`, `version`.  
Values: URL-safe `^[a-zA-Z0-9._-]+$`, ≤64 chars, ≤8 entries.

```python
sess = sentinel.session(tags={"team": "growth", "feature": "chat"})
# Use sess.session_id as _sentinel_session_id on wrapped calls.
# sess.record_call(call) stamps tags when CallRecord.tags is empty.
```

#### `mark_long_running` / `unmark_long_running`

Opt a `session_id` out of (or back into) the `zombie` rule — for overnight research jobs and other intentionally silent agents.

#### `close(timeout=5.0) -> bool`

Flush cloud sink and stop policy poller. Optional for short scripts; recommended for long-lived workers before exit.

### Attributes

- `project`, `mode`, `config`, `min_confidence`
- `tracer` — internal ring buffer (not a stable API; useful in tests)

---

## `Session`

Lightweight handle from `Sentinel.session()`:

- `session_id: str`
- `tags: dict[str, str]`
- `record_call(call: CallRecord) -> list[LeakEvent]` — stamps `session_id` / tags when empty

No `close()` — sessions age out via `max_sessions` LRU.

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
    usage_extra: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
```

| Field | Description |
|---|---|
| `provider` | e.g. `anthropic`, `openai`, `gemini`, `bedrock`, `voyage`, `cohere`, `replicate`, `deepgram`, `elevenlabs` |
| `method` | e.g. `messages.create`, `chat.completions.create`, `embeddings.create`, `embed`, `rerank`, `converse`, `transcribe_file`, `text_to_speech.convert`, … |
| `request_hash` | Stable hash of the call shape for `retry_storm` / `rerank_thrash` |
| `tool_calls` | `[{"name": str, "arguments": dict\|str}, ...]` |
| `user_facing_output` | Typically true when response has text and no tool calls |
| `raw_request` | Provider-shaped request fragment used by rules |
| `raw_response_meta` | Stop reason, `streamed`, `usage_unavailable`, `prompt_tokens_details`, … |
| `usage_extra` | Non-token billing: `dimension_kind`, `dimension_value`, optional `model_specific_meta` |
| `tags` | Chargeback tags from `Session` / enrichers |

---

## `LeakEvent` / `WasteEvent`

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
    raised_at: datetime = ...
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
```

`type` is one of the fifteen rule types (or policy types when raised via policy exceptions):

```
tool_loop, context_bloat, embedding_waste, zombie, model_misroute,
retry_storm, tool_definition_bloat, retrieval_thrash,
vision_re_upload, vision_high_detail_misroute, vision_cost_concentration,
audio_multichannel_doubling, voice_switching_loop, rerank_thrash, repair_loop
```

Policy-originated events may use `kill_switch`, `budget_exceeded`, `velocity_exceeded`.

`metadata` may later carry cloud judge verdicts when events round-trip through paid cloud. SDK rules leave it empty.

`str(event)` includes type, confidence, burn, rule, `session_id`, and evidence **keys** (not values — privacy).

---

## Exceptions

### `LeakDetected` / `WasteDetected`

Raised in `mode="block"` after handlers run, with the highest-confidence event on `exc.event`.

### `BudgetExceeded`, `VelocityExceeded`, `KillSwitchActive`

Subclasses of `LeakDetected`. Raised from `record_call` when a cloud **policy** is active — **independent of `mode`**. Only relevant if you configure cloud + policy (paid Team+ tiers on TokenSentinel Cloud).

| Exception | When |
|---|---|
| `KillSwitchActive` | Operator kill-switch on |
| `BudgetExceeded` | Projected session USD burn would exceed policy budget |
| `VelocityExceeded` | Projected tokens/min would exceed policy cap |

**Important:** wrappers invoke the provider **before** `record_call`. Policy raises cannot un-bill the call that just completed; they stop your agent from treating the response as success / continuing the loop. See [Modes](./03-modes.md).

---

## Rule names and config keys

| Rule | Config keys (prefix with `rule_name.`) | Typical confidence |
|---|---|---|
| `tool_loop` | `window_seconds`, `min_calls`, `cosine_threshold`, `similarity_metric`, `charngram_size`, `include_raw_args`, `max_arg_bytes`, `max_total_corpus_bytes`, `polling_tools` | 0.6–0.99 |
| `context_bloat` | `lookback_turns`, `slope_threshold`, `min_turns` | 0.55–0.95 |
| `embedding_waste` | (none) | 0.99 |
| `zombie` | `threshold_minutes`, `min_recent_calls` | 0.75 |
| `model_misroute` | `max_prompt_tokens`, `max_completion_tokens` | 0.7 |
| `retry_storm` | `window_seconds`, `min_retries` | 0.9 |
| `tool_definition_bloat` | `tool_count_threshold`, `tool_definition_bytes_threshold`, `max_tool_bytes`, `max_total_bytes`, `redact_names` | 0.85 / 0.95 |
| `retrieval_thrash` | same similarity knobs as tool_loop + `retrieval_tool_patterns` | 0.55–0.95 |
| `vision_re_upload` | `window_seconds`, `min_calls`, `max_image_bytes` | 0.65–0.99 |
| `vision_high_detail_misroute` | `max_completion_tokens` | 0.75 |
| `vision_cost_concentration` | `image_share_threshold`, `max_completion_tokens` | 0.65 |
| `audio_multichannel_doubling` | (none) | 0.75 / 0.85 |
| `voice_switching_loop` | `window_seconds`, `min_voices` | 0.7–0.9 |
| `rerank_thrash` | `window_seconds`, `min_calls` | 0.75–0.9 |
| `repair_loop` | `window_turns`, `min_corrections`, `similarity_threshold`, `length_ratio` | 0.65–0.9 |

Full semantics: [Leak rules](./04-waste-rules.md).

```python
Sentinel(
    project="my-agent",
    rules=["embedding_waste", "retry_storm", "rerank_thrash"],
    config={"tool_loop.cosine_threshold": 0.80},  # ignored if tool_loop not loaded
)
```

---

## Enrichers (optional extras)

Not re-exported from the package root; import explicitly after installing the extra:

```python
# pip install token-sentinel[langchain]
from token_sentinel.enrichers.langchain import TokenSentinelCallbackHandler

# pip install token-sentinel[otel]
from token_sentinel.enrichers.otel import TokenSentinelSpanProcessor
```

See [Integrations](./06-integrations.md).

---

## Stability

**Stable:** `Sentinel` public methods and constructor kwargs listed above; `CallRecord` / `LeakEvent` field names; exception types; rule **names** and documented config keys.

**Not stable:** tracer/rule/wrapper internals; exact confidence formulas; `suggested_action` string wording; cloud wire extras.

`__version__` is the installed package version string (e.g. `"1.0.1"`).
