# Quickstart — AWS Bedrock

A 5-minute end-to-end: install, wrap, see a leak fire.

## 1. Install

```bash
pip install token-sentinel[bedrock]
```

The `[bedrock]` extra pulls in `boto3>=1.35.0` (which brings `botocore`). Python 3.10+.

## 2. Set your credentials

Bedrock uses the standard AWS credential chain — environment variables, `~/.aws/credentials`, EC2 / ECS / EKS instance profiles, etc. The simplest local path:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

You also need to have requested model access in the Bedrock console (per-model, per-region). See the **Going to production** section for regional considerations.

## 3. Wrap your client

```python
from token_sentinel import Sentinel
import boto3

sentinel = Sentinel(project="my-agent", mode="log")

@sentinel.on_leak
def handle(event):
    print(
        f"LEAK type={event.type} "
        f"confidence={event.confidence:.2f} "
        f"burn=${event.estimated_burn:.4f} "
        f"action={event.suggested_action} "
        f"evidence_keys={list(event.evidence.keys())}"
    )

client = sentinel.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))
```

`sentinel.wrap` mutates the client in place — `client.converse` and `client.converse_stream` are now instrumented. The returned object is the same boto3 client, so all your IDE type hints continue to work.

Only the `bedrock-runtime` service client is wrapped. `boto3.client("bedrock")` (the control-plane client for managing model access, custom models, etc.) is a different service surface and is not instrumented — TokenSentinel cares about runtime traffic, not control plane.

## 4. Trigger a leak

The simplest signal to fire reliably with one real call is `model_misroute`: a classification-shaped prompt aimed at a frontier model.

```python
SESSION = "demo-session-1"

client.converse(
    modelId="anthropic.claude-sonnet-4-5-v2:0",
    messages=[{
        "role": "user",
        "content": [{"text": "Classify this as positive or negative: 'I love this movie'"}],
    }],
    inferenceConfig={"maxTokens": 10},
    _sentinel_session_id=SESSION,
)
```

The `_sentinel_session_id` kwarg is intercepted by the wrapper before the call goes out, so boto3 never sees it. Pass any stable string to group calls into one logical agent run.

The rule fires because:
- The prompt is small (under 500 tokens).
- The output is short.
- The prompt contains the keyword `classify`.
- The `modelId` resolves to a frontier-model family where a smaller model would do.

## 5. See it land in your handler

You should see something like:

```
LEAK type=model_misroute confidence=0.70 burn=$0.0050 action=route_to_claude-haiku-4-5 evidence_keys=['model', 'prompt_tokens', 'completion_tokens', 'matched_keywords', 'recommended_alternative']
```

The handler receives a `LeakEvent` dataclass with these fields:

| Field | Type | What it is |
|---|---|---|
| `type` | str | One of the eight V0 rule types (or composite types fired cloud-side) |
| `confidence` | float | 0.0-1.0; below `min_confidence` (default 0.5) the event is dropped |
| `project` | str | What you passed to `Sentinel(project=...)` |
| `session_id` | str | Identifies a single agent run |
| `rule` | str | Which rule fired (e.g. `v0.model_misroute`) |
| `evidence` | dict | Rule-specific payload — keys documented per rule |
| `estimated_burn` | float | Rough dollar figure for the wasted spend this leak represents |
| `suggested_action` | str | Machine-readable hint |
| `raised_at` | datetime | UTC timestamp |
| `metadata` | dict | Cloud-side judge verdict trail when ratification fires (Pro tier) |

## Streaming

`converse_stream` returns a dict whose `stream` key is a boto3 `EventStream`. The wrapper replaces `response["stream"]` with a proxy that observes each event on the way through and finalises the `CallRecord` on stream end / `close()` / `__del__`.

```python
response = client.converse_stream(
    modelId="anthropic.claude-sonnet-4-5-v2:0",
    messages=[{
        "role": "user",
        "content": [{"text": "Write a haiku about token leaks."}],
    }],
    inferenceConfig={"maxTokens": 200},
    _sentinel_session_id=SESSION,
)
for event in response["stream"]:
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"].get("delta", {})
        if "text" in delta:
            print(delta["text"], end="", flush=True)
print()
```

Token usage is read from the late-arriving `metadata` event (`event["metadata"]["usage"]["inputTokens"] / outputTokens`). `toolUse` deltas arrive across multiple `contentBlockDelta` events with the same `contentBlockIndex` — the wrapper stitches them by index and `json.loads`-es the final argument string.

If you abandon the stream early (break out, drop the response), the proxy still finalises on `__del__` with `LeakDetected` suppressed and a `RuntimeWarning` raised, so block-mode halts are best-effort on abandoned streams. Use `with response["stream"]:` (or fully iterate) for guaranteed halts.

## Async

`boto3` is **sync-only**. There is no `client.aio` surface like the Anthropic / OpenAI / Gemini SDKs expose, so there is nothing for the wrapper to instrument on the async side.

For async Bedrock you have two options:

1. **`aioboto3`** — a third-party shim that wraps boto3 with asyncio. It is **not currently instrumented** by TokenSentinel; calls through `aioboto3` will not produce CallRecords. If you need async Bedrock with leak detection today, run the sync `boto3` client inside `asyncio.to_thread(...)`:
   ```python
   import asyncio, boto3
   from token_sentinel import Sentinel

   sentinel = Sentinel(project="my-agent", mode="log")
   client = sentinel.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))

   async def converse_async(**kwargs):
       return await asyncio.to_thread(client.converse, **kwargs)

   await converse_async(
       modelId="anthropic.claude-sonnet-4-5-v2:0",
       messages=[...],
   )
   ```
2. **Use Anthropic's native SDK directly for the Claude models hosted on Bedrock**, via the `anthropic[bedrock]` package, and wrap with `wrap_anthropic` (which has full async + async-streaming support). Trade-off: you bypass IAM / Bedrock observability for that path.

## Going to production

- Switch from `mode="log"` to `mode="alert"` to get cloud-side dashboards. Configure `cloud_endpoint` and `api_key`:
  ```python
  sentinel = Sentinel(
      project="my-agent",
      mode="alert",
      cloud_endpoint="https://api.tokensentinel.dev",
      api_key="tsk_...",
  )
  ```
- For hard intervention, set `mode="block"` to raise `LeakDetected` at the next call boundary. Wrap calls in `try / except LeakDetected as exc:` and inspect `exc.event`.
- Pair with the cloud (Team / Pro) — see [`docs/pricing.md`](pricing.md). Pro adds the Intervention Pack which raises `BudgetExceeded` / `VelocityExceeded` / `KillSwitchActive` regardless of mode.
- Long-running agents: call `sentinel.close(timeout=5.0)` before exit to flush the cloud sink and stop the policy poller daemon thread.

### Regional considerations

- **`us-east-1` has the broadest model selection** as of May 2026 — Anthropic Claude (full family), Meta Llama, Mistral, Cohere, Amazon Titan, Amazon Nova, AI21. Other regions cover a subset; check the [Bedrock model availability matrix](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) for your target region.
- **One client per region.** Bedrock is a regional service. If your app calls models in multiple regions, build one `boto3.client("bedrock-runtime", region_name=...)` per region and wrap each separately. The wrapper instruments per-instance, not per-service.
- **Cross-region inference profiles** (e.g., `us.anthropic.claude-sonnet-4-5-v2:0`) route the call through a regional pool. The wrapper still records the `modelId` you pass and the resolved usage; the regional pool is opaque to TokenSentinel and that's fine for leak detection.
- **VPC endpoints + PrivateLink** work transparently — the wrapper sees boto3's response shape regardless of how it traveled to AWS.

## Common issues

- **`boto3.Session` is not the same as a client.** Only `bedrock-runtime` clients are wrapped — passing a `Session` to `sentinel.wrap` will raise `TypeError: Unsupported client type`. Build the client first: `boto3.client("bedrock-runtime", ...)`.
- **`invoke_model` / `invoke_model_with_response_stream` are not instrumented.** Their request/response bodies are JSON-encoded strings with a per-vendor shape (Anthropic, Cohere, AI21, Mistral, Llama, Titan all differ). A per-vendor parser registry is required before this is safe to wire up. Prefer `converse` / `converse_stream` — they cover all current Bedrock-supported model families and are universally instrumented.
- **`AccessDeniedException: You don't have access to the model with the specified model ID.`** — request access in the Bedrock console (`Model access` page) for that specific model in that specific region. Access is per-AWS-account, per-region, per-model.
- **Streaming records show `prompt_tokens=0`** — the `metadata` event arrived after iteration ended (you broke out early or the stream was force-closed). Check `record.raw_response_meta["streamed"]` is `True` and that you fully iterated the `response["stream"]`.
- **`ThrottlingException` causing retry storms** — boto3's default retry policy can amplify a tool-loop into a `retry_storm` event. That's a true positive — the rule is doing its job. Tune your boto3 retry config (`Config(retries={"max_attempts": 2})`) if the noise outweighs the signal.
