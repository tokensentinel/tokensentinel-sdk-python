"""Synthetic demo: feed the Sentinel hand-crafted CallRecords to demonstrate
each rule firing. No real API calls — useful for fast local validation.

Run:
    cd sdk/python && pip install -e .
    python ../../examples/tool_loop_demo.py
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from datetime import datetime, timedelta, timezone

from token_sentinel import CallRecord, Sentinel

# Ensure unicode-safe stdout on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

sentinel = Sentinel(project="demo", mode="log")


@sentinel.on_leak
def show(event):
    print(
        f"  [LEAK] {event.type:18s} confidence={event.confidence:.2f} "
        f"burn=${event.estimated_burn:.4f}"
    )
    print(f"    evidence: {event.evidence}")


def make_call(
    *,
    session_id: str,
    model: str = "claude-sonnet-4-6",
    method: str = "messages.create",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    tool_calls: list | None = None,
    user_facing_output: bool = False,
    request_hash: str | None = None,
    timestamp: datetime | None = None,
    raw_request: dict | None = None,
) -> CallRecord:
    if request_hash is None:
        request_hash = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    return CallRecord(
        session_id=session_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        provider="anthropic",
        model=model,
        method=method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=120,
        request_hash=request_hash,
        tool_calls=tool_calls or [],
        user_facing_output=user_facing_output,
        raw_request=raw_request or {},
    )


def demo_tool_loop():
    print("\n=== tool_loop ===")
    sid = "tool-loop-session"
    sentinel.tracer.clear(sid)
    now = datetime.now(timezone.utc)
    similar_args = [
        {"query": "web of life game"},
        {"query": "Web of Life game"},
        {"query": '"Web of Life" game'},
    ]
    for i, args in enumerate(similar_args):
        sentinel.record_call(
            make_call(
                session_id=sid,
                tool_calls=[{"name": "web_search", "arguments": args}],
                timestamp=now + timedelta(seconds=i * 5),
            )
        )


def demo_retry_storm():
    print("\n=== retry_storm ===")
    sid = "retry-storm-session"
    sentinel.tracer.clear(sid)
    same_hash = hashlib.sha256(b"same-call").hexdigest()
    now = datetime.now(timezone.utc)
    for i in range(6):
        sentinel.record_call(
            make_call(
                session_id=sid,
                request_hash=same_hash,
                timestamp=now + timedelta(seconds=i * 3),
            )
        )


def demo_model_misroute():
    print("\n=== model_misroute ===")
    sid = "model-misroute-session"
    sentinel.tracer.clear(sid)
    sentinel.record_call(
        make_call(
            session_id=sid,
            model="claude-sonnet-4-6",
            prompt_tokens=80,
            completion_tokens=5,
            user_facing_output=True,
            raw_request={
                "messages": [
                    {
                        "role": "user",
                        "content": "Classify this sentence as positive or negative: 'great movie'",
                    }
                ]
            },
        )
    )


def demo_context_bloat():
    print("\n=== context_bloat ===")
    sid = "context-bloat-session"
    sentinel.tracer.clear(sid)
    now = datetime.now(timezone.utc)
    for i in range(8):
        sentinel.record_call(
            make_call(
                session_id=sid,
                prompt_tokens=2000 + i * 2500,
                user_facing_output=True,
                timestamp=now + timedelta(seconds=i * 30),
            )
        )


def demo_embedding_waste():
    print("\n=== embedding_waste ===")
    sid = "embedding-waste-session"
    sentinel.tracer.clear(sid)
    same_input = "user query: top 5 movies of all time"
    for _ in range(2):
        sentinel.record_call(
            make_call(
                session_id=sid,
                model="text-embedding-3-small",
                method="embeddings.create",
                prompt_tokens=12,
                completion_tokens=0,
                raw_request={"input": same_input},
            )
        )


def demo_zombie():
    print("\n=== zombie ===")
    sid = "zombie-session"
    sentinel.tracer.clear(sid)
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    sentinel.record_call(
        make_call(
            session_id=sid,
            user_facing_output=True,
            timestamp=base,
        )
    )
    for i in range(6):
        sentinel.record_call(
            make_call(
                session_id=sid,
                tool_calls=[{"name": "tool", "arguments": {"i": i}}],
                timestamp=base + timedelta(minutes=8) + timedelta(seconds=i * 10),
            )
        )


if __name__ == "__main__":
    demo_tool_loop()
    demo_retry_storm()
    demo_model_misroute()
    demo_context_bloat()
    demo_embedding_waste()
    demo_zombie()
    print("\nAll six rules demonstrated.")
