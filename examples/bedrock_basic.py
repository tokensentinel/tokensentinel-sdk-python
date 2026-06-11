"""Minimal example: wrap an AWS Bedrock client and watch for leaks.

Run:
    pip install token-sentinel[bedrock]
    # AWS credentials via env, ~/.aws/credentials, or instance profile
    export AWS_REGION=us-east-1
    python examples/bedrock_basic.py
"""

from __future__ import annotations

import os
import sys

import boto3

from token_sentinel import Sentinel

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

sentinel = Sentinel(project="example-bedrock", mode="log")


@sentinel.on_leak
def on_leak(event):
    print(
        f"\n>>> LEAK DETECTED <<<\n"
        f"  type:        {event.type}\n"
        f"  confidence:  {event.confidence:.2f}\n"
        f"  rule:        {event.rule}\n"
        f"  est_burn:    ${event.estimated_burn:.4f}\n"
        f"  suggestion:  {event.suggested_action}\n"
        f"  evidence:    {event.evidence}\n"
    )


region = os.environ.get("AWS_REGION", "us-east-1")
client = sentinel.wrap(boto3.client("bedrock-runtime", region_name=region))

# Demo 1: model_misroute — classification-shaped prompt to a frontier model.
# Bedrock charges Sonnet pricing for this call; Haiku would do the same job for
# ~190x less. The model_misroute rule should fire.
print("Demo 1: model_misroute (classification routed to Sonnet)")
client.converse(
    modelId="anthropic.claude-sonnet-4-5-v2:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"text": ("Classify this sentence as positive or negative: 'I love this movie'")}
            ],
        }
    ],
    inferenceConfig={"maxTokens": 10},
)

# Demo 2: streaming — same wrapper, same instrumentation, plus event-by-event
# token aggregation. Iterate the stream as you would normally; the wrapper
# siphons events into a usage accumulator and emits one CallRecord on
# stream end.
print("\nDemo 2: streaming converse")
response = client.converse_stream(
    modelId="anthropic.claude-sonnet-4-5-v2:0",
    messages=[{"role": "user", "content": [{"text": "Tell me one short fact about bees."}]}],
    inferenceConfig={"maxTokens": 100},
)
for event in response["stream"]:
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"].get("delta", {})
        if "text" in delta:
            sys.stdout.write(delta["text"])
            sys.stdout.flush()
print()

print(
    "\nDemo complete. Run a real agent with this client to see all six rules "
    "fire on production-like patterns."
)
