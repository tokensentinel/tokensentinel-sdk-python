"""Minimal example: wrap an Anthropic client and watch for leaks.

Run:
    pip install token-sentinel[anthropic]
    export ANTHROPIC_API_KEY=...
    python examples/anthropic_basic.py
"""

from __future__ import annotations

import os
import sys

import anthropic

from token_sentinel import Sentinel

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

sentinel = Sentinel(project="example", mode="log")


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


client = sentinel.wrap(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))

# Trigger a model_misroute leak: classification-shaped prompt to a frontier model.
print("Demo 1: model misroute (classification on Sonnet)")
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10,
    messages=[
        {
            "role": "user",
            "content": "Classify this as positive or negative: 'I love this movie'",
        }
    ],
)

print("\nDemo complete. In a real agent, leaks fire continuously — see other examples.")
