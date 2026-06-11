"""Minimal example: wrap a Google Gemini ``google-genai`` client and watch for leaks.

Run:
    pip install token-sentinel[gemini]
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
    python examples/gemini_basic.py

The wrapper supports both the direct Gemini API path
(``genai.Client(api_key=...)``) and the Vertex AI backend
(``genai.Client(vertexai=True, project=..., location=...)``) — the dispatch
key is the client class's ``__module__`` starting with ``google.genai``.
"""

from __future__ import annotations

import os
import sys

from google import genai

from token_sentinel import Sentinel

# Ensure unicode-safe stdout on Windows cp1252 consoles.
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


api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit(
        "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment to run this example."
    )

client = sentinel.wrap(genai.Client(api_key=api_key))

# Trigger a model_misroute leak: classification-shaped prompt to a frontier model.
print("Demo 1: model misroute (classification on Gemini 2.5 Pro)")
client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Classify this as positive or negative: 'I love this movie'",
)

# Trigger a retry_storm: same call repeatedly.
print("\nDemo 2: retry_storm (5 identical calls)")
for _ in range(5):
    client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is 2+2?",
        _sentinel_session_id="retry-demo",
    )

print("\nDemo complete. In a real agent, leaks fire continuously — see other examples.")
