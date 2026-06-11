# Installation

TokenSentinel is published to PyPI as `token-sentinel`. The core package has zero dependencies — provider SDKs are pulled in via extras, so you only install what you use.

## Python version

TokenSentinel requires **Python 3.10 or later**. It is tested on 3.10, 3.11, and 3.12. Earlier versions will not work — the SDK uses `dataclass` keyword-only parameters, structural pattern matching in a couple of internal helpers, and the modern `from __future__ import annotations` lazy-eval semantics.

## Core install

```bash
pip install token-sentinel
```

The core package gives you the `Sentinel` class, the rules engine, the tracer, and the type definitions (`CallRecord`, `LeakEvent`, `LeakDetected`). It will not, on its own, instrument any provider client — for that you need at least one of the provider extras below.

## Provider extras

Install the extras for the providers you use. You can combine them.

| Extra | Pulls in | Use when |
|---|---|---|
| `[anthropic]` | `anthropic>=0.39.0` | You call `anthropic.Anthropic` or `anthropic.AsyncAnthropic`. |
| `[openai]` | `openai>=1.50.0` | You call `openai.OpenAI` or `openai.AsyncOpenAI`. Also covers DeepSeek, Together, Fireworks, Groq, OpenRouter, Anyscale, Mistral, Perplexity, vLLM, Ollama, TGI, LM Studio — anything OpenAI-compatible. |
| `[gemini]` | `google-genai>=1.0.0` | You call `google.genai.Client` (direct API or Vertex backend). |
| `[bedrock]` | `boto3>=1.35.0` | You call `boto3.client("bedrock-runtime")`. |
| `[all]` | All of the above | You don't know yet, or you support multiple providers in one app. |

```bash
# Single provider
pip install token-sentinel[anthropic]

# Multi-provider
pip install token-sentinel[anthropic,openai,bedrock]

# Everything
pip install token-sentinel[all]
```

Per the table, the `[openai]` extra is enough to cover every OpenAI-compatible provider — DeepSeek, Together, Groq, vLLM, Ollama, etc. — because they all use the official `openai` Python SDK with a custom `base_url`. See [Providers](./05-providers.md) for the matrix.

## Embeddings extra (V1, optional)

```bash
pip install token-sentinel[embeddings]
```

This pulls in `sentence-transformers` and `numpy`. It is **not required** for V0 detection — the V0 `tool_loop` and `retrieval_thrash` rules use a pure-Python TF-IDF char-n-gram similarity that needs no model dependency.

The embeddings extra is the opt-in path for V1 sentence-transformer-based semantic similarity (model: `all-MiniLM-L6-v2`). It is forward-looking — V1 is on the roadmap, not shipped yet. You can install it now without changing your detection behavior; the rules continue to use the deterministic V0 metric until you opt in.

## Dev extra

```bash
pip install token-sentinel[dev]
```

Pulls in `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-cov`, `ruff`, and `mypy`. You only need this if you are contributing to TokenSentinel itself.

## Verifying your install

Run this one-liner to confirm the package imports cleanly:

```bash
python -c "from token_sentinel import Sentinel; print(Sentinel.__module__)"
```

Expected output:

```
token_sentinel.sentinel
```

To verify a specific provider extra is installed and the wrapper dispatches correctly, run the corresponding line:

```bash
# Anthropic
python -c "import anthropic; from token_sentinel import Sentinel; Sentinel(project='check').wrap(anthropic.Anthropic(api_key='test')); print('anthropic ok')"

# OpenAI (no API call — just verifies wrapping)
python -c "import openai; from token_sentinel import Sentinel; Sentinel(project='check').wrap(openai.OpenAI(api_key='test')); print('openai ok')"

# Gemini
python -c "from google import genai; from token_sentinel import Sentinel; Sentinel(project='check').wrap(genai.Client(api_key='test')); print('gemini ok')"

# Bedrock — needs AWS credentials configured but won't make a network call here
python -c "import boto3; from token_sentinel import Sentinel; Sentinel(project='check').wrap(boto3.client('bedrock-runtime', region_name='us-east-1')); print('bedrock ok')"
```

Each line should print `<provider> ok` and exit cleanly. If you see `Unsupported client type`, the wrapper dispatch did not recognize the client — most often because the SDK is on an old version or you are passing a subclass. See [Troubleshooting](./08-troubleshooting.md).

## Upgrading

```bash
pip install --upgrade token-sentinel
```

The public API (`Sentinel`, `wrap`, `on_leak`, `record_call`, `LeakEvent`, `CallRecord`, `LeakDetected`) follows semantic versioning. Anything starting with `_` or living under `token_sentinel.tracer` / `token_sentinel.rules` is internal and may change between minor versions — pin if you import those directly.

## Troubleshooting installation

**`No matching distribution found for token-sentinel`** — your `pip` is too old or you are on a Python version older than 3.10. Run `pip install --upgrade pip` and check `python --version`.

**`No matching distribution found for token-sentinel[anthropic]`** — extras syntax requires `pip>=20.3`. Upgrade pip first.

**`anthropic.Anthropic` import works but `Sentinel.wrap(...)` raises `Unsupported client type`** — you are likely on an old `anthropic` version that was renamed since. The SDK detects clients by `type(client).__module__`, so module renames are the usual culprit. Upgrade `anthropic` to `>=0.39.0`.

Once the basic install works, head to [Quickstart](./02-quickstart.md).
