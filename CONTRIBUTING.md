# Contributing to TokenSentinel Python SDK

Thank you for your interest in contributing to the TokenSentinel Python SDK! We welcome contributions to improve the SDK's performance, stability, and support for new providers.

This document provides guidelines for setting up your development environment, running tests, and submitting contributions.

## Code of Conduct

We expect all contributors to adhere to standard respectful open-source collaboration guidelines:
- Be welcoming, respectful, and friendly.
- Focus on constructive feedback and collaboration.
- Maintain a high standard of code quality and documentation.

## Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Support for Linux, macOS, or Windows (via WSL)

### 2. Setup Development Environment
Clone the repository and install the package in editable mode with all development dependencies:

```bash
git clone https://github.com/tokensentinel/tokensentinel-sdk-python.git
cd tokensentinel-sdk-python

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with development, testing, and optional wrapper extras
pip install -e ".[dev,test,vision-perceptual]"
```

## Development Workflow

We enforce strict quality standards for formatting, style, and static typing.

### 1. Code Style and Linting
We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Run the following checks before committing code:

```bash
# Check code style and run lints
python3 -m ruff check token_sentinel tests examples

# Automatically format the code
python3 -m ruff format token_sentinel tests examples
```

### 2. Static Type Checking
All code must be statically typed and pass `mypy` check:

```bash
python3 -m mypy token_sentinel
```

### 3. Testing
TokenSentinel has a comprehensive test suite. Ensure all tests pass before proposing any changes:

```bash
# Run the full test suite
python3 -m pytest
```

You can also run tests for specific wrappers or rules:
```bash
# Example: Run Anthropic wrapper tests
python3 -m pytest tests/test_anthropic_wrapper.py

# Example: Run vision rule tests
python3 -m pytest tests/test_rules_vision_re_upload.py
```

## Guidelines for Adding Rules or Wrappers

### Adding a Waste/Leak Rule
- Rules must inherit from `token_sentinel.rules.base.Rule`.
- Implement `evaluate(session_buffer: list[CallRecord], project: str) -> Optional[LeakEvent]`.
- **Performance**: Rule evaluation runs in the hot path of wrapped LLM calls. Rules must be deterministic, run in sub-millisecond times, and perform **zero I/O operations**.
- **Scope**: Ensure the rule focuses on detecting waste patterns (e.g. loops, repetition, misroutes) rather than general content moderation.
- Include thorough unit tests in the `tests/` directory with a high level of edge-case coverage.

### Adding a Wrapper
- Instrument LLM clients by wrapping their core call methods (e.g. chat completions, embeddings).
- Mutate the live client instance rather than subclassing to preserve IDE type hints for end users.
- Implement two-level safety boundaries:
  1. Record-building or instrumentation errors must be caught and swallowed silently so that wrapper failures never crash the host application's LLM calls.
  2. Exceptions from `record_call` (such as `LeakDetected` under `mode="block"`) should propagate normally.

## Security & Privacy Audit

When contributing code, comments, or documentation:
- Ensure no internal endpoints, credentials, or proprietary configurations are checked into the codebase.
- Avoid referencing internal company directories, paths, or specifications (e.g., `docs-internal/`).
- Redact customer input text/payloads in trace records where possible (e.g., keeping only counts or hashes of embedding payloads rather than raw content).

## Submitting a Pull Request

1. Fork the repository and create your branch from `main`.
2. Make your changes and add corresponding unit tests.
3. Verify that the build, formatting, typing, and tests all pass cleanly:
   ```bash
   python3 -m ruff check token_sentinel tests
   python3 -m ruff format token_sentinel tests
   python3 -m mypy token_sentinel
   python3 -m pytest
   ```
4. Push your branch to GitHub and open a Pull Request. Provide a clear description of the problem your PR solves and your implementation approach.

## Need Help?

For questions, bug reports, or feature requests, please open an issue in the GitHub repository or reach out to us at [shakyasmreta@gmail.com](mailto:shakyasmreta@gmail.com).
