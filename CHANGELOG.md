# Changelog

All notable changes to the TokenSentinel Python SDK are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-11

Initial standalone release of the TokenSentinel Python SDK (`token-sentinel`), decoupled from the monorepo.

### Changed
- Decoupled from the monorepo into a standalone repository (`tokensentinel-sdk-python`).
- Scrubbed pre-release version labels and developmental cycle annotations (e.g., V0.x, V1.x, V2.x, cycle comments) from source code comments and docstrings.
- Updated package metadata (version `1.0.0` and package URLs pointing to the standalone GitHub repository `https://github.com/tokensentinel/tokensentinel-sdk-python` in `pyproject.toml` and `__init__.py`).

## [0.19.0] — 2026-05-22

### Added
- Added CrewAI / AutoGen / Pydantic AI OpenTelemetry enrichers (`token_sentinel/enrichers/otel.py`). Implements a `TokenSentinelSpanProcessor` that extracts `gen_ai.*` semantic convention attributes and maps framework-specific metadata to tags.
- Added optional `[otel]` extra for OpenTelemetry dependencies.
- Added Whisper streaming audio duration fallback in `wrappers/openai.py` (`_probe_streaming_audio_duration`).

## [0.18.0] — 2026-05-22

### Added
- Added tag-based session cost attribution. Users can now pass tags via `Sentinel.session(tags={"team": "growth"})` with validation (keys must match `team`, `feature`, `customer`, `environment`, or `version`; values restricted to max 64 characters and alphanumeric/dash/dot symbols).
- Added the `Session` helper class to wrap active sessions and expose convenience methods for logging and manual tracking.
- Wired tags serialization and validation to `CloudClient` and event payload builders.

## [0.17.0] — 2026-05-22

### Added
- Added the `repair_loop` rule to identify conversational-repair waste (detects correcting user queries coupled with high agent output similarity via TF-IDF char-3-grams).
- Added a LangChain `BaseCallbackHandler` enricher (`token_sentinel/enrichers/langchain.py`) under the `[langchain]` extra.
- Added perceptual image hashing for the `vision_re_upload` rule using `imagehash.phash` under the `[vision-perceptual]` extra.
- Added brand aliases (`WasteEvent = LeakEvent`, `WasteDetected = LeakDetected`, and `on_waste = on_leak`).
- Added a unified `[all]` extra to install all optional integration dependencies.

## [0.16.0] — 2026-05-13

### Changed
- Relicensed the SDK under the Apache-2.0 license.
- Swept package URLs and emails to target `tokensentinel.dev` and `hello@tokensentinel.dev`.

### Added
- Added the `voice_switching_loop` rule to flag ElevenLabs voice changes on identical text inputs.
- Added the `rerank_thrash` rule to flag duplicate Cohere reranking requests.
- Added Whisper audio duration fallback using the `mutagen` library under the optional `[audio-metadata]` extra.

## [0.15.0] — 2026-05-13

### Added
- Added SDK wrappers for ElevenLabs TTS, OpenAI Whisper, and Cohere.
- Added the `audio_multichannel_doubling` rule.
