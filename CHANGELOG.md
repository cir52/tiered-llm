# Changelog

## 0.1.0 - 2026-09-23

First public release, extracted from the Trade52 trading platform.

- Provider interface with Anthropic, Gemini, OpenAI, DeepSeek, OpenRouter, Ollama, llama.cpp and generic OpenAI-compatible implementations
- Per-provider circuit breaker (closed / open / half-open, exponential recovery backoff, `Retry-After` support, generation-stamped permits)
- `FallbackChain` with typed error handling, optional same-provider retries and per-attempt timeouts
- `Router` for per-task model selection over shared breakers, configurable from a plain mapping (TOML/YAML/JSON)
- `Cascade` for two-stage cheap-screen / expensive-decide workloads with cost statistics
- `HealthMonitor` for background recovery probes
- `DecisionLog`: append-only, buffered, crash-tolerant JSONL audit trail
- Tolerant JSON extraction (`parse_json`) for fenced, prose-wrapped, `<think>`-prefixed and Python-literal model output
- `ScriptedProvider` for offline tests and chaos drills
