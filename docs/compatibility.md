# Ecosystem compatibility matrix

This matrix records observed public integration state. It is not a promise that every feature works in every environment.

| Component | Current state | Public capability | Consumer |
|---|---|---|---|
| Vibe Coding Instructions | main | policy, skills, role contracts | CHAD agents / coding agents |
| LapisLLM 0.2.0 | current verified repo release | model discovery, non-streaming & streaming chat HTTP, structured error taxonomy, generation cancellation | CHAD Model Gateway |
| CHAD | current development | conversation application + Lapis HTTP client; agent runtime planned | end users |

## Lapis -> CHAD compatibility currently observed

The current Lapis HTTP serving script exposes:

- GET /v1/models (includes `context_length` and `capabilities` metadata)
- POST /v1/chat/completions (supports `stream: bool` parameter for SSE streaming chunks)

The current serving request supports `model`, `messages`, `temperature`, `max_tokens`, `top_k`, `top_p`, and `stream`.

Therefore:

- model discovery: PRESENT;
- non-streaming generation: PRESENT;
- context_length metadata: ADVERTISED in `/v1/models`;
- capabilities metadata: ADVERTISED in `/v1/models` (including `streaming` and `cancellation`);
- HTTP streaming: PRESENT (`stream: true` SSE `text/event-stream`);
- structured error taxonomy: PRESENT (`lapis.inference.errors` mapped to JSON `error` payload and HTTP status codes);
- HTTP cancellation: PRESENT (checks client disconnects and cancellation callbacks during token generation);
- mixed-precision inference: PRESENT (supports `float32`, `float16`, `bfloat16` in `LapisRuntime` and developer CLI).

CHAD must not infer unsupported capabilities from the underlying Python runtime.

## Vibe -> CHAD compatibility

Vibe Coding Instructions currently provides portable skills and agent-role governance. CHAD's planned runtime adapter will consume selected policy/role contracts.

State:

- role policy: PRESENT;
- runtime adapter: PLANNED;
- automatic synchronization of arbitrary framework files: NOT REQUIRED.

## Change protocol

When one capability changes materially:

1. update the owning repository's public contract;
2. update this matrix;
3. update the dependent adapter;
4. add or update integration tests;
5. record incompatibility or migration notes;
6. verify the public interface.

Unknown remains UNKNOWN until evidence exists.
