---
name: llm-api-architecture
description: Design and modify Lapis LLM APIs around stable contracts, transport isolation, model adapters, validation, streaming, tools, errors, and compatibility.
version: 1.0.0
status: stable
category: LLM API
---
# LLM API Architecture
## Purpose
Keep the public API independent from Transformer internals.
## When to use
Use for endpoints, schemas, API compatibility, serving, adapters, or client-facing changes.
## Workflow
Map request validation → policy → routing → model/runtime → tools → output validation → response/stream → telemetry. Reuse existing boundaries before adding abstractions.
## Verification
Test valid/invalid requests, error contracts, streaming, backwards compatibility, and model/runtime isolation.
## Safety
Do not leak internal stack traces, secrets, hidden prompts, tenant data, or unsupported capabilities through the public API.
