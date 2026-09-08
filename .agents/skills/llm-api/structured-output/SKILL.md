---
name: structured-output
description: Build reliable typed LLM outputs using explicit schemas, validation, schema evolution, semantic checks, and safe failure handling.
version: 1.0.0
status: stable
category: LLM API
---
# Structured Output
## Purpose
Turn model output into an explicit, versioned data contract.
## When to use
Use for JSON responses, typed extraction, tool arguments, classifiers, and machine-consumed generation.
## Workflow
Define schema; prefer native constrained output when supported; validate syntax and semantics; handle missing/extra fields; version compatible schema changes; fail explicitly when guarantees are unavailable.
## Verification
Test malformed output, boundary values, schema evolution, partial generation, provider differences, and semantic invalidity.
## Safety
Never trust schema-valid data as authorization. Validate security-sensitive semantics separately.
