---
name: observability
description: Instrument Lapis agents and LLM APIs with request correlation, traces, metrics, structured logs, token usage, cost, latency, retries, and failure provenance.
version: 1.0.0
status: stable
category: Operations
---
# Observability
## Purpose
Make agent and API behavior diagnosable without exposing sensitive data.
## When to use
Use for logging, tracing, metrics, latency, token accounting, cost, incidents, and agent-run history.
## Workflow
Assign request/trace/span identifiers; record model/provider/prompt/skill versions, tool calls, timings, usage, retries, and terminal state; redact sensitive data; preserve causality across agent steps.
## Verification
Test correlation IDs, redaction, metric correctness, failure traces, sampling, and absence of cross-request leakage.
## Safety
Never log API keys, cookies, hidden prompts, private user data, or raw secrets. Observability must not change authorization boundaries.
