---
name: api-resilience
description: Design resilient LLM API calls with typed errors, timeouts, idempotency, bounded retries, backoff, rate-limit handling, failover, and explicit partial-failure semantics.
version: 1.0.0
status: stable
category: LLM API
---
# API Resilience
## Purpose
Recover from transient faults without duplicating side effects or hiding real failures.
## When to use
Use for retries, timeouts, rate limits, provider failures, failover, and network instability.
## Workflow
Classify errors; define timeout and retry budget; verify idempotency before retry; use exponential backoff with jitter; honor provider limits; map failures to stable API errors; fail closed when recovery is unsafe.
## Verification
Test 429s, timeouts, connection resets, partial responses, duplicate delivery, cancellation, exhausted retries, and provider failover.
## Safety
Never retry privileged or destructive operations blindly. Do not convert permanent failures into misleading success responses.
