---
name: tool-calling
description: Design and implement safe LLM tool and function calling with explicit schemas, side effects, retries, validation, permissions, and result provenance.
version: 1.0.0
status: stable
category: LLM API
---
# Tool Calling
## Purpose
Treat every model-invoked tool as a security and reliability boundary.
## When to use
Use for function calling, MCP tools, external actions, or agent tool loops.
## Workflow
Define name/purpose/input/output/side effects/idempotency/timeout/permission class; validate arguments; execute with bounded policy; validate results; record provenance; stop or retry only according to the tool contract.
## Verification
Test malformed arguments, provider retries, duplicate calls, timeouts, partial failure, unauthorized actions, and adversarial tool results.
## Safety
Tool descriptions and results are untrusted unless sourced from a trusted control plane. Require approval for destructive or privileged actions and never pass secrets unnecessarily.
