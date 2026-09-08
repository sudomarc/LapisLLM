---
name: provider-compatibility
description: Normalize differences across LLM providers and local runtimes using capability discovery, adapters, explicit contracts, and verified fallbacks.
version: 1.0.0
status: stable
category: LLM API
---
# Provider Compatibility
## Purpose
Keep application behavior stable while provider APIs evolve.
## When to use
Use for provider adapters, model routing, migrations, fallbacks, or capability negotiation.
## Workflow
Inventory capabilities for messages, tools, schemas, streaming, multimodality, context, errors, usage, and limits; map them to a canonical Lapis contract; isolate provider-specific code; make unsupported features explicit.
## Verification
Run provider-specific contract tests and cross-provider parity tests; verify error mapping, usage accounting, streaming, tool calls, and fallback behavior.
## Safety
Never silently downgrade security, data isolation, tool permissions, or output guarantees during fallback.
