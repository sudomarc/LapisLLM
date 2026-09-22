# ADR 0001: Keep LapisLLM, CHAD, and Vibe Coding Instructions separate

## Context

The three repositories now form one AI ecosystem. Each has a different responsibility and release/security boundary.

## Decision

Keep three repositories:

- Vibe Coding Instructions: portable agent governance, skills, evidence rules, token economics, self-improvement policy, and reusable role contracts.
- CHAD: user-facing product, agent runtime, orchestration, tools, memory, files, permissions, and model gateway.
- LapisLLM: model architecture, tokenization, training, checkpoints, inference, generation, evaluation, and serving primitives.

The dependency direction for model use is CHAD -> LapisLLM.

Vibe Coding Instructions can be consumed as policy by CHAD agents, but it is not a runtime dependency that executes product tasks.

## Consequences

Positive:

- model and product evolve independently;
- agent/runtime concerns do not contaminate model-engine code;
- governance can be reused by coding agents and CHAD agents;
- provider/model replacement remains possible.

Costs:

- cross-repository API compatibility becomes an explicit maintenance obligation;
- contract changes require coordinated documentation and integration tests.

## Verification / follow-up

Maintain docs/ecosystem.md and docs/roadmap.md in Lapis and CHAD, and docs/ecosystem.md plus docs/agent-runtime-contract.md in Vibe Coding Instructions.
