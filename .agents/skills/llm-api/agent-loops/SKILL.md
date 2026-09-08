---
name: agent-loops
description: Control autonomous LLM loops with explicit budgets, progress checks, stop conditions, tool limits, retries, and escalation to humans.
version: 1.0.0
status: stable
category: LLM API
---
# Agent Loops
## Purpose
Prevent infinite, wasteful, circular, or unsafe autonomous behavior.
## When to use
Use for multi-step agents, tool loops, orchestration, retries, and autonomous workflows.
## Workflow
Define max steps, tokens, tool calls, wall time, and cost; define success and fatal-stop conditions; track progress; detect repeated states; require approval for high-risk transitions; fail closed when budgets or evidence are exhausted.
## Verification
Test successful termination, no-progress loops, repeated tool calls, timeout, cancellation, budget exhaustion, and recovery after transient failures.
## Safety
Do not permit an agent to expand scope merely because a loop failed. Autonomous execution must remain inside explicit permissions and resource budgets.
