---
name: prompt-contracts
description: Treat LLM prompts as versioned interfaces with explicit goals, context, constraints, tool policy, output contracts, failure rules, and measurable success criteria.
version: 1.0.0
status: stable
category: LLM API
---
# Prompt Contracts
## Purpose
Make prompts testable, composable, and resilient instead of relying on vague instructions.
## When to use
Use when changing system prompts, agent instructions, templates, or prompt assembly.
## Workflow
Separate stable instructions from dynamic context; define inputs, outputs, constraints, stop conditions, and failure behavior; version meaningful changes; compare against a fixed evaluation set.
## Verification
Test normal, adversarial, boundary, and regression cases; inspect token growth and structured-output compatibility.
## Safety
Never use prompts to bypass security controls, extract secrets, override authorization, or treat untrusted content as trusted instructions.
