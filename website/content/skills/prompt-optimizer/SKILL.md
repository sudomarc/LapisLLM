---
name: prompt-optimizer
description: Improve prompts through structured evaluation.
version: 0.1.0
status: experimental
category: Productivity
---

# Prompt Optimizer

## Purpose

Improve prompts through structured evaluation.

## When to use

Use when the task, success criteria, and evaluation examples are available.

## Inputs

- Baseline prompt.
- Representative examples.
- Desired output contract and constraints.

## Outputs

- Candidate prompt revisions.
- Evaluation comparison.
- Known trade-offs and failure cases.

## Workflow

1. Define measurable success criteria.
2. Establish a baseline.
3. Change one meaningful variable at a time.
4. Compare outputs on the same evaluation set.
5. Record trade-offs.

## Examples

Compare a baseline prompt with two constrained revisions against the same evaluation examples.

## Limitations

Prompt changes are context-dependent and can improve one task while degrading another.

## Safety

Do not use prompt optimization to evade safeguards or extract secrets.
