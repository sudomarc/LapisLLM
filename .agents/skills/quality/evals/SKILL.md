---
name: evals
description: Evaluate agent, prompt, tool, model, and API changes with fixed datasets, regression checks, measurable success criteria, safety cases, cost, and latency.
version: 1.0.0
status: stable
category: Quality
---
# Evals
## Purpose
Replace subjective claims with repeatable evidence.
## When to use
Use for prompt/model/tool/API changes and agent behavior changes.
## Workflow
Define baseline, representative test set, success criteria, failure categories, and thresholds; change one material variable at a time; compare candidate against baseline; record trade-offs.
## Verification
Measure correctness, safety, tool accuracy, structured-output validity, regressions, latency, token usage, and cost when relevant.
## Safety
Never fabricate scores, benchmark results, or pass/fail claims. Preserve evaluation provenance and avoid leaking test secrets into prompts.
