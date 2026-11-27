# ADR 0001 — From-scratch coding model

- Status: accepted
- Date: 2026-11-27
- Decision owner: project maintainer

## Context

Compute is limited and the project needs an independent model identity. The primary ambition is coding assistance rather than a general-purpose consumer chatbot. A third-party pretrained checkpoint would reduce training cost but would not satisfy the selected requirement to own the base-model lineage and weights.

## Decision

- Train the base model from random initialization.
- Focus initial data and evaluation on software development.
- Use Python and PyTorch for the first implementation.
- Start with a 5–10M Micro profile and scale only after measured evidence.
- Keep inference/model engineering separate from consumer chat and agent orchestration.
- Make no frontier-model parity claim without reproducible results.

## Consequences

The initial model will have limited capability. Data quality, tokens seen, training stability and evaluation matter more than an ambitious parameter target. Fine-tuning may later be applied to LapisLLM's own checkpoints without changing the base model's from-scratch origin.

## Rejected alternatives

- Third-party pretrained base: conflicts with the independent from-scratch strategy.
- Large model first: conflicts with limited compute and an unproven pipeline.
