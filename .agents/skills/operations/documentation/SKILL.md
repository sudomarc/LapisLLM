---
name: documentation
description: Maintain accurate Lapis documentation, API references, model cards, changelogs, and agent skill docs from verified repository behavior.
version: 1.0.0
status: stable
category: Operations
---
# Documentation
## Purpose
Keep public and internal documentation synchronized with real behavior.
## When to use
Use when implementation, API contracts, skills, releases, models, or workflows change.
## Workflow
Inspect the implementation first; document supported behavior, examples, constraints, limitations, and migration notes; link canonical sources; update related indexes.
## Verification
Check every example against current interfaces and remove claims that cannot be reproduced or verified.
## Safety
Never document secrets, private endpoints, unverifiable capabilities, fake metrics, or security-sensitive internals.
