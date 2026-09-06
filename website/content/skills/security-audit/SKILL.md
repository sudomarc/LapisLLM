---
name: security-audit
description: Analyze source code for common security weaknesses.
version: 0.1.0
status: experimental
category: Security
---

# Security Audit

## Purpose

Analyze source code for common security weaknesses.

## When to use

Use only when the task matches this scope and the operator has the required authorization and context.

## Inputs

- Clear objective.
- Relevant source material or repository context.
- Scope and constraints for external systems.

## Outputs

- Structured, auditable findings.
- Explicit uncertainty where evidence is incomplete.

## Workflow

1. Inspect context and define the task boundary.
2. Analyze in explicit stages.
3. Separate evidence from assumptions.
4. Return actionable findings.
5. Record limitations and validation steps.

## Examples

Start from a concrete codebase and request a categorized review with evidence references.

## Limitations

Experimental specification. No implicit access to external systems.

## Safety

Operate only within authorized scope. Never bypass controls or expose secrets.
