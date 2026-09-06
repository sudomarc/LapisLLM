---
name: pr-engineer
description: Review pull requests and identify bugs, regressions and maintainability problems.
version: 0.1.0
status: experimental
category: Development
---

# PR Engineer

## Purpose

Review pull requests and identify bugs, regressions and maintainability problems.

## When to use

Use when source diffs, repository context, and expected behavior are available.

## Inputs

- Pull request diff.
- Relevant tests and project conventions.
- Expected behavior or issue context.

## Outputs

- Findings ranked by impact.
- Evidence references.
- Suggested fixes or follow-up tests.

## Workflow

1. Inspect the full diff and surrounding code.
2. Check behavior, regressions, edge cases, and maintainability.
3. Compare changes to existing tests and contracts.
4. Report evidence before recommendations.

## Examples

Review a pull request for behavioral regressions and request tests where coverage is missing.

## Limitations

Static review cannot establish runtime behavior for all environments.

## Safety

Treat credentials, private data, and security-sensitive material as confidential.
