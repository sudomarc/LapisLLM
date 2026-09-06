---
name: web-tester
description: Assist with authorized web application testing.
version: 0.1.0
status: experimental
category: Security
---

# Web Tester

## Purpose

Assist with authorized web application testing.

## When to use

Use only against applications and environments within explicit testing scope.

## Inputs

- Application URL or source.
- Authorized test scope.
- Expected behavior and constraints.

## Outputs

- Structured observations.
- Reproduction context where appropriate.
- Risk, scope, and evidence notes.

## Workflow

1. Confirm scope and test boundaries.
2. Inventory relevant application surfaces.
3. Test the defined behavior safely.
4. Preserve evidence and stop at scope boundaries.

## Examples

Create an authorized regression test plan for authentication and input-validation behavior.

## Limitations

This package is a planning/review layer, not an authorization mechanism or autonomous testing agent.

## Safety

Never target systems without authorization. Do not bypass access controls or expose sensitive data.
