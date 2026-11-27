# ADR 0002 — Colab-first experimental training

- Status: accepted
- Date: 2026-11-27
- Decision owner: project maintainer

## Context

Local compute is limited. Colab offers temporary runtimes but accelerator availability, usage limits and session lifetime are dynamic. GitHub Actions is appropriate for deterministic CI, not as a guaranteed long-running GPU training service.

## Decision

- Use Google Colab as the primary experimental compute environment.
- Keep reusable training code in importable Python modules; notebooks are thin.
- Use GitHub Actions for CPU CI, config validation and tiny smoke tests.
- Design all training runs for interruption and recovery.
- Use local runtime disk for staging and durable storage for complete checkpoints.
- Build manual notebook training before optional Colab API automation.

## Colab API scope

The official Colab API lists eligible runtime specifications, creates and manages runtime assignments, and returns a managed Jupyter runtime's connection information. It does not provide a one-call train-script endpoint. An automated controller must separately authenticate, connect using short-lived proxy credentials and interact with Jupyter APIs. This controller is deferred until the manual flow is reliable.

## Consequences

Initial runs are operator-triggered. A selected accelerator may not be available at request time. Experiments must make useful progress in bounded sessions. No workflow may bypass Colab restrictions or represent Colab as unlimited compute.

## Revisit when

Micro training and resume work reliably, OAuth/API access is verified for the intended account, Jupyter execution is tested, artifact storage is reliable, and resource/cost behavior is understood.
