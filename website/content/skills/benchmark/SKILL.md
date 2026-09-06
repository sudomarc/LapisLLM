---
name: benchmark
description: Structure and analyze LLM evaluations.
version: 0.1.0
status: experimental
category: Evaluation
---

# Benchmark

## Purpose

Structure and analyze LLM evaluations.

## When to use

Use when model version, evaluation protocol, and reference data are defined.

## Inputs

- Model/version identifier.
- Benchmark suite and scoring protocol.
- Hardware/runtime details where relevant.

## Outputs

- Reproducible evaluation record.
- Metrics with provenance.
- Error analysis and limitations.

## Workflow

1. Freeze the model and evaluation protocol.
2. Run the defined suite consistently.
3. Record raw measurements and environment.
4. Analyze results without cherry-picking.

## Examples

Create an evaluation matrix that records task, version, metric, run configuration, and evidence source.

## Limitations

A benchmark result is not a complete capability profile.

## Safety

Do not fabricate or silently transform benchmark results. Preserve provenance.
