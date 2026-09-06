---
name: dataset-cleaner
description: Identify and clean problematic training data.
version: 0.1.0
status: experimental
category: Data
---

# Dataset Cleaner

## Purpose

Identify and clean problematic training data.

## When to use

Use when the dataset source, licensing constraints, and quality criteria are known.

## Inputs

- Dataset samples.
- Cleaning rules.
- License and provenance metadata.

## Outputs

- Cleaned records or a review queue.
- Transformation log.
- Data-quality notes.

## Workflow

1. Validate source provenance and constraints.
2. Detect malformed, duplicate, low-quality, or policy-problematic records.
3. Apply deterministic transformations where appropriate.
4. Preserve an auditable manifest.

## Examples

Build a deterministic cleaning report that records each rule, count, and affected sample class.

## Limitations

Cleaning rules can remove useful data or miss subtle quality issues; review remains necessary.

## Safety

Never ingest data of unknown provenance when licensing is required. Keep sensitive data protected.
