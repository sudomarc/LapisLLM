# Contributing to LapisLLM

LapisLLM is an experimental from-scratch coding-model project. Contributions should improve correctness, reproducibility, data provenance, model behavior or developer usability.

## Before starting

Read CODE_OF_CONDUCT.md, SECURITY.md, AGENTS.md and the relevant documents in docs/. Search existing issues before opening a new one. For substantial work, agree on scope and acceptance criteria in an issue first.

## Current state

The default branch is a documentation-first clean slate. Do not assume that installation, training, inference or hardware diagnostics are implemented. Follow the roadmap and inspect the actual repository before writing code.

## Development setup

Exact install commands will be added once the Python package manifest and supported dependency matrix are implemented. Until then, do not copy unverified commands from historical revisions or claim that the package is installable.

## Engineering expectations

- Keep changes focused and reviewable.
- Trace the implementation before modifying behavior.
- Prefer explicit configuration, typed boundaries and useful errors.
- Add regression tests for bug fixes.
- Keep normal CI CPU-only and independent of the maintainer's PC, installed GPU drivers and local datasets.
- Training must work locally and detect actual hardware rather than assume a device.
- Keep large corpora, runs and checkpoints outside Git.
- Update documentation and changelog when behavior or contracts change.
- Never commit credentials, private source code or credential-bearing logs.

## Dataset contributions

A dataset proposal must include canonical source, immutable revision or retrieval method, license evidence, intended use, redistribution constraints, attribution, provenance fields, filtering plan and known risks. A public URL alone is not sufficient. See docs/data-policy.md.

## Model and experiment contributions

Every experiment should define a falsifiable hypothesis, controlled variables, compute budget, dataset manifest, configuration, baseline, evaluation method and stop criteria. Record local hardware, software versions, peak memory, tokens/second and wall-clock time. Do not publish private hardware identifiers or sensitive host data.

Model artifacts must include model/tokenizer identity, source revision, training lineage, evaluation report and checksums.

## Pull requests

Submit normal changes through a pull request. Include the problem, scope, linked issue, acceptance criteria, tests/commands actually run, data and model provenance, risks, skipped checks and remaining uncertainty.

CI must be inspected on the final PR commit. A later fix invalidates earlier test evidence for changed code.

## Licensing

Repository software is under MIT. The license for model weights is a separate decision and remains pending; see MODEL_LICENSE.md. Contributors must have the rights needed to submit material under the applicable license.
