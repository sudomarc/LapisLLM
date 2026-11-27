# ADR 0002 — Local PC as the training environment

- Status: accepted
- Date: 2026-11-27
- Decision owner: project maintainer

## Context

The maintainer has selected local-PC training and does not want the project to rely on hosted notebook runtimes or external GPU sessions. The exact hardware has not yet been inventoried, so model and batch profiles cannot be finalized responsibly.

## Decision

- Train LapisLLM on the maintainer's own PC.
- Use Python and PyTorch.
- Implement hardware discovery before freezing the model's compute profile.
- Support CPU for correctness and tiny experiments.
- Implement NVIDIA CUDA as the first intended accelerator path when available and compatible.
- Treat AMD ROCm, Apple MPS and other backends as unsupported until verified.
- Store datasets, run logs and checkpoints on local storage outside Git.
- Use GitHub Actions for CPU CI and bounded smoke tests only.
- Design training to survive Ctrl+C, process crash and PC restart.

## Consequences

No GPU model or training duration is promised before hardware discovery. CPU-only operation may be too slow for meaningful pre-training beyond tiny experiments. Every scale-up must be justified by measured peak memory, tokens per second, data availability and evaluation improvement.

## Revisit when

The project may consider remote or distributed training only through a new explicit architecture decision supported by a compute/cost plan. This is not part of the current scope.
