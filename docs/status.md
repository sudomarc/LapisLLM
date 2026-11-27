# Project status

**Status: documentation baseline; implementation pending.**

The current default branch was reduced to an empty source tree before this documentation foundation.

## Verified baseline

- Repository: sudomarc/LapisLLM.
- Default branch: main.
- The current main commit points to an empty Git tree.
- There is no current package manifest, model implementation, tokenizer, trainer, dataset registry, local training runner or CI workflow on main.
- Historical commits contain an earlier Python/PyTorch implementation. Those files are not part of the current default branch and are not assumed compatible with the new direction.

## Accepted direction

- Independent decoder-only coding model.
- Random initialization and from-scratch pre-training.
- Python and PyTorch.
- Training on the maintainer's own PC.
- GitHub Actions for CPU CI and lightweight validation.
- Small model first; select actual training profiles only after inspecting local hardware and measuring throughput.

## Completion policy

Update this page when code is reintroduced. Documentation, an issue closure or a local training attempt alone does not make a feature complete. Completion requires code in the default branch, relevant tests, reproducible evidence and matching documentation.

Never publish parameter counts, training throughput, benchmark scores, capabilities or compatibility claims without source artifacts and a reproducible method.
