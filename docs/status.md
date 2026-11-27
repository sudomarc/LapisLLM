# Project status

**Status: documentation baseline; implementation pending.**

The current default branch was reduced to an empty source tree before this documentation foundation.

## Verified baseline

- Repository: sudomarc/LapisLLM.
- Default branch: main.
- The current main commit points to an empty Git tree.
- There is no current package manifest, model implementation, tokenizer, trainer, dataset registry, Colab notebook or CI workflow on main.
- Historical commits contain an earlier Python/PyTorch implementation and Colab scripts. Those files are not part of the current default branch and are not assumed compatible with the new direction.

## Accepted direction

- Independent decoder-only coding model.
- Random initialization and from-scratch pre-training.
- Python and PyTorch.
- Google Colab as the primary constrained experimental compute environment.
- GitHub Actions for CPU CI and lightweight validation.
- Small model first; scale after measuring throughput, memory, data quality and evaluation.

## Completion policy

Update this page when code is reintroduced. Documentation, an issue closure or a notebook run alone does not make a feature complete. Completion requires code in the default branch, relevant tests, reproducible evidence and matching documentation.

Never publish parameter counts, training throughput, benchmark scores, capabilities or compatibility claims without source artifacts and a reproducible method.
