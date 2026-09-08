# Contributing to LapisLLM

LapisLLM prioritizes correctness, reproducibility, and small reviewable changes.

## Local setup

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Required checks

Run the following before opening a pull request:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

For changes to the training stack, also run a Tiny CPU smoke test and evaluate the resulting checkpoint:

```bash
python -m scripts.train --config configs/tiny.yaml --epochs 1 --device cpu
python -m scripts.evaluate --checkpoint checkpoints/latest.pt --device cpu
```

## Pull requests

Open changes against `main` through a pull request. Do not force-push or rewrite shared history.

Behavior changes require tests. Architecture changes require documentation and a changelog entry.

Do not commit credentials, `.env` files, checkpoints, generated corpora, or other machine-local runtime artifacts unless the repository explicitly documents them as source assets.

## Datasets

Preserve source attribution and provenance metadata. Do not silently change dataset licenses or source URLs; update the dataset registry and tests together.

## Security

Do not report credentials or other secrets in public issues. Use GitHub's private security reporting flow when available.
