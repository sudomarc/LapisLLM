<div align="center">

# L A P I S L L M

**An inspectable, from-scratch language model and training stack in Python/PyTorch.**

Build the tokenizer. Build the Transformer. Train it. Inspect what it learns.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Website](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-000000)](LICENSE)

[**Website**](https://sudomarc.github.io/LapisLLM/) · [**Documentation**](docs/) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](https://sudomarc.github.io/LapisLLM/roadmap/)

</div>

> [!IMPORTANT]
> **Project status — Lapis 0.1.1.** Lapis is an experimental research and learning project. It is designed for transparency, reproducibility, and engineering practice; it is **not** presented as a competitive pretrained foundation model. Capability and benchmark claims are intentionally withheld until reproducible evaluation exists.

## Overview

LapisLLM is a compact decoder-only Transformer stack built from first principles in Python and PyTorch.

The project focuses on the parts of a language model that are usually hidden behind a framework boundary:

- byte-level BPE tokenization
- causal self-attention with grouped-query attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward blocks
- optimizer and scheduler construction
- checkpointing and RNG state preservation
- evaluation and generation
- terminal chat and OpenAI-style serving
- experiment history and live learning observability

The goal is not to imitate a particular commercial model. The goal is to make the full pipeline understandable, testable, and extensible.

## Why Lapis exists

Most LLM projects expose an inference API and hide the training stack.

Lapis takes the opposite approach: the repository is the product surface.

You can inspect the model architecture, tokenize your own corpus, run a training job, watch generations change during optimization, reload the resulting checkpoint, evaluate it, and chat with the trained model from the terminal.

## Models

Lapis currently uses a small development configuration while the training and evaluation stack are being hardened.

| Model | Status | Hidden size | Layers | Attention | Context |
|---|---|---:|---:|---:|---:|
| **Lapis Tiny / Small development config** | Experimental | 256 | 6 | 8 heads / 4 KV heads | 512 |
| Lapis 1B | Roadmap | — | — | — | — |
| Lapis 3B | Roadmap | — | — | — | — |
| Lapis 7B | Roadmap | — | — | — | — |

The current configuration is stored in [`configs/tiny.yaml`](configs/tiny.yaml). Exact instantiated parameter counts are reported by the training code rather than hard-coded in this README.

## Architecture

```text
Text
  │
  ▼
ByteLevel BPE tokenizer
  │
  ▼
Token embeddings
  │
  ▼
┌────────────────────────────────────────────┐
│ Transformer block × N                      │
│                                            │
│  RMSNorm → GQA causal attention + RoPE     │
│        ↓                                   │
│  residual                                  │
│        ↓                                   │
│  RMSNorm → SwiGLU MLP                      │
│        ↓                                   │
│  residual                                  │
└────────────────────────────────────────────┘
  │
  ▼
RMSNorm
  │
  ▼
Language-model head
  │
  ▼
Next-token probabilities
```

### Tokenizer

Lapis uses a Hugging Face `tokenizers` BPE backend with ByteLevel pre-tokenization and NFKC normalization. The tokenizer is versioned and stored alongside checkpoints so inference can verify that the vocabulary and tokenizer implementation match the trained weights.

### Attention

The attention implementation is causal and supports grouped-query attention by projecting fewer key/value heads and repeating them across query groups. Rotary frequencies are stored separately from model parameters and are kept in `complex64` during model dtype conversion.

### Objective

Training uses next-token prediction with cross-entropy loss and supports gradient accumulation, gradient clipping, linear warmup, and cosine decay.

## Quickstart

### Local

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the correctness suite:

```bash
pip install -e ".[dev]"
python -m pytest
ruff check .
```

### User mode

User mode is the default runtime surface. It exposes inference-oriented commands without requiring access to the training workflow:

```bash
lapis chat
lapis serve
lapis system
```

### Developer mode

Developer tooling is explicitly namespaced under `lapis dev`:

```bash
lapis dev train --config configs/local-dev.yaml
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "The future of computing is"
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
```

See [`docs/modes.md`](docs/modes.md) for the USER/DEV boundary.

## Training

The canonical training path is:

```bash
lapis dev train --config configs/tiny.yaml --device cpu
```

For non-interactive environments, add `--no-tui`.

## Evaluation and generation

```bash
lapis dev evaluate --checkpoint checkpoints/latest.pt --device cpu
lapis dev generate "A language model learns by" --checkpoint checkpoints/latest.pt --device cpu
```

## Security

Inference checkpoint loading uses restricted PyTorch deserialization. Training-resume paths may require broader compatibility and should only consume checkpoints from trusted sources. See [`SECURITY.md`](SECURITY.md).

The CI security job runs dependency auditing and tracked-file/Git-history credential scans.

## Development

The repository uses GitHub Actions for Python test matrices, training/evaluation smoke tests, Ruff validation, package-build checks, dependency auditing, and secret scanning. The website workflow separately validates and deploys the static site.

Before opening a PR, run:

```bash
pip install -e ".[dev]"
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
```

## Project structure

```text
lapis/          reusable model, config, inference, UI, training code
scripts/        packaged CLI entry points
configs/        training, data, and user configuration
runs/           local runtime state (ignored)
checkpoints/    local checkpoints (ignored)
tests/          regression and integration tests
website/        static project website
```

## License

LapisLLM is released under the MIT License. See [`LICENSE`](LICENSE).
