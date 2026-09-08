<div align="center">

# L A P I S L L M

**An inspectable, from-scratch language model and training stack in Python/PyTorch.**

Build the tokenizer. Build the Transformer. Train it. Inspect what it learns.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Website](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-000000)](LICENSE)

[**Website**](https://sudomarc.github.io/LapisLLM/) · [**Documentation**](docs/) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](https://sudomarc.github.io/LapisLLM/roadmap/)

</div>

> [!IMPORTANT]
> **Project status — Lapis 0.2.x development.** Lapis is an experimental research and learning project focused on transparency, reproducibility, and engineering practice. It is not presented as a competitive pretrained foundation model.

> [!WARNING]
> **Security note:** `tokenizers` 0.23.2 is currently affected by CVE-2026-85670 according to the available advisory, with no upstream fixed release identified at audit time. Treat external `tokenizer.json` files as untrusted input and do not promote unreviewed tokenizer artifacts into trusted production environments.

## Overview

LapisLLM is a compact decoder-only Transformer stack built from first principles in Python and PyTorch.

The repository exposes the full pipeline:

- byte-level BPE tokenization
- causal self-attention with grouped-query attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward blocks
- optimizer and scheduler construction
- checkpointing and RNG-state preservation
- evaluation and generation
- terminal chat and OpenAI-style local serving
- experiment history and live learning observability

## Models

| Model | Status | Hidden size | Layers | Attention | Context |
|---|---|---:|---:|---:|---:|
| **Lapis Tiny / Small development config** | Experimental | 256 | 6 | 8 heads / 4 KV heads | 512 |
| Lapis 1B | Roadmap | — | — | — | — |
| Lapis 3B | Roadmap | — | — | — | — |
| Lapis 7B | Roadmap | — | — | — | — |

The active development configuration is `configs/tiny.yaml`. Exact instantiated parameter counts are reported by the training code.

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
RMSNorm → language-model head
  │
  ▼
Next-token probabilities
```

## Installation

### Runtime

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

### Development

```bash
pip install -e ".[dev]"
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

## Training

The low-level trainer is:

```bash
python -m scripts.train \
  --config configs/tiny.yaml \
  --device cpu \
  --epochs 1
```

For rapid CPU iteration:

```bash
python -m scripts.train \
  --config configs/cpu-fast.yaml \
  --device cpu \
  --epochs 1
```

The CPU-fast profile intentionally changes the architecture and cannot resume a checkpoint produced by another model shape.

## Lapis Console

The unified CLI is:

```bash
lapis
```

Training can be launched with:

```bash
lapis train
```

The console coordinates data preparation, training, learning-monitor output, checkpoint verification, experiment history and chat. It refuses to overwrite unrelated local source changes and keeps generated checkpoints/corpora outside normal source commits.

The repository does **not** rely on automatic destructive resets. Training history is lightweight source data; model checkpoints and generated corpora remain local by default.

## Learning Monitor

At configured intervals the monitor records:

- training loss and perplexity
- learning rate
- tokens seen
- fixed-prompt sample generations

Records are written as JSONL so training runs remain inspectable and reproducible.

## Evaluation

```bash
evaluate --checkpoint checkpoints/latest.pt --device cpu
```

Evaluation validates checkpoint metadata and tokenizer compatibility before constructing the model.

## Generation

```bash
generate \
  --checkpoint checkpoints/latest.pt \
  --prompt "The transformer architecture" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

Generation validates sampling parameters and truncates oversized prompts to the model context window.

## Terminal chat

```bash
lapis chat
```

Equivalent launchers are also available:

```bash
chat
lapis-chat
python -m scripts.chat
```

Chat supports session history and commands such as `/help`, `/reset`, `/stats`, `/context`, `/model`, `/temperature`, `/tokens`, `/save`, and `/exit`.

Checkpoint loading uses restricted tensor-oriented deserialization. Checkpoints and tokenizer directories should only be loaded from sources whose provenance you trust.

## Serving

```bash
serve \
  --checkpoint checkpoints/latest.pt \
  --host 127.0.0.1 \
  --port 8000
```

Endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
```

The server is intended for local experimentation and integration testing, not production deployment.

## Open training corpus

Lapis can build bounded mixed-domain corpora from public/open datasets:

```bash
python scripts/fetch_open_corpus.py --max-chars 50000000
```

For the registry-driven dataset pipeline:

```bash
pip install -e ".[data]"
python scripts/download_datasets.py --list
python scripts/download_datasets.py --profile development
```

Generated datasets stay outside Git. Provenance metadata is recorded with each preparation run. See [`docs/datasets.md`](docs/datasets.md).

## Colab

From a fresh checkout:

```bash
%cd /content/LapisLLM
!git pull --ff-only origin main
!pip install -e .
!lapis train
```

Authenticated pushes are optional and should use a GitHub/Colab secret such as `GITHUB_TOKEN`; never place a PAT in source code or Git remotes.

## Repository layout

```text
LapisLLM/
├── lapis/
│   ├── config/            # configuration models and runtime validation
│   ├── model/             # Transformer, attention, RoPE, MLP, normalization
│   ├── tokenizer/         # BPE tokenizer
│   ├── training/          # learning monitor and training helpers
│   └── ui/                # training console
├── scripts/                # CLI and automation entry points
├── configs/                # YAML experiment configurations
├── tests/                  # correctness and integration tests
├── training_history/       # lightweight verified experiment records
├── docs/                   # technical documentation
├── website/                # GitHub Pages developer site
├── CHANGELOG.md
├── AGENTS.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Validation and CI

The CI matrix targets Python 3.11, 3.12 and 3.13. It runs tests, CPU training/evaluation smoke checks and Ruff. Security is checked in a separate least-privilege job with `pip-audit` and working-tree/Git-history secret scans so lint failures cannot suppress those checks.

The website workflow validates and builds the site on pull requests. Pages deployment requires `pages: write` and `id-token: write` only in the deployment job.

## Security and Git policy

Do not commit credentials, `.env` files, checkpoints, generated corpora or machine-local runtime artifacts.

Changes to `main` should go through pull requests and required checks. The current repository still requires a GitHub administrator to enable the branch ruleset because the connected integration cannot modify repository protection settings.

## Current limitations

Lapis does not yet provide:

- exact mid-epoch data-loader/sampler replay
- mixed-precision training
- distributed training
- efficient KV-cache generation
- large-scale pretraining
- broad benchmark suites
- production-grade web-scale data processing

These limitations are explicit roadmap items.

## Roadmap

### Phase 1 — Foundations

- [x] Decoder-only Transformer
- [x] ByteLevel BPE tokenizer
- [x] RoPE
- [x] GQA
- [x] RMSNorm
- [x] SwiGLU
- [x] Checkpoint save/load
- [x] Generation
- [x] Terminal chat

### Phase 2 — Training reliability

- [x] CPU-fast profile
- [x] Learning Monitor
- [x] Experiment history
- [x] Checkpoint validation gates
- [x] Colab workflow
- [ ] Exact mid-epoch replay
- [ ] Stronger training resume semantics

### Phase 3 — Scale and evaluation

- [ ] Reproducible benchmark suite
- [ ] Larger model configurations
- [ ] Mixed precision
- [ ] Efficient attention / KV cache
- [ ] Distributed training
- [ ] Model release artifacts

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).

Required local checks:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

## License

LapisLLM is released under the MIT License. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{lapisllm2026,
  author = {sudomarc},
  title = {LapisLLM: A From-Scratch Decoder-Only Transformer Language Model},
  year = {2026},
  url = {https://github.com/sudomarc/LapisLLM}
}
```
