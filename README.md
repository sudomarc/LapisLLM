# LapisLLM

> A compact, end-to-end implementation of an LLM stack in Python—from tokenizer to serving.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main)](https://github.com/sudomarc/LapisLLM/actions)

## Overview

**Lapis is a from-scratch decoder-only Transformer language model.**

LapisLLM provides a clean, modular, and reproducible implementation of a complete LLM pipeline. Each component—tokenizer, data processing, training, and inference—is standalone, well-tested, and configurable. The project prioritizes **transparency** (all code visible, no binary dependencies), **reproducibility** (versioned artifacts and deterministic training), and **developer ergonomics** (configuration-driven workflows).

Perfect for:
- Learning how LLMs work end-to-end
- Running experiments with custom datasets and models
- Serving local inference with a lightweight API
- Building on top of a well-structured foundation

## Current Status

**LAPIS 0.1.1 — CORRECTNESS**

> The core Transformer training pipeline is now correctness-focused and covered by automated component tests. The next milestone is validation through reproducible training experiments and tiny-dataset overfitting.

## Quick Start

### Prerequisites
- Python 3.11 or later
- ~2GB+ disk space for dependencies and example configs

### Installation

```bash
# Clone the repository
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### First Run

```bash
# Copy and customize environment
cp .env.example .env
# Edit .env as needed for your hardware/setup

# Run tests to verify installation
pytest -q

# Train a tokenizer on example data
python scripts/train_tokenizer.py --config configs/tiny.yaml

# Prepare data (packing, manifests)
python scripts/prepare_data.py --config configs/development.yaml

# Train a model
python scripts/train.py --config configs/development.yaml

# Generate from checkpoint
python scripts/generate.py --checkpoint checkpoints/latest --prompt "Hello, world"
```

## Core Features

### Architecture

- **Decoder-only Transformer** (GPT-style)
- **RMSNorm** for efficient layer normalization
- **Rotary Positional Embeddings (RoPE)** for rotation-based position encoding
- **Grouped Query Attention (GQA)** for memory efficiency
- **SwiGLU** feed-forward networks
- **Causal self-attention** for language modeling
- Configurable model dimensions (hidden size, layers, heads)

### 🔤 Tokenizer
- BPE ByteLevel tokenizer with configurable vocabulary size
- Serializable tokenizer artifacts linked to model checkpoints
- Clean separation between training and inference pipelines

### 📊 Data Pipeline
- **Ingestion**: Download and parse raw datasets
- **Cleaning**: Remove duplicates, filter by quality metrics
- **Packing**: Efficient sequence packing with cross-document boundaries
- **Manifests**: Deterministic data references for reproducible training runs

### 🎓 Training
- Correct causal target alignment (no double-shifting)
- Padding-aware loss masking (uses -100 for ignored positions)
- Linear warmup followed by cosine decay schedule
- Comprehensive checkpointing (weights, optimizer, scheduler, RNG state)
- Tensorboard logging and evaluation hooks
- Gradient accumulation and gradient clipping

### 🚀 Inference & Serving
- Multiple generation strategies (greedy, temperature sampling, top-k/top-p)
- Interactive chat interface
- Local HTTP API with streaming support
- Export to safetensors format

## Usage

### Common Commands

| Task | Command |
|------|----------|
| Train tokenizer | `python scripts/train_tokenizer.py --config configs/tiny.yaml` |
| Prepare data | `python scripts/prepare_data.py --config configs/development.yaml` |
| Train model | `python scripts/train.py --config configs/development.yaml` |
| Evaluate | `python scripts/evaluate.py --checkpoint checkpoints/latest` |
| Generate (single) | `python scripts/generate.py --checkpoint checkpoints/latest --prompt "Hello"` |
| Interactive chat | `python scripts/chat.py --checkpoint checkpoints/latest` |
| Serve API | `python scripts/serve.py --checkpoint checkpoints/latest --port 8080` |

### Configuration

All runs are controlled by YAML config files in `configs/`. Provided configs:
- **tiny.yaml** — Small model for rapid testing (~10M parameters)
- **development.yaml** — Medium model for iteration (~100M parameters)
- **small.yaml** — Realistic small model (~300M parameters)
- **base.yaml** — Larger model for production workloads

Edit configs to adjust:
- Model architecture (hidden size, layers, attention heads)
- Training hyperparameters (learning rate, batch size, warmup)
- Data sources and processing options
- Checkpoint and logging paths

See `src/lapis/config/` for all available options.

## Project Structure

```
LapisLLM/
├── src/lapis/
│   ├── tokenizer/          # Tokenizer training and inference
│   ├── data/               # Data loaders, cleaners, packers
│   ├── model/              # Transformer architecture
│   ├── training/           # Trainer, optimizers, checkpointing
│   ├── inference/          # Generation, sampling, KV cache
│   ├── export/             # safetensors / GGUF exporters
│   ├── api/                # Serving API and utilities
│   └── config/             # Configuration dataclasses
├── scripts/                # CLI entrypoints
├── configs/                # Example run configurations
├── docs/                   # Design documentation
├── tests/                  # Unit and integration tests
└── README.md               # This file
```

## Checkpoints

Checkpoints are versioned, reproducible snapshots containing:
- **Model weights** (float32)
- **Optimizer & scheduler state** (for resuming training)
- **Global step** (iteration counter)
- **RNG seed state** (for deterministic resumption)
- **Tokenizer reference** (vocabulary and BPE merges)

All checkpoints use `safetensors` by default for safety and portability.

## Reproducibility

LapisLLM is built for reproducible research:
- All hyperparameters in YAML configs
- Deterministic data ordering via manifests
- Seeded RNG state in checkpoints
- No external binary model dependencies
- Documented architecture decisions in `docs/`

## Development

### Contributing

1. **Read first**: See `docs/` for architecture and design principles, `AGENTS.md` for code conventions
2. **Keep focused**: One feature or bugfix per PR
3. **Test**: Add unit or integration tests; run `pytest` before submitting
4. **Document**: Update docstrings and relevant docs
5. **Describe**: Write a clear PR title and description with rationale

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/lapis

# Run specific test
pytest tests/test_tokenizer.py -v
```

### Extending

Common extension points:
- **New datasets**: Add loader to `src/lapis/data/loaders/`
- **Model variants**: Create new config in `configs/` or modify architecture in `src/lapis/model/`
- **Export formats**: Add exporter to `src/lapis/export/`
- **Training techniques**: Extend `src/lapis/training/trainer.py`

## Licensing & Attribution

- **License**: MIT (see [LICENSE](./LICENSE))
- **Datasets**: Ensure datasets have clear, permissive licenses before adding to the repo
- **External code**: All significant external dependencies are listed in `pyproject.toml`

## Resources

- 📖 **Documentation**: [https://sudomarc.github.io/LapisLLM/](https://sudomarc.github.io/LapisLLM/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/sudomarc/LapisLLM/issues)
- 💬 **Discussions**: Start a [Discussion](https://github.com/sudomarc/LapisLLM/discussions) for questions

## Citation

If you use LapisLLM in research or production, please cite:

```bibtex
@software{lapislLM2026,
  author = {sudomarc},
  title = {LapisLLM: From-Scratch Decoder-Only Transformer Language Model},
  url = {https://github.com/sudomarc/LapisLLM},
  year = {2026}
}
```

## Acknowledgments

LapisLLM draws inspiration from:
- [Karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Minimal GPT implementation
- [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) — Large-scale training
- [Hugging Face Transformers](https://huggingface.co/transformers/) — Architecture reference
- Open-source LLM community best practices

## Contact & Support

- **Maintainer**: [@sudomarc](https://github.com/sudomarc)
- **Repository**: [https://github.com/sudomarc/LapisLLM](https://github.com/sudomarc/LapisLLM)

---

**Status**: Correctness milestone (0.1.1). Architecture is stable; feedback welcome.
