# LapisLLM

**Lapis is a from-scratch decoder-only Transformer language model and training stack written in Python/PyTorch.**

The project exists to make LLM engineering inspectable: tokenizer, data pipeline, Transformer architecture, optimization, checkpointing, evaluation, inference, and serving are explicit parts of the repository.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Pages](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Current status — Lapis 0.1.1: Correctness.** The project is experimental and is not yet a competitive pretrained foundation model. Benchmark and capability claims are intentionally withheld until reproducible validation exists.

## Why Lapis

Lapis takes a build-first approach to language models. The goal is not to hide the implementation behind a hosted endpoint or a large training framework; it is to make the mechanics understandable enough to debug, reproduce, and extend.

Current model components include:

- decoder-only causal language modeling
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward blocks
- configurable model dimensions
- ByteLevel BPE tokenization
- PyTorch training and checkpointing

## Models

The development model is **Lapis Tiny**. Its current development configuration is intentionally small for local iteration. The exact instantiated parameter count is reported by the training code rather than hard-coded here.

Larger configurations such as Lapis Small, Lapis 1B, Lapis 3B, and Lapis 7B are roadmap targets, not released or benchmarked models.

## Quickstart

Requirements: Python 3.11+ and a development installation of PyTorch.

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest
python scripts/train.py --config configs/tiny.yaml
```

### CPU-only fast training

When a Colab account or local machine has no usable accelerator, use the dedicated CPU profile instead of trying to train the larger configuration unchanged:

```bash
python scripts/train.py --config configs/cpu-fast.yaml --device cpu --epochs 1
```

For a one-off fast run from another configuration, the training entrypoint also supports:

```bash
python scripts/train.py --config configs/tiny.yaml --cpu-fast --epochs 1
```

`--cpu-fast` reduces model width, layer count, and context length while keeping the tokenizer vocabulary compatible with the Tiny setup. It is intended for fast iteration and smoke experiments, not final model training. It cannot be combined with `--resume` because the model architecture changes.

CPU runtime settings are configurable under `runtime.cpu` (`threads`, `interop_threads`, `dataloader_workers`, and `pin_memory`). The trainer also reports the effective CPU thread configuration at startup.

The normal `tiny.yaml` path remains available for the full configured experiment. The CPU-fast path is deliberately separate so that CPU development does not silently change the architecture of an existing training run.

## Evaluation philosophy

Lapis uses evidence gates:

1. **Correctness** — shapes, causality, masking, tokenizer behavior, gradients, and checkpoint state.
2. **Learning** — tiny-dataset overfitting.
3. **Generalization** — held-out loss and perplexity.
4. **Generation** — deterministic and sampling regression tests.
5. **Scaling** — larger configurations only after the earlier stages are reproducible.

No benchmark result is published in this repository until the underlying experiment is reproducible.

## Skills

The website includes a Lapis-native skills ecosystem for modular developer workflows. Skills are versioned packages with a normative `SKILL.md`, explicit inputs/outputs, workflow stages, limitations, and safety boundaries.

Current skill catalog:

- Security Audit
- Pentest Planner
- Reverse Engineer
- PR Engineer
- Docs Writer
- Benchmark
- Dataset Cleaner
- Prompt Optimizer
- Web Tester
- Release Manager

These packages are an experimental specification layer; they do not imply autonomous access to external systems.

## Repository layout

```text
LapisLLM/
├── lapis/                 # model, tokenizer, data, config, training helpers
├── scripts/               # train, evaluate, generate, chat, serve
├── configs/               # reproducible YAML experiments
├── tests/                 # correctness and integration tests
├── docs/                  # technical project documentation
├── website/               # static developer platform / GitHub Pages site
├── CHANGELOG.md
├── AGENTS.md
├── LICENSE
└── README.md
```

## Reproducibility

Checkpoints preserve model, optimizer, scheduler, training/configuration state, tokenizer references, and RNG state. Training accepts an explicit seed. Exact mid-epoch data-loader replay is still being hardened.

## Known limitations

Lapis is an experimental research/learning project. Large-scale distributed training, mixed-precision training, efficient KV-cache generation, comprehensive benchmark evaluation, production-grade web-scale data processing, and exact mid-epoch replay are not yet complete.

## Contributing

Keep changes focused, add tests for behavioral changes, run the test suite and Ruff, and document architecture or behavior changes.

```bash
python -m pytest
ruff check .
```

See [`AGENTS.md`](AGENTS.md) for repository engineering rules.

## Website

The `website/` directory is a static developer platform for Models, Skills, Docs, Research, Roadmap, Changelog, and About. It is deployed to GitHub Pages and does not require a permanent server.

Local website checks:

```bash
cd website
npm install
npm run lint
npm run build
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
