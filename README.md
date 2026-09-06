# LapisLLM

**Lapis is a from-scratch decoder-only Transformer language model and training stack written in Python/PyTorch.**

The project makes the tokenizer, data pipeline, Transformer architecture, optimization, checkpointing, evaluation, inference, and serving explicit and inspectable.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Pages](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Current status — Lapis 0.1.1: correctness hardening.** The project is experimental and is not a competitive pretrained foundation model. Benchmark and capability claims are withheld until reproducible validation exists.

## Why Lapis

Lapis takes a build-first approach to language models. The goal is to keep the mechanics understandable enough to debug, reproduce, and extend rather than hiding them behind a hosted endpoint or large training framework.

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

The repository's default correctness profile is **Lapis Tiny**. `configs/tiny.yaml` currently uses a 512-token vocabulary, hidden size 96, two Transformer layers, four query heads, two key/value heads, and a 64-token model context. The instantiated parameter count is reported by the training code.

Larger configurations are development/roadmap targets and are not released or benchmarked models.

## Quickstart

Requirements: Python 3.11+ and a compatible PyTorch installation.

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
python -m pytest
python scripts/train.py --config configs/tiny.yaml --device cpu --checkpoint outputs/tiny.pt
python scripts/generate.py --checkpoint outputs/tiny.pt --prompt "LAPIS" --temperature 0 --top-k 1 --top-p 1
```

The tiny profile is designed for local development and CI. GPU profiles are available separately through configuration.

## Training and resume

Training uses a deterministic validation split of the supplied raw text. The tokenizer is trained from the training partition only, so the held-out partition is not used to construct the tokenizer vocabulary.

Checkpoints are versioned and self-contained. They include model, optimizer, scheduler, training state, tokenizer serialization, data fingerprint, and RNG state. Checkpoints are written atomically to avoid replacing a good artifact with a partial file.

Resume is strict: model architecture, training hyperparameters that affect continuation, tokenizer metadata, and the dataset fingerprint must match the checkpoint. Incompatible state fails early instead of silently restarting or mixing incompatible artifacts.

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --device cpu \
  --checkpoint outputs/tiny.pt \
  --resume outputs/tiny.pt
```

The current release does not promise exact mid-epoch replay because DataLoader position is not serialized. Resume is therefore validated only at checkpoint boundaries.

## Evaluation

Evaluation uses the held-out portion of the same supplied corpus and reports token-weighted cross-entropy and perplexity. For a meaningful experiment, provide the same dataset that was used to create the checkpoint.

```bash
python scripts/evaluate.py --checkpoint outputs/tiny.pt --data path/to/corpus.txt --device cpu
```

## Generation

`temperature=0` is explicit greedy decoding. Positive temperatures enable sampling. `top-k` may be `0` to disable filtering; `top-p` must be in `(0, 1]`. Prompts longer than the model context are intentionally left-truncated during generation.

```bash
python scripts/generate.py \
  --checkpoint outputs/tiny.pt \
  --prompt "LAPIS" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

## Evaluation philosophy

Lapis uses evidence gates:

1. **Correctness** — shapes, causality, masking, tokenizer behavior, gradients, and checkpoint state.
2. **Learning** — tiny-dataset overfitting.
3. **Generalization** — held-out loss and perplexity.
4. **Generation** — deterministic and sampling regression tests.
5. **Scaling** — larger configurations only after the earlier stages are reproducible.

No benchmark result is published until the underlying experiment is reproducible.

## Repository layout

```text
LapisLLM/
├── lapis/                 # model, tokenizer, data, config, checkpoint helpers
├── scripts/               # train, evaluate, generate, chat, serve, data utilities
├── configs/               # reproducible YAML configurations
├── tests/                 # correctness and integration tests
├── docs/                  # technical documentation
├── website/               # static GitHub Pages site
├── CHANGELOG.md
├── AGENTS.md
├── LICENSE
└── README.md
```

The repository does not require a committed model checkpoint for installation or CI; release validation creates temporary checkpoints as part of the test workflow.

## Reproducibility and limitations

Training accepts an explicit seed and records RNG state in checkpoints. Exact mid-epoch DataLoader replay is not yet implemented. Distributed training, efficient KV-cache generation, production-scale data processing, broad benchmark evaluation, and large-model scaling remain future work.

## Development

```bash
pip install -e ".[test]"
python -m pytest
ruff check .
python -m build
```

See [`AGENTS.md`](AGENTS.md) for repository engineering rules.

## Website

The `website/` directory contains the static developer platform for Models, Skills, Docs, Research, Roadmap, Changelog, and About. It is deployed to GitHub Pages.

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
