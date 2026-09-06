# LapisLLM

**Lapis is a from-scratch decoder-only Transformer language model and training stack written in Python/PyTorch.**

The project makes the tokenizer, data pipeline, Transformer architecture, optimization, checkpointing, evaluation, inference, and serving explicit and inspectable.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Pages](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Current status — Lapis 0.1.2: correctness and reproducibility hardening.** The project is experimental. No benchmark or foundation-model capability claims are made here.

## Architecture

Lapis uses a decoder-only causal Transformer with:

- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward blocks
- configurable model dimensions
- ByteLevel BPE tokenization
- PyTorch training, checkpointing, evaluation, generation, chat, and serving entry points

## Development model

`configs/tiny.yaml` is the canonical development configuration. It currently defines a 4096-token vocabulary, 256 hidden units, 1024 intermediate units, 6 Transformer layers, 8 query heads, 4 key/value heads, and a 512-token model context. Training uses a 511-token data sequence length so one additional token can be retained for next-token targets.

The repository also includes `configs/local-dev.yaml` for a smaller CPU-oriented development path. Larger configurations such as Lapis 1B, 3B, and 7B are roadmap targets, not released or benchmarked models.

## Quickstart

Requirements: Python 3.11+.

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
python scripts/train.py --config configs/local-dev.yaml --device cpu --epochs 1
```

Training creates checkpoints and tokenizer artifacts locally under the requested checkpoint directory. Generated artifacts are intentionally ignored by Git.

## Training and resume

A checkpoint stores model, optimizer, scheduler, configuration, tokenizer metadata, training progress, and RNG state. Checkpoint writes are atomic so an interrupted write does not replace the previous complete `.pt` file.

Resume is strict: the checkpoint format, model configuration, training schedule, data sequence configuration, tokenizer version, and tokenizer vocabulary must match the current run. The checkpoint tokenizer is used automatically; `--resume` cannot be combined with `--tokenizer`.

The current data loader uses a deterministic per-epoch generator and stores the next batch position, allowing single-process mid-epoch continuation. Multi-worker and distributed replay are not promised.

Example:

```bash
python scripts/train.py --config configs/local-dev.yaml --device cpu --epochs 1 --checkpoint /tmp/lapis/checkpoint.pt
python scripts/train.py --config configs/local-dev.yaml --device cpu --epochs 1 \
  --resume /tmp/lapis/checkpoint.pt --checkpoint /tmp/lapis/resumed.pt
```

## Evaluation and generation

Evaluation is an explicit corpus-level loss/perplexity smoke test unless `--data` is supplied:

```bash
python scripts/evaluate.py --checkpoint /tmp/lapis/resumed.pt --device cpu
```

Generation validates its inputs before execution. `temperature=0` selects greedy decoding; invalid `top_k`, `top_p`, or token counts are rejected; prompts beyond the model context are rejected rather than silently truncated.

```bash
python scripts/generate.py --checkpoint /tmp/lapis/resumed.pt --device cpu \
  --prompt "Lapis" --max-new-tokens 32 --temperature 0
```

## Data preparation

The packaged `prepare_data` command concatenates sorted UTF-8 `.txt` files deterministically:

```bash
prepare_data --input-dir data/raw --output data/combined.txt
```

Empty input directories and corpora containing only blank files fail explicitly.

## Reproducibility

Use `--seed` for deterministic single-process experiments. Python and PyTorch RNG state is stored in checkpoints. The deterministic data-loader generator is seeded independently per epoch. Exact bitwise determinism across different hardware, CUDA versions, distributed workers, or kernels is not guaranteed.

## Validation philosophy

Lapis uses evidence gates:

1. correctness — tensor shapes, causality, masking, tokenizer behavior, gradients, checkpoint compatibility;
2. learning — tiny-dataset overfitting;
3. generalization — held-out loss and perplexity;
4. generation — deterministic and sampling edge cases;
5. scaling — larger configurations only after earlier stages are reproducible.

No benchmark result is published until the underlying experiment is reproducible.

## Repository layout

```text
LapisLLM/
├── lapis/                 # model, tokenizer, data, config, helpers
├── scripts/               # installed train/evaluate/generate/chat/serve CLIs
├── configs/               # reproducible YAML configurations
├── tests/                 # unit, integration, and release-hardening tests
├── docs/                  # technical project documentation
├── website/               # static GitHub Pages site
├── training_data/         # repository-managed training corpus samples
├── CHANGELOG.md
├── AGENTS.md
├── LICENSE
└── README.md
```

## Known limitations

Large-scale distributed training, mixed-precision training, efficient KV-cache generation, comprehensive benchmark evaluation, and web-scale data processing are not implemented. CUDA execution is supported by the model/training paths when a compatible CUDA environment is available, but this repository does not provide a GPU runner for guaranteed CI coverage.

Checkpoint loading accepts native PyTorch `.pt` files and should therefore be treated as loading trusted checkpoint artifacts. Do not load untrusted checkpoints.

## Development checks

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q lapis scripts tests
python -m pip wheel . --no-deps --wheel-dir dist
```

The CI workflow additionally runs CPU training, resume, evaluation, generation, and installed CLI smoke tests.

## Skills

The website includes an experimental Lapis-native skills specification layer. Skills are versioned packages with explicit inputs, outputs, workflow stages, limitations, and safety boundaries; they do not imply autonomous access to external systems.

## Website

The `website/` directory is a static developer platform for Models, Skills, Docs, Research, Roadmap, Changelog, and About. It is deployed to GitHub Pages.

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
