# LapisLLM

**Lapis is a from-scratch decoder-only Transformer language model and training stack written in Python/PyTorch.**

The project is built to keep the important parts of an LLM inspectable: tokenizer, data pipeline, Transformer architecture, optimization, checkpointing, evaluation, inference, and serving.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Pages](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Current status — Lapis 0.1.1: Correctness.** Lapis is an experimental research/learning project, not a competitive pretrained foundation model. Benchmark and capability claims are intentionally withheld until reproducible validation exists.

## Quickstart

Requirements: Python 3.11+ and PyTorch 2.0+.

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e .
```

Run the test suite:

```bash
python -m pytest
```

### Launch the chatbot directly

A trained checkpoint and its tokenizer are already exposed through the repository's chat entrypoint. After the editable install, launch Lapis with:

```bash
chat
```

or, with the explicit launcher name:

```bash
lapis-chat
```

You can also run it directly from the repository without reinstalling the package:

```bash
python -m scripts.chat
```

The default runtime looks for:

```text
checkpoints/latest.pt
checkpoints/tokenizer/
```

Those paths can be overridden when testing another model:

```bash
chat --checkpoint checkpoints/my-model.pt --device cpu
```

### Terminal interface

The CLI is intentionally local-first and terminal-native. Its interface uses a compact, Claude Code-inspired layout with a model header, device/checkpoint status, streaming responses, and slash commands.

Available commands inside the chat:

```text
/help   show commands
/clear  clear the terminal and redraw the session header
/exit   leave the chat
```

Useful generation options:

```bash
chat --max-new-tokens 128
chat --temperature 0.7 --top-k 40 --top-p 0.95
chat --no-color
```

The chatbot runs entirely against the selected local checkpoint. It does not require an external API key or hosted inference service.

## CPU-only fast training

When no accelerator is available, use the dedicated CPU profile:

```bash
python scripts/train.py --config configs/cpu-fast.yaml --device cpu --epochs 1
```

For a one-off fast run from another configuration:

```bash
python scripts/train.py --config configs/tiny.yaml --cpu-fast --epochs 1
```

`--cpu-fast` reduces model width, layer count, and context length while keeping the tokenizer vocabulary compatible with the Tiny setup. It is intended for fast iteration and smoke experiments, not final model training. It cannot be combined with `--resume` because the model architecture changes.

CPU runtime settings are configurable under `runtime.cpu` (`threads`, `interop_threads`, `dataloader_workers`, and `pin_memory`). The trainer reports the effective CPU thread configuration at startup.

## Model

The development model is **Lapis Tiny**. Its exact instantiated parameter count is reported by the training code rather than hard-coded here.

Larger configurations such as Lapis Small, Lapis 1B, Lapis 3B, and Lapis 7B are roadmap targets, not released or benchmarked models.

Current model components include:

- decoder-only causal language modeling
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward blocks
- configurable model dimensions
- ByteLevel BPE tokenization
- PyTorch training and checkpointing

## Evaluation philosophy

Lapis uses evidence gates:

1. **Correctness** — shapes, causality, masking, tokenizer behavior, gradients, and checkpoint state.
2. **Learning** — tiny-dataset overfitting.
3. **Generalization** — held-out loss and perplexity.
4. **Generation** — deterministic and sampling regression tests.
5. **Scaling** — larger configurations only after the earlier stages are reproducible.

No benchmark result is published until the underlying experiment is reproducible.

## Serving

For application integrations, Lapis also provides an OpenAI-style HTTP API:

```bash
serve --checkpoint checkpoints/latest.pt --host 127.0.0.1 --port 8000
```

The API exposes `/v1/models` and `/v1/chat/completions`.

## Skills

The website includes a Lapis-native skills ecosystem for modular developer workflows. Skills are versioned packages with a normative `SKILL.md`, explicit inputs/outputs, workflow stages, limitations, and safety boundaries.

Current catalog:

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
├── checkpoints/           # local model checkpoints + tokenizer
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
