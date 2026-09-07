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

## Lapis Console

The recommended entrypoint for Colab and local experimentation is the unified Lapis Console:

```bash
lapis
```

or directly from a checkout:

```bash
python scripts/lapis.py
```

The console starts by synchronizing with `origin/main`. It refuses to overwrite a dirty working tree. The training command then orchestrates data preparation, GPU training, the Learning Monitor, strict checkpoint validation, experiment history, Git commit/push, and finally the trained chatbot.

### One-command training

```bash
lapis train
```

The console asks:

```text
How many training runs? ›
Learning monitor interval [500]? ›
```

For each run it performs:

```text
GIT PULL
   ↓
VERIFY WORKTREE
   ↓
BUILD TRAINING CORPUS
   ↓
TRAIN ON CUDA
   ↓
LIVE PROGRESS + LEARNING MONITOR
   ↓
VERIFY CHECKPOINT / STEPS / MONITOR
   ↓
WRITE TRAINING HISTORY
   ↓
GIT COMMIT + PUSH
   ↓
LAUNCH CHAT
```

A run is never reported as successful without a checkpoint, a matching recorded optimizer step count, and monitor records. Incomplete runs do not trigger the final history push.

Large model checkpoints remain local by default. Only lightweight experiment history is written to `training_history/` and pushed: metadata, final metrics, and final learning samples.

## Learning Monitor

The trainer observes the model while it is learning. At each configured interval it reports:

- loss and perplexity
- learning rate
- tokens seen
- generated samples from fixed prompts

Example:

```text
──────────────────────────────────────────────────────────────────────
LEARNING MONITOR
step=05000  loss=...  ppl=...  lr=...  tokens=...

WHAT LAPIS IS LEARNING

Prompt : Machine learning is
Lapis  : ...

Prompt : The transformer architecture
Lapis  : ...
```

Monitor records are stored as JSONL next to the local checkpoint. The trainer temporarily switches the model to evaluation mode for sampling and restores training mode afterward.

Low-level customization:

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --device cuda \
  --data training_data/combined.txt \
  --monitor-interval 500 \
  --monitor-sample-tokens 64
```

Custom prompts use `||` as separators:

```bash
python scripts/train.py \
  --monitor-prompts "Machine learning is||The transformer architecture||Language models learn"
```

Use `--monitor-interval 0` to disable the monitor in the low-level trainer.

## Training progress

The unified console converts training steps into a live terminal progress display:

```text
[████████████████░░░░░░░░░░░░░░░░]  5,000/10,000  50.00% loss=1.4821 18.7 step/s ETA 4m 27s
```

The progress display includes step count, percentage, loss, steps/second, and estimated time remaining.

## Training history

Every verified run creates:

```text
training_history/
└── run-YYYYMMDD-HHMMSS-xxxxxx/
    ├── summary.json
    └── samples.md
```

Inspect previous runs:

```bash
lapis history
```

Compare two runs:

```bash
lapis compare
```

The comparison includes final loss, perplexity, tokens seen, duration, and qualitative learning samples.

## Chat

Launch the latest local trained checkpoint with:

```bash
lapis chat
```

The direct entrypoints remain available:

```bash
chat
lapis-chat
python -m scripts.chat
```

The terminal interface is local-first and uses a compact developer-console design inspired by modern coding assistants. It maintains conversation context within the model's context window and reports generation throughput.

### Chat commands

```text
/help
/clear
/reset
/stats
/context
/model
/temperature 0.7
/tokens 128
/save [file]
/exit
```

The session exposes model/device information, context usage, generated-token statistics, runtime temperature, and maximum output length. Conversations can be saved to Markdown.

## System information

```bash
lapis system
```

This reports the Git branch and CUDA/GPU information available to the current environment.

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
├── scripts/               # train, evaluate, generate, chat, serve, lapis console
├── configs/               # reproducible YAML experiments
├── checkpoints/           # local model checkpoints + tokenizer
├── training_history/      # lightweight verified experiment records
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

The Lapis Console adds per-run metadata and qualitative samples under `training_history/` and validates the recorded optimizer step count before marking a run complete.

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
