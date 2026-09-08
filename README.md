<div align="center">

# L A P I S L L M

**An inspectable, from-scratch language model and training stack in Python/PyTorch.**

Build the tokenizer. Build the Transformer. Train it. Inspect what it learns.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Website](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-000000)](LICENSE)

[**Website**](https://sudomarc.github.io/LapisLLM/) · [**Documentation**](docs/) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](https://sudomarc.github.io/LapisLLM/roadmap/)

</div>

> [!IMPORTANT]
> **Project status — Lapis 0.2.0.** Lapis is an experimental research and learning project. It is designed for transparency, reproducibility, and engineering practice; it is **not** presented as a competitive pretrained foundation model. Capability and benchmark claims are intentionally withheld until reproducible evaluation exists.

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
python -m pytest
```

Check static analysis:

```bash
ruff check .
```

## LapisLLM modes

LapisLLM is one project with one shared model/tokenizer/checkpoint core and two explicit operating modes.

### USER mode — inference by default

For an end user, the default entrypoint is deliberately simple:

```bash
lapis
```

Equivalent explicit command:

```bash
lapis chat
```

USER mode loads the configured/default checkpoint and exposes inference controls only. User-facing settings are kept in `configs/user/default.yaml`:

```yaml
model: latest
device: auto
temperature: 0.8
top_k: 40
top_p: 0.95
max_new_tokens: 128
```

Training parameters such as learning rate, optimizer state, epochs, batch size and dataset paths are not part of the USER configuration.

The stable Python inference API is:

```python
from lapis.inference import LapisRuntime

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")
response = runtime.generate("Explain DNS.")
```

The USER runtime loads checkpoints read-only and does not expose training or checkpoint-writing operations.

### DEV mode — explicit development tooling

Developer operations live behind an explicit namespace:

```bash
lapis dev --help
lapis dev train --config configs/tiny.yaml
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "A language model learns by"
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
lapis dev checkpoint inspect --checkpoint checkpoints/latest.pt
```

DEV owns training, evaluation, benchmarking, checkpoint inspection and the detailed training console. Existing lower-level scripts and legacy top-level commands remain available for compatibility.

The design rule is:

```text
lapis       → USER → inference/chat
lapis dev   → DEV  → train/evaluate/inspect/benchmark
```

See [`docs/modes.md`](docs/modes.md) for the full separation contract.

## Lapis Console

The recommended user entrypoint is now `lapis`, which launches USER chat. Developer experiments should use the explicit `lapis dev` namespace.

The original experiment console remains available from a checkout:

```bash
python scripts/lapis.py
```

### One-command experiment

```bash
lapis dev train
```

The console:

```text
PULL
  ↓
VERIFY WORKTREE
  ↓
BUILD / REUSE CORPUS
  ↓
TRAIN
  ↓
LIVE PROGRESS + LEARNING MONITOR
  ↓
VERIFY CHECKPOINT
  ↓
WRITE HISTORY
  ↓
COMMIT + PUSH HISTORY
  ↓
LAUNCH CHAT
```

Before synchronization, generated artifacts are separated from user-authored source changes. The console refuses to overwrite unrelated local modifications.

For every training run, the pipeline verifies that:

- the process exited successfully
- the checkpoint exists and is non-trivial in size
- the recorded optimizer step matches the configured target
- the Learning Monitor produced records
- the final metrics are available
- the checkpoint can be reloaded by the inference stack

Large checkpoints and generated corpora remain local by default; lightweight experiment history is tracked under `training_history/`.

## Training

The lower-level trainer remains available when you need direct control:

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --device cuda \
  --data training_data/combined.txt \
  --monitor-interval 500 \
  --monitor-sample-tokens 64
```

### CPU-fast profile

For a quick CPU iteration:

```bash
python scripts/train.py \
  --config configs/cpu-fast.yaml \
  --device cpu \
  --epochs 1
```

Or apply the CPU profile to another configuration:

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --cpu-fast \
  --device cpu \
  --epochs 1
```

The CPU-fast profile intentionally changes the model architecture and therefore cannot be resumed into a normal Tiny/Small checkpoint.

## Learning Monitor

The Learning Monitor is designed to answer a practical question:

> **What is the model learning while the optimizer is changing the weights?**

At configured intervals it records:

- training loss
- perplexity
- learning rate
- tokens seen
- generated samples from fixed prompts

Example:

```text
──────────────────────────────────────────────────────────────────────
LEARNING MONITOR
step=05000  loss=1.8421  ppl=6.31  lr=1.5e-04  tokens=10,220,000

WHAT LAPIS IS LEARNING

Prompt : Machine learning is
Lapis  : ...

Prompt : The transformer architecture
Lapis  : ...
```

The final training step can be forced into the monitor so experiment history does not accidentally describe an earlier checkpoint state.

Custom prompts use `||` as separators:

```bash
python scripts/train.py \
  --monitor-prompts "Machine learning is||The transformer architecture||Language models learn"
```

Disable the monitor explicitly:

```bash
python scripts/train.py --monitor-interval 0
```

## Training progress

The unified console renders the trainer as a live progress stream:

```text
[████████████████░░░░░░░░░░░░░░░░]  5,000/10,000  50.00% loss=1.4821 18.7 step/s ETA 4m 27s
```

The display is intentionally dependency-free and reports optimizer step progress, loss, throughput, and ETA.

## Open training corpus

Lapis can build a bounded mixed-domain corpus from public/open datasets with streaming ingestion:

```bash
python scripts/fetch_open_corpus.py \
  --max-chars 50000000
```

The collector records provenance and source-level statistics in `training_data/open/manifest.json`.

The repository does **not** keep generated multi-megabyte training corpora in Git by default. Training data is an experiment input, not source code.

## Evaluation

Lapis follows an evidence-gated development sequence:

```text
Correctness
    ↓
Tiny-dataset learning
    ↓
Held-out loss / perplexity
    ↓
Generation regression
    ↓
Scaling
```

Run validation on a checkpoint:

```bash
evaluate --checkpoint checkpoints/latest.pt --device cuda
```

The evaluation stack reuses the checkpoint's tokenizer and verifies tokenizer-version and vocabulary compatibility before loading the model.

## Generation

Generate text directly from a checkpoint:

```bash
generate \
  --checkpoint checkpoints/latest.pt \
  --prompt "The transformer architecture" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

Sampling validates temperature, top-k, top-p, and resulting probability tensors before drawing the next token.

## Terminal chat

Launch the latest locally trained checkpoint:

```bash
lapis chat
```

Direct launchers are also available:

```bash
chat
lapis-chat
python -m scripts.chat
```

The terminal UI keeps conversation history during the session, trims old turns to the model context window, reports generation throughput, and exposes model/runtime controls.

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

## Serving

Lapis provides a lightweight OpenAI-style HTTP API:

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

The server is intended for local experimentation and integration testing rather than production inference infrastructure.

## Colab

Google Colab is a first-class development environment for Lapis.

From a fresh checkout:

```bash
%cd /content/LapisLLM
!git pull --ff-only origin main
!pip install -e .
```

Then use the developer namespace:

```bash
!lapis dev train
```

The console performs the Git synchronization at startup, asks how many complete training runs to execute, validates each run, records its history, and attempts to push only the lightweight history back to GitHub.

> [!NOTE]
> GitHub push authentication must be configured in the Colab runtime. A public repository can still be cloned and pulled without write credentials, but pushing to `main` requires authenticated Git access.

## Repository layout

```text
LapisLLM/
├── lapis/
│   ├── config/            # configuration models and runtime validation
│   ├── inference/         # shared inference-only runtime
│   ├── model/             # Transformer, attention, RoPE, MLP, normalization
│   ├── tokenizer/         # BPE tokenizer
│   ├── user/              # end-user inference facade
│   ├── dev/               # explicit developer CLI namespace
│   └── training/          # Learning Monitor and training helpers
├── scripts/
│   ├── lapis.py           # unified experiment console
│   ├── train.py           # low-level trainer
│   ├── evaluate.py        # validation loss / perplexity
│   ├── generate.py        # checkpoint generation
│   ├── chat.py            # terminal chat
│   ├── serve.py           # OpenAI-style HTTP server
│   └── fetch_open_corpus.py
├── configs/               # YAML experiment configurations
│   └── user/              # user-only inference configuration
├── tests/                 # correctness and integration tests
├── training_history/      # lightweight verified experiment records
├── docs/                  # technical documentation
├── website/               # GitHub Pages developer site
├── CHANGELOG.md
├── AGENTS.md
├── LICENSE
└── README.md
```

## Engineering and reliability

The project treats checkpoint and experiment integrity as part of model correctness.

### Checkpoint integrity

A saved checkpoint contains model weights plus the optimizer, scheduler, configuration, and RNG state required for reproducible continuation.

Checkpoint publication is staged so model and tokenizer updates are committed together; a failed publication attempts to restore the previous pair.

### Git safety

The experiment console:

- pulls with `--ff-only`
- refuses to overwrite unrelated local changes
- keeps generated checkpoints/corpora out of normal source commits
- pushes lightweight history instead of binary training artifacts

No destructive `git reset --hard` flow is used by the training console.

## Current limitations

Lapis is intentionally incomplete. Known engineering gaps include:

- distributed training
- production-scale data sharding and deduplication
- high-performance KV-cache inference
- broad benchmark suites
- full mixed-precision optimization strategy
- exact mid-epoch dataloader replay
- production serving hardening
- large-scale model releases

These are roadmap items, not hidden assumptions.

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

## Skills

The Lapis website also contains a modular skills layer for developer workflows, including security audit, PR engineering, documentation, benchmarking, dataset cleaning, web testing, and release management.

Skills are treated as versioned specifications with explicit inputs, outputs, limitations, and safety boundaries. They do not imply unrestricted autonomous access to external systems.

## Contributing

Contributions should preserve the project's emphasis on correctness and reproducibility.

Before opening a pull request:

```bash
python -m pytest
ruff check .
```

For behavior changes, add or update tests. For architecture changes, update the relevant documentation and changelog.

See [`AGENTS.md`](AGENTS.md) for repository-specific engineering rules.

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

## Contact

Issues and discussions are the preferred way to report reproducibility problems, bugs, and proposed architectural changes.
