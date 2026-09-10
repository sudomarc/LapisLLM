<div align="center">

# L A P I S L L M

**An inspectable, from-scratch decoder-only language model and ML/runtime engineering platform built with Python and PyTorch.**

Build the tokenizer. Build the Transformer. Train it. Evaluate it. Inspect every layer of the stack.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Website](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/static.yml?branch=main&label=website)](https://github.com/sudomarc/LapisLLM/actions/workflows/static.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-000000)](LICENSE)

[**Website**](https://sudomarc.github.io/LapisLLM/) · [**Documentation**](docs/) · [**Testing**](docs/testing.md) · [**Contributing**](CONTRIBUTING.md) · [**Security**](SECURITY.md) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](https://sudomarc.github.io/LapisLLM/roadmap/)

</div>

> [!IMPORTANT]
> **Current release: Lapis 0.2.0.** Lapis is an experimental research and learning project. It is designed to make model engineering, training behavior, runtime boundaries, and validation inspectable and reproducible. It is **not** currently presented as a competitive pretrained foundation model, and this repository does not claim benchmark performance or general-purpose capability without reproducible evidence.

## What Lapis is

LapisLLM is the **engine** behind a language-model system, not the consumer application around it.

The repository owns the model and ML/runtime engineering layers:

- decoder-only Transformer architecture;
- ByteLevel BPE tokenization;
- causal attention with grouped-query attention (GQA);
- Rotary Position Embeddings (RoPE);
- RMSNorm and SwiGLU blocks;
- training, optimization, gradient accumulation, clipping, warmup, and cosine decay;
- checkpointing and reproducibility state;
- evaluation and autoregressive generation;
- a stable, product-agnostic inference runtime;
- developer/research CLI and experiment tooling;
- testing, packaging, security, and CI infrastructure.

The user-facing conversational product is **CHAD**, a separate repository. CHAD owns chat UX, conversations, history, accounts, user settings, and other consumer workflows. Lapis provides the model/runtime layer that an external application can consume.

```text
                         LapisLLM
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     Training            Runtime          Developer CLI
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    public runtime API
                            │
                            ▼
                           CHAD
                            │
                          users
```

### What Lapis is not

LapisLLM deliberately does **not** provide a first-class consumer/user mode. It must not grow accounts, consumer onboarding, conversation persistence, profile/settings systems, or a top-level consumer chatbot. The interactive terminal console exists for developer and research validation only.

## Why this project exists

Lapis is built around a simple engineering principle: **understand the system before scaling the system**.

A useful LLM project needs more than a Transformer implementation. It needs reliable configuration, tokenizer/model compatibility, deterministic experiments, checkpoint validation, generation safeguards, tests that catch regressions, reproducible data provenance, and a clean interface between the model engine and applications.

The repository therefore treats correctness and observability as first-class engineering features rather than polish added after training.

## Architecture

```text
Text
  │
  ▼
ByteLevel BPE tokenizer
  │
  ▼
Token IDs
  │
  ▼
Token embeddings
  │
  ▼
┌─────────────────────────────────────────────┐
│ Transformer block × N                       │
│                                             │
│  RMSNorm → GQA causal self-attention        │
│           + RoPE                            │
│             ↓                               │
│        residual connection                  │
│             ↓                               │
│  RMSNorm → SwiGLU feed-forward              │
│             ↓                               │
│        residual connection                  │
└─────────────────────────────────────────────┘
  │
  ▼
Final RMSNorm
  │
  ▼
Language-model head
  │
  ▼
Next-token logits / probabilities
```

### Tokenization

The tokenizer uses the Hugging Face `tokenizers` library with a ByteLevel BPE backend and NFKC normalization. Tokenizer compatibility is checked at inference time against checkpoint metadata and vocabulary size.

### Transformer core

The current Tiny development configuration uses 6 layers, hidden size 256, 8 query heads, 4 key/value heads, and a 512-token context window. The model uses causal attention, RoPE, RMSNorm, and SwiGLU.

### Training objective

Training uses next-token prediction with cross-entropy loss. The training stack includes gradient accumulation, gradient clipping, linear warmup, and cosine learning-rate decay.

## Current model

The repository currently has one named development configuration. Larger sizes are roadmap targets, not released model claims.

| Configuration | Status | Hidden | Layers | Q heads | KV heads | Context |
|---|---|---:|---:|---:|---:|---:|
| **Lapis Tiny** | Experimental | 256 | 6 | 8 | 4 | 512 |
| Lapis 1B | Roadmap | — | — | — | — | — |
| Lapis 3B | Roadmap | — | — | — | — | — |
| Lapis 7B | Roadmap | — | — | — | — | — |

Canonical configuration: [`configs/tiny.yaml`](configs/tiny.yaml).

The current Tiny configuration defines a 4,096-token vocabulary and is intended for engineering validation, small experiments, and development—not as evidence of production-scale model capability.

## Quickstart

### 1. Install Lapis

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

For corpus generation, install the optional data dependencies too:

```bash
pip install -e ".[data,dev]"
```

### 2. Inspect the developer CLI

```bash
lapis --help
lapis dev --help
```

The explicit `lapis dev ...` namespace is the preferred interface for development and research workflows.

### 3. Train

For the normal Tiny development path, first build a bounded open corpus and then train against the generated combined file:

```bash
python scripts/fetch_open_corpus.py --max-chars 50000000
python scripts/train.py \
  --config configs/tiny.yaml \
  --device cuda \
  --data training_data/open/combined.txt
```

The collector is intentionally bounded. It streams public/open datasets, cleans text, records per-source provenance, and writes `training_data/open/manifest.json` alongside source files. Generated corpora are experiment inputs and are not committed to Git by default.

### 4. Fast CPU iteration

For a small local or CI-style iteration:

```bash
python scripts/train.py \
  --config configs/cpu-fast.yaml \
  --device cpu \
  --epochs 1
```

The CPU-fast profile is an intentionally different development architecture. Its checkpoints are not interchangeable with normal Tiny checkpoints.

## Developer CLI

The top-level CLI is intentionally developer/research oriented.

```text
lapis
└── dev
    ├── train
    ├── evaluate
    ├── generate
    ├── chat
    ├── inspect
    ├── benchmark
    ├── checkpoint inspect
    └── publish
```

Common commands:

```bash
lapis dev train
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "A language model learns by"
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
lapis dev checkpoint inspect --checkpoint checkpoints/latest.pt
```

A lower-level script interface remains available for direct control and compatibility. New integrations and documentation should prefer `lapis dev ...` unless a lower-level script is specifically useful.

## Runtime API

Applications such as CHAD should use the stable public runtime instead of importing internal model modules.

```python
from lapis.inference import LapisRuntime, SamplingConfig

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")

response = runtime.generate(
    "Explain DNS.",
    sampling=SamplingConfig(
        max_new_tokens=128,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
    ),
)
```

The runtime surface currently exposes:

- checkpoint-backed model loading;
- tokenizer/model compatibility checks;
- tokenization;
- text generation;
- streaming generation;
- model metadata such as context length, vocabulary size, parameter count, device, and tokenizer version.

Inference loading uses PyTorch's restricted `weights_only=True` mode. Runtime generation validates sampling parameters and rejects non-finite logits or probabilities rather than silently continuing with corrupted numerical state.

## Generation and evaluation

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

Generate from a checkpoint with the lower-level generator:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/latest.pt \
  --prompt "The transformer architecture" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

Generation is an engine capability. It is not a consumer chat feature. The runtime handles EOS termination, context limits, empty prompts, long prompts, sampling validation, deterministic configuration where supported, device/dtype handling, and checkpoint-backed generation.

## Developer inference console

For interactive debugging:

```bash
lapis dev chat --checkpoint checkpoints/latest.pt
```

This console exists for checkpoint smoke tests, tokenizer/prompt debugging, sampling experiments, CPU/CUDA verification, regression investigation, and rapid local experimentation.

It is intentionally **not** a user product and must not acquire accounts, onboarding, consumer persistence, profiles, or other application-layer responsibilities.

## Training data

The repository includes a bounded streaming collector for a mixed open/public corpus. The current source registry includes:

| Domain | Current sources |
|---|---|
| General web | FineWeb, C4 |
| Educational web | FineWeb-Edu, Cosmopedia |
| Knowledge | Wikipedia |
| Research | S2ORC arXiv |
| Books | Project Gutenberg filtered corpus |
| Mathematics | Open Web Math, OpenR1 Math |
| Conversational | OpenAssistant |
| Code | The Stack Smol |

The collector normalizes and cleans text, enforces global and per-source budgets, skips unusable samples, and records dataset/config/split/field statistics in a machine-readable manifest.

These public/open sources are **Lapis's engineering inputs**; their inclusion does not imply that they are used to train Claude, GPT, or any other proprietary model. Anthropic's private training corpus is not publicly disclosed by Anthropic.

Because dataset licenses and terms vary, preserve upstream provenance and review the applicable license before redistributing generated corpora or model weights.

## Checkpoints and reproducibility

Checkpoint integrity is part of model correctness.

A valid development run should be tied to the configuration, tokenizer, data provenance, and random state needed to understand or reproduce the result. Training infrastructure tracks optimizer/scheduler and RNG state for checkpointing, while inference additionally verifies tokenizer compatibility before constructing the model.

For inference, the expected layout is a checkpoint together with its matching tokenizer directory:

```text
checkpoints/
└── latest.pt
    └── ../tokenizer/
```

Do not treat a checkpoint as trusted executable input. Use the safe inference loading path for untrusted model artifacts; training-resume paths with broader serialization requirements must only consume trusted checkpoints.

## Learning Monitor

The Learning Monitor makes training behavior visible while weights are changing. Depending on configuration, it records signals such as:

- training loss;
- perplexity;
- learning rate;
- tokens seen;
- generated samples from fixed prompts.

This supports regression investigation and experiment comparison without turning the repository into a consumer analytics platform.

## Testing and quality gates

Testing is a lifecycle, not a single green badge.

```text
change
  ↓
smallest relevant test
  ↓
affected regression suite
  ↓
repository-wide tests
  ↓
CI matrix and specialized checks
  ↓
final commit verification
```

Normal local validation:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

For training changes, also run a Tiny CPU smoke test and evaluate the resulting checkpoint:

```bash
python -m scripts.train --config configs/tiny.yaml --epochs 1 --device cpu
python -m scripts.evaluate --checkpoint checkpoints/latest.pt --device cpu
```

Bug fixes should normally follow:

```text
reproduce → regression test → confirm failure → smallest fix → confirm pass → broader validation
```

Important risk areas include tensor shapes, broadcasting, devices, dtypes, masking and causality, RoPE, tokenizer/model vocabulary compatibility, NaN/Inf propagation, gradients, RNG state, context limits, EOS behavior, sampling parameters, checkpoint compatibility, packaging, and CPU/CUDA behavior.

See [`docs/testing.md`](docs/testing.md) for the detailed repository policy.

## CI, pull requests, and releases

The repository uses GitHub Actions as the automated validation layer for pull requests and the main branch. CI covers multiple Python versions, the test suite, coverage, Tiny CPU training/evaluation, linting, packaging/wheel installation, and security-oriented checks.

The intended workflow is:

```text
inspect repository state
        ↓
make focused change
        ↓
run targeted validation
        ↓
open PR
        ↓
CI + review
        ↓
fix failures / regressions
        ↓
re-run affected checks
        ↓
merge
        ↓
post-merge regression monitoring
```

A successful earlier commit does not validate a later changed commit. Final validation must correspond to the final code state.

Before contributing, read [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/testing.md`](docs/testing.md), and [`SECURITY.md`](SECURITY.md).

## Colab

Google Colab is a supported development environment for model experiments.

A basic setup is:

```bash
%cd /content/LapisLLM
!git pull --ff-only origin main
!pip install -e ".[dev]"
```

Use the same developer interfaces as local development:

```bash
!lapis dev train
```

Colab is development infrastructure, not a replacement for the repository CI contract. Large generated corpora and checkpoints remain local unless explicitly needed for an experiment or release workflow.

## Documentation map

| Resource | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Engineering contract, architecture boundaries, safety rules, and agent workflow |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution, testing, PR, and ownership workflow |
| [`docs/testing.md`](docs/testing.md) | Detailed validation lifecycle and risk-based testing policy |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting and checkpoint/secret handling |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history and known project evolution |
| [`configs/`](configs/) | Model, runtime, and experiment configurations |
| [`tests/`](tests/) | Correctness and regression coverage |
| [`website/`](website/) | Developer-facing project site |

## Repository layout

```text
LapisLLM/
├── lapis/
│   ├── config/            # configuration models and validation
│   ├── data/              # dataset/data preparation pipeline
│   ├── dev/               # explicit developer CLI namespace
│   ├── inference/         # checkpoint-backed public runtime
│   ├── model/              # decoder-only Transformer implementation
│   ├── tokenizer/          # tokenizer implementation/training
│   ├── training/          # training, checkpoints, optimization, monitoring
│   ├── ui/                # developer training-console components
│   ├── cli.py             # top-level CLI entry point
│   └── logging.py         # logging helpers
├── scripts/               # executable workflows and compatibility entry points
├── configs/               # YAML model/training/runtime configuration
├── tests/                 # correctness and regression tests
├── docs/                  # technical documentation
├── website/               # GitHub Pages developer site
├── training_history/      # lightweight verified experiment records
├── .agents/               # agent operating system and skills
├── AGENTS.md              # repository engineering contract
├── CONTRIBUTING.md        # contribution workflow
├── SECURITY.md            # security policy
├── CHANGELOG.md           # release history
├── LICENSE                # MIT license
└── README.md
```

## Reliability and security

Lapis treats source code, configuration, checkpoints, datasets, logs, model outputs, issue/PR text, and external documents as data that must be validated according to context.

The project explicitly rejects:

- committing credentials or private keys;
- unsafe model deserialization when a safe compatible path exists;
- arbitrary code execution through configuration;
- silently accepting invalid numerical parameters;
- weakening tests or security controls just to make CI green;
- unsupported claims about benchmark quality or model capability.

See [`SECURITY.md`](SECURITY.md) for the reporting policy and checkpoint guidance.

## Current limitations

Lapis is intentionally incomplete. The current engineering gaps include:

- exact mid-epoch dataloader/sampler replay;
- stronger training-resume semantics;
- mixed-precision training;
- distributed and multi-GPU training;
- production-scale data sharding, deduplication, and processing;
- efficient KV-cache generation;
- broad benchmark coverage;
- performance regression tracking;
- production serving hardening;
- large-scale model releases.

These are explicit roadmap items, not hidden claims of maturity.

## Roadmap

### Foundation

- [x] Decoder-only Transformer
- [x] ByteLevel BPE tokenizer
- [x] RoPE
- [x] GQA
- [x] RMSNorm
- [x] SwiGLU
- [x] Checkpoint save/load
- [x] Generation

### Reliability

- [x] Configuration validation
- [x] CPU-fast development profile
- [x] Learning Monitor
- [x] Experiment history
- [x] Checkpoint validation gates
- [x] Colab workflow
- [ ] Exact mid-epoch replay
- [ ] Stronger resume semantics

### Inference and scale

- [ ] Efficient KV-cache generation
- [ ] Mixed-precision optimization
- [ ] Distributed / multi-GPU training
- [ ] Larger model configurations
- [ ] Broader benchmark suite
- [ ] Performance regression tracking
- [ ] Production serving hardening

### Post-training and export

- [ ] Instruction tuning
- [ ] Preference optimization
- [ ] Expanded safety/evaluation workflows
- [ ] Safetensors/export pipeline
- [ ] Quantization architecture
- [ ] GGUF/Ollama-compatible packaging where supported

## License

LapisLLM is released under the MIT License. See [`LICENSE`](LICENSE).
