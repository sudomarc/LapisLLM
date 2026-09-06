# LapisLLM

**Lapis is a from-scratch decoder-only Transformer language model and training stack written in Python/PyTorch.**

The project is designed to make the mechanics of an LLM inspectable: tokenizer, data pipeline, Transformer architecture, optimization, checkpointing, evaluation, and inference are developed as explicit components rather than hidden behind a large framework.

[![Tests](https://img.shields.io/github/actions/workflow/status/sudomarc/LapisLLM/tests.yml?branch=main&label=tests)](https://github.com/sudomarc/LapisLLM/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Project status:** `Lapis 0.1.1 — Correctness`  
> The current milestone focuses on making the training core correct, testable, and reproducible. Lapis is **not yet a competitive pretrained foundation model**.

---

## What is Lapis?

LapisLLM is an attempt to build an LLM stack from first principles while keeping the implementation small enough to study and modify.

The core model is a causal Transformer with:

- decoder-only autoregressive training
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward blocks
- configurable model dimensions
- ByteLevel BPE tokenization
- PyTorch training and checkpointing

The goal is not to hide complexity. The goal is to expose it.

## Current release

### Lapis 0.1.1 — Correctness

This release concentrates on the parts that must be right before scaling the model.

**Included**

- corrected causal target alignment
- padding-aware loss masking
- linear warmup followed by cosine decay
- optimizer, scheduler, model, and RNG checkpoint state
- tokenizer artifacts stored with checkpoints
- correct propagation of the model `bias` configuration
- tests for causality, GQA shapes, RoPE behavior, loss masking, tokenizer round-trips, and dataset alignment
- reproducible local training configuration

**Next milestone: 0.1.2 — Validation**

- validation loss and perplexity
- tiny-dataset overfitting experiment
- deterministic checkpoint round-trip tests
- generation regression tests
- stronger configuration validation
- documented training experiments

No benchmark or capability claims are made until those experiments exist.

---

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
┌──────────────────────────────┐
│ Transformer block × N        │
│                              │
│ RMSNorm                      │
│   └─ GQA + RoPE              │
│ Residual                     │
│                              │
│ RMSNorm                      │
│   └─ SwiGLU MLP              │
│ Residual                     │
└──────────────────────────────┘
 │
 ▼
RMSNorm
 │
 ▼
LM head
 │
 ▼
Next-token logits
```

### Attention

Lapis uses **Grouped Query Attention**: multiple query heads share key/value heads. The number of query heads must be divisible by the number of key/value heads.

### Position encoding

The model uses **RoPE** rather than learned absolute position embeddings. Rotary frequencies are precomputed up to `max_position_embeddings`.

### Language-model objective

For an input sequence:

```text
x₀ x₁ x₂ x₃ ... xₙ
```

the model predicts:

```text
    x₁ x₂ x₃ ... xₙ
```

The dataset keeps the complete `seq_len + 1` window; the model performs the causal shift exactly once. Padding targets are represented by `-100` and ignored by cross-entropy.

---

## Model configurations

The repository currently provides several YAML configurations. The important development configuration is `configs/tiny.yaml`:

| Parameter | Tiny |
|---|---:|
| Vocabulary | 256 |
| Hidden size | 128 |
| Layers | 2 |
| Attention heads | 4 |
| KV heads | 2 |
| FFN size | 256 |
| Context | 128 tokens |
| Dropout | 0.0 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Warmup | 10 steps |
| Training budget | 200 steps |

The exact instantiated parameter count is reported by the training script rather than being hard-coded in the documentation.

---

## Quick start

### Requirements

- Python 3.11+
- PyTorch 2.0+
- a few GB of disk space for a development environment
- CPU is sufficient for the tiny development configuration; CUDA can be used when available

### Install

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM

python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e .
```

### Run the test suite

```bash
python -m pytest
```

### Train the tiny model

```bash
python scripts/train.py --config configs/tiny.yaml
```

The smoke training path is self-contained and uses the built-in development corpus when no `--data` file is supplied.

### Train on your own text

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --data ./data/train.txt
```

### Resume a checkpoint

```bash
python scripts/train.py \
  --config configs/tiny.yaml \
  --resume checkpoints/latest.pt
```

---

## Repository layout

```text
LapisLLM/
├── lapis/
│   ├── config/          # Model, training, and data configuration
│   ├── data/            # Cleaning, filtering, packing, manifests
│   ├── model/           # Transformer implementation
│   └── tokenizer/       # ByteLevel BPE tokenizer
├── scripts/
│   ├── train.py         # Training entry point
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── chat.py
│   └── serve.py
├── configs/             # Reproducible YAML experiments
├── tests/               # Unit and correctness tests
├── docs/                # Technical documentation
├── .github/workflows/   # CI and GitHub Pages workflows
├── CHANGELOG.md
├── AGENTS.md
└── README.md
```

---

## Reproducibility

Lapis treats reproducibility as a training-system requirement rather than a README claim.

Checkpoints currently preserve:

- model weights
- optimizer state
- scheduler state
- training step
- epoch
- model/training configuration
- tokenizer version and tokenizer files
- Python RNG state
- PyTorch RNG state
- CUDA RNG state when CUDA is available

Training also accepts an explicit seed through `--seed`.

Exact mid-epoch replay is still being hardened; the next validation milestone will address data-loader/sampler state explicitly.

---

## Data

The data pipeline contains separate components for:

- source ingestion
- text cleaning
- quality filtering
- duplicate removal
- sequence packing
- deterministic manifests

The current filtering and deduplication logic is intended for development and experimentation. It should **not** be described as a production-scale Common Crawl pipeline yet.

---

## Evaluation philosophy

A decreasing training loss is not enough to claim that an LLM works.

Lapis will therefore validate progress in stages:

1. **Correctness** — tensor shapes, causality, masking, tokenizer behavior, gradients.
2. **Learning** — the tiny model must overfit a deliberately small dataset.
3. **Generalization** — validation loss and perplexity on held-out text.
4. **Generation** — deterministic and sampling-based generation regression tests.
5. **Scaling** — larger configurations only after the previous stages are reproducible.

Until these stages are complete, Lapis makes no claims about reasoning, coding ability, factual knowledge, or benchmark performance.

---

## Roadmap

### 0.1 — Foundation

- [x] Transformer architecture
- [x] ByteLevel BPE tokenizer
- [x] causal language-model loss
- [x] GQA + RoPE + SwiGLU
- [x] checkpointing
- [x] correctness-focused tests

### 0.1.2 — Validation

- [ ] validation split
- [ ] perplexity reporting
- [ ] tiny-overfit experiment
- [ ] deterministic checkpoint round-trip
- [ ] generation regression suite
- [ ] stronger config validation

### 0.2 — Small model

- [ ] larger validated configuration
- [ ] mixed precision
- [ ] efficient KV cache
- [ ] gradient checkpointing
- [ ] dataset/checkpoint sharding

### Later

- [ ] distributed training
- [ ] larger-scale pretraining
- [ ] broader evaluation suite
- [ ] model export and interoperability

---

## Design principles

### Small enough to understand

Lapis intentionally avoids hiding the model behind a high-level training framework during the early milestones.

### Correct before fast

Optimization and scaling come after the training objective, masking, checkpointing, and tests are trustworthy.

### Configuration over hard-coded experiments

Model and training dimensions live in YAML configurations so experiments can be compared and reproduced.

### Evidence over claims

Benchmarks, parameter counts, and capability statements should come from reproducible experiments, not estimates in documentation.

---

## Known limitations

Lapis is an experimental research/learning project. In particular:

- the current tiny configuration is not intended to produce high-quality general-purpose text
- the built-in corpus is only a smoke-training corpus
- data filtering is not yet production-grade
- exact mid-epoch resume is not fully deterministic
- large-scale distributed training is not implemented
- model quality has not yet been established through a comprehensive benchmark suite

These limitations are intentional parts of the current development stage, not hidden behind a production-style README.

---

## Contributing

Contributions should preserve the project's development discipline:

1. keep changes focused
2. add or update tests for behavioral changes
3. run `python -m pytest`
4. run `ruff check .`
5. document architecture or behavior changes

See [`AGENTS.md`](AGENTS.md) for repository-specific engineering guidance.

---

## License

LapisLLM is released under the MIT License. See [`LICENSE`](LICENSE).

---

## Acknowledgements

Lapis is informed by the open-source LLM ecosystem, including the design and documentation practices of projects such as Meta Llama, OpenAI's open-weight work, Hugging Face Transformers, and nanoGPT.

The project does not copy model weights or training data from those projects.

## Citation

```bibtex
@software{lapisllm2026,
  author = {sudomarc},
  title = {LapisLLM: A From-Scratch Decoder-Only Transformer Language Model},
  year = {2026},
  url = {https://github.com/sudomarc/LapisLLM}
}
```

---

**Lapis 0.1.1 — Correctness**  
Build the foundation first. Scale it second.
