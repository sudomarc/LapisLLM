# LAPIS — AGENTS.md

## What LAPIS is

LAPIS is a research and engineering project to build our own Large Language Model (LLM) from scratch.
This is not an application wrapping OpenAI, Claude, Gemini, Llama, Qwen, Mistral, Gemma, or any external model.
We are building our own neural network, tokenizer pipeline, data pipeline, training system, inference system, and distribution infrastructure.

## Repository structure

```
lapis/
├── README.md
├── LICENSE
├── pyproject.toml
├ .gitignore
├── .env.example
├── AGENTS.md

├── configs/
│   ├── development.yaml
│   ├── tiny.yaml
│   ├── small.yaml
│   └── base.yaml

├── src/
│   └── lapis/
│       ├── __init__.py
│       ├── version.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── model_config.py
│       │   ├── training_config.py
│       │   └── data_config.py
│       │
│       ├── tokenizer/
│       │   ├── __init__.py
│       │   ├── tokenizer.py
│       │   ├── train.py
│       │   └── special_tokens.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── sources.py
│       │   ├── downloader.py
│       │   ├── cleaner.py
│       │   ├── filters.py
│       │   ├── deduplication.py
│       │   ├── packing.py
│       │   ├── dataset.py
│       │   └── manifests.py
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── rope.py
│       │   ├── attention.py
│       │   ├── normalization.py
│       │   ├── mlp.py
│       │   ├── block.py
│       │   ├── transformer.py
│       │   └── lapis_model.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   ├── checkpointing.py
│       │   ├── precision.py
│       │   ├── distributed.py
│       │   └── metrics.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── perplexity.py
│       │   ├── evaluator.py
│       │   └── benchmarks.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── generate.py
│       │   ├── sampling.py
│       │   ├── kv_cache.py
│       │   └── chat.py
│       │
│       ├── export/
│       │   ├── __init__.py
│       │   ├── safetensors.py
│       │   └── gguf.py
│       │
│       └── api/
│           ├── __init__.py
│           ├── app.py
│           ├── routes.py
│           ├── schemas.py
│           └── streaming.py

├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── chat.py
│   ├── export.py
│   └── serve.py

├── tests/
│   ├── tokenizer/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── inference/
│   └── api/

├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── deduplicated/
│   ├── tokenized/
│   └── manifests/

├── checkpoints/
├── artifacts/
├── outputs/
├── logs/
└── runs/

└── docs/
    ├── architecture.md
    ├── tokenizer.md
    ├── dataset.md
    ├── training.md
    ├── evaluation.md
    ├── inference.md
    ├── export.md
    ├── api.md
    └── roadmap.md
```

## Architecture rules

- The project must be: modular, reproducible, configurable, testable, scalable, documented, maintainable
- Priority order: correctness > simplicity > modularity > optimization
- Do not prematurely optimize
- Do not add complex frameworks without reason
- Do not create temporary code that becomes permanent
- The LLM must be a real decoder-only causal Transformer
- Architecture must be fully configurable
- Never hardcode model parameter counts
- The tokenizer must be versioned with the model
- Each dataset preparation must produce a manifest
- Never automatically ingest a source with unknown license
- Training objective: causal language modeling (cross-entropy)
- Checkpoints must store: model weights, optimizer state, scheduler state, global step, epoch, training configuration, random states, tokenizer reference
- Use safetensors for weights when possible
- API must be standard enough to be consumed by: web apps, desktop apps, mobile apps, IDEs, bots, agents, CLI
- The client interface must never know internal Transformer details
- Export pipeline: native checkpoint → safetensors → GGUF / Ollama packaging (when compatible)
- Ollama is only a possible runtime, not LAPIS itself
- Quantization: provide abstraction for FP32, FP16, BF16, INT8, INT4 (start with extensibility)
- Scalability: 1 GPU → multiple GPUs → distributed training → large-scale training
- Abstraction for: DDP, FSDP, and other strategies later
- Memory optimization: gradient checkpointing, activation checkpointing, mixed precision, memory-efficient attention, gradient accumulation (planned later)
- Performance order: correctness → profiling → optimization
- Never sacrifice correctness for performance
- Hardware detection at startup: OS, CPU, RAM, GPU, VRAM, CUDA, PyTorch version
- System must work without GPU
- Project must remain executable on a small machine for: unit tests, architecture tests, tokenizer experiments, tiny model experiments, debugging
- Larger trainings can be done on GPU cloud
- Google Colab / Cloud: architecture must be executable via: local development → Git → cloud GPU → training → checkpoint → download → local inference
- Colab is only a possible runtime, not a dependency
- Versioning: Semantic Versioning (0.1.0, 0.2.0, 1.0.0)
- Models use their own version (e.g., Lapis-Tiny-v0.1)
- Each public model must generate a Model Card with: name, version, architecture, parameters, context length, tokenizer, training data, data licenses, training procedure, evaluation, known limitations, intended use, unintended use, safety considerations, hardware

## Agent responsibilities

1. **LAPIS-ARCHITECT** — global architecture, interfaces, decoupling, scalability, design decisions
2. **LAPIS-ML** — Transformer architecture, attention, RoPE, RMSNorm, MLP, embeddings, language modeling head
3. **LAPIS-DATA** — dataset ingestion, cleaning, filtering, deduplication, tokenization pipeline, dataset manifests
4. **LAPIS-TRAINING** — optimizer, scheduler, training loop, mixed precision, checkpointing, resume, distributed training
5. **LAPIS-EVAL** — validation, perplexity, benchmarks, metrics, evaluation methodology
6. **LAPIS-INFERENCE** — generation, sampling, KV cache, streaming, inference optimizations
7. **LAPIS-API** — FastAPI, schemas, streaming, model serving, API compatibility
8. **LAPIS-SYSTEMS** — CUDA, GPU memory, distributed runtime, performance, hardware detection
9. **LAPIS-SECURITY** — dependency safety, API security, secret handling, model distribution security
10. **LAPIS-DEBUG** — reproduction, root cause analysis, minimal fixes, regression prevention
11. **LAPIS-DOCUMENTATION** — README, architecture docs, training docs, model cards, API docs, release notes

## Coding rules

- Use Python 3.11+
- Use PyTorch, NumPy, safetensors, FastAPI, Pydantic, pytest, Ruff, Git
- Add other dependencies only when they bring clear value
- CUDA support must be planned
- CPU must remain usable for development and small tests
- Never hardcode model parameters — framework must compute them automatically
- Tokenizer must support: encode(), decode(), batch_encode(), save(), load()
- Tokenizer must provide: BOS, EOS, PAD, UNK as needed
- Training loop must manage: forward, loss, backward, gradient accumulation, gradient clipping, optimizer step, scheduler step, checkpointing, logging, validation
- Support FP32, FP16, BF16 optionally per hardware
- Every critical component must have tests: Tokenizer, Encoding, Decoding, RMSNorm, RoPE, Attention, Causal Mask, MLP, Transformer Block, Full Model, Loss, Generation, Sampling, Checkpoint Save, Checkpoint Load, Resume
- Test tensor dimensions
- Micro-test: random input → model → logits → loss → backpropagation
- Mini-training overfit on tiny dataset to demonstrate learning pipeline works
- Determinism: support seed, deterministic mode, record seed in experiments
- Export: native checkpoint → safetensors → quantization → GGUF / Ollama packaging support
- Quantization abstraction: FP32, FP16, BF16, INT8, INT4 (start with extensibility)
- Do not simulate capabilities without signaling it
- Do not create fake metrics, benchmark results, training results, or model capabilities

## Testing rules

- Minimum tests per component: Tokenizer, Encoding, Decoding, RMSNorm, RoPE, Attention, Causal Mask, MLP, Transformer Block, Full Model, Loss, Generation, Sampling, Checkpoint Save, Checkpoint Load, Resume
- Test tensor dimensions
- Micro-test: random input → model → logits → loss → backpropagation
- Mini-training overfit on tiny dataset

## Git rules

- Commit often with descriptive messages
- Never commit secrets
- Write concise commit messages matching repo style
- Before committing, inspect `git status`, `git diff`, `git log --oneline -10`
- Stage only intended files
- Do not update git config, skip hooks, use interactive `-i`, force-push, or create empty commits
- Before creating a PR, inspect status, diff, remote tracking, recent commits, and diff from base branch
- Review all commits included in the PR, not just the latest

## Naming conventions

- Project: LAPIS
- Models: Lapis Tiny, Lapis Small, Lapis 1B, Lapis 3B, Lapis 7B
- Software versions: Lapis 0.1.0, Lapis 0.2.0, Lapis 1.0.0
- Do not call the project: MyLLM, GPT clone, Llama clone
- Config files: base.yaml, tiny.yaml, small.yaml, development.yaml
- Scripts: train_tokenizer.py, prepare_data.py, train.py, evaluate.py, generate.py, chat.py, serve.py
- Tests directory: tests/ with subdirectories per component

## Performance principles

- correctness → profiling → optimization
- Never sacrifice correctness for performance
- Do not prematurely optimize
- Profile before optimizing

## Security principles

- Dependency safety
- API security
- Secret handling (never hardcode keys, use .env)
- Model distribution security

## Roadmap

### PHASE 0 — FOUNDATION

Repository, configuration, logging, CLI, tests, documentation, AGENTS.md

### PHASE 1 — TOKENIZER

Tokenizer training, encoding, decoding, special tokens, serialization, versioning

### PHASE 2 — DATA

Loading, cleaning, filtering, deduplication, manifests, packing, sharding

### PHASE 3 — LAPIS MODEL

Embeddings, RoPE, RMSNorm, causal attention, MLP, Transformer block, Transformer stack, LM head

### PHASE 4 — TRAINING

Loss, optimizer, scheduler, gradient accumulation, mixed precision, checkpointing, resume, logging, validation

### PHASE 5 — INFERENCE

Generation, temperature, top-k, top-p, KV cache, streaming

### PHASE 6 — API

/v1/models, /v1/completions, /v1/chat/completions, streaming

### PHASE 7 — EXPORT

safetensors, quantization architecture, GGUF pipeline, Ollama packaging support

### PHASE 8 — SCALING

multi-GPU, distributed training, FSDP, gradient checkpointing, performance optimization

### PHASE 9 — POST-TRAINING

base model → instruction tuning → preference optimization → safety tuning → evaluation
DO NOT start by this phase