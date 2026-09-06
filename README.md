# LapisLLM

A compact, end-to-end implementation of an LLM stack in Python: tokenizer, data pipeline, decoder-only Transformer, training loop, inference primitives, export tooling, and a minimal serving API. LapisLLM is built for reproducibility, auditability, and extensibility — suitable for research and production prototyping.

Status
- Active. Modular components with configuration-driven workflows and test coverage.
- Goal: transparent implementations (no external model binaries), reproducible checkpoints, and ergonomic developer UX.

Highlights
- Tokenizer training and versioned tokenizer artifacts.
- Data ingestion: cleaning, deduplication, packing, and manifest generation.
- Decoder-only Transformer implemented in Python with KV cache support.
- Training utilities: checkpointing (weights, optimizer, scheduler, RNG), logging, and evaluation.
- Export: safetensors / GGUF exporters and lightweight serving API for local inference.
- Config-first: all runs controlled by YAML configs under configs/.

Quickstart (development)
1. Clone
   git clone https://github.com/sudomarc/LapisLLM.git
   cd LapisLLM

2. Environment
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

3. Configure
   cp .env.example .env
   edit .env and YAML configs in configs/ to suit your hardware and dataset.

4. Run tests
   pytest -q

Common commands
- Train tokenizer
  python scripts/train_tokenizer.py --config configs/tiny.yaml

- Prepare data (packing/manifests)
  python scripts/prepare_data.py --config configs/development.yaml

- Train model
  python scripts/train.py --config configs/development.yaml

- Evaluate
  python scripts/evaluate.py --checkpoint checkpoints/latest --config configs/development.yaml

- Generate / chat
  python scripts/generate.py --checkpoint checkpoints/latest --prompt "Hello"
  python scripts/chat.py --checkpoint checkpoints/latest

- Serve (local API)
  python scripts/serve.py --checkpoint checkpoints/latest --host 0.0.0.0 --port 8080

Configuration
- configs/ contains example YAMLs (development, tiny, small, base).
- Model, training, and data parameters are configurable in src/lapis/config/.
- Prefer small test configs for CI and rapid iteration.

Repository layout (high level)
- src/lapis/
  - tokenizer/: tokenizer implementation & training
  - data/: downloaders, cleaners, dedupe, packers, manifests
  - model/: transformer blocks and model wrapper
  - training/: trainer, optimizers, schedulers, checkpointing
  - inference/: sampling, generation, KV cache, chat primitives
  - export/: safetensors / GGUF exporters
  - api/: serving app and streaming utilities
- scripts/: CLI entrypoints (train, evaluate, generate, serve)
- configs/: example run configs
- docs/: design and operational documentation
- tests/: unit and integration tests

Checkpoints & artifacts
- Checkpoints must include model weights, optimizer & scheduler state, global step, RNG state, and tokenizer reference.
- Use safetensors where supported.

Contribution
- Read docs/ and AGENTS.md for architecture rules and conventions.
- Keep changes focused, documented, and covered by tests.
- Open issues and PRs with a concise description, rationale, and reproduction steps.

Security & licensing
- Do not add datasets with unclear licensing. See LICENSE for repo license.

Contact
- Repository: https://github.com/sudomarc/LapisLLM
- Maintainer: sudomarc
