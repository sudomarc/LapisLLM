# Changelog

## [v0.1.0] - 2026-09-06

### 🚀 Initial Release: LapisLLM Stack

#### Core Components
- **Tokenizer**: Training and versioned tokenizer artifacts for efficient text encoding
- **Data Pipeline**: End-to-end data processing including cleaning, deduplication, packing, and manifest generation
- **Transformer Model**: Decoder-only architecture implemented in pure Python with KV cache support for efficient inference
- **Training Framework**: Complete training loop with checkpointing (weights, optimizer state, scheduler, RNG), logging, and evaluation utilities
- **Export Tools**: Multi-format export support (safetensors, GGUF) for model portability
- **Inference Engine**: Sampling, generation, and chat primitives with KV cache optimization
- **Serving API**: Lightweight local serving API with streaming support for inference

#### Features
- ✅ Configuration-driven workflows (YAML configs for all runs)
- ✅ Transparent implementations (no external model binaries)
- ✅ Reproducible checkpoints with full state preservation
- ✅ Test coverage for core components
- ✅ Modular architecture with clear separation of concerns
- ✅ Ergonomic developer UX with CLI entrypoints

#### Project Status
- **Active development** with modular, well-documented codebase
- Designed for educational purposes and custom LLM implementations
- Ready for development and experimentation

#### Getting Started
```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
```

#### Quick Commands
- Train tokenizer: `python scripts/train_tokenizer.py --config configs/tiny.yaml`
- Prepare data: `python scripts/prepare_data.py --config configs/development.yaml`
- Train model: `python scripts/train.py --config configs/development.yaml`
- Evaluate: `python scripts/evaluate.py --checkpoint checkpoints/latest`
- Generate: `python scripts/generate.py --checkpoint checkpoints/latest --prompt "Hello"`
- Chat: `python scripts/chat.py --checkpoint checkpoints/latest`
- Serve: `python scripts/serve.py --checkpoint checkpoints/latest --host 0.0.0.0 --port 8080`

#### Repository Structure
- `src/lapis/tokenizer/` - Tokenization implementation
- `src/lapis/data/` - Data processing pipeline
- `src/lapis/model/` - Transformer architecture
- `src/lapis/training/` - Training utilities and checkpointing
- `src/lapis/inference/` - Generation and inference primitives
- `src/lapis/export/` - Model export tools
- `src/lapis/api/` - Serving application
- `scripts/` - CLI entrypoints
- `configs/` - Example configurations
- `docs/` - Architecture and design documentation
- `tests/` - Unit and integration tests

#### License
MIT License - see LICENSE file for details

#### Contributing
- Read `docs/` and `AGENTS.md` for architecture guidelines
- Keep changes focused, documented, and tested
- Report issues with clear reproduction steps

---

**Maintainer**: [@sudomarc](https://github.com/sudomarc)
**Repository**: https://github.com/sudomarc/LapisLLM
