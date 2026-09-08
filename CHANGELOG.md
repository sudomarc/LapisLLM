# Changelog

All notable changes to LapisLLM are documented here.

## [Unreleased]

### Security

- Raised the minimum PyTorch version to 2.14.0 and current maintained FastAPI/Uvicorn/Pydantic floors.
- Switched checkpoint loading in the trainer, evaluator, generator, terminal chat, and HTTP serving runtime to `weights_only=True`.
- Added CI dependency vulnerability auditing and Git working-tree/history secret scans.
- Scoped GitHub Pages write and OIDC permissions to the deployment job.

### Performance

- Changed `TextDataset` to construct sample tensors lazily instead of materializing every dataset tensor during initialization.

### CI

- Added Python 3.11, 3.12, and 3.13 test matrix coverage.
- Added `pytest-cov` coverage reporting.
- Added `pip-audit` dependency checks.
- Added validation on pushes to `maintenance/**` branches.

### Documentation

- Added `CONTRIBUTING.md` and `SECURITY.md`.
- Removed unused direct `numpy` and `safetensors` dependencies from the runtime package metadata.

### GitHub administration remaining

- `main` branch protection/rulesets still require GitHub repository administration outside the connected integration. The current API access reports `main` as unprotected and no repository rulesets.

---

## [0.1.1] - 2026-09-06

### Added

- Transformer correctness tests covering forward/backward passes, parameter counts, GQA shapes, RoPE behavior, causality, and loss masking.
- ByteLevel BPE tokenizer round-trip tests.
- Dataset target-alignment and padding-mask tests.
- Checkpoint persistence for model, optimizer, scheduler, epoch, configuration, tokenizer, and RNG state.
- Linear warmup followed by cosine decay scheduling.
- A correctness-focused development configuration in `configs/tiny.yaml`.

### Fixed

- Corrected the double causal target shift between the dataset and model loss.
- Padding target positions are now ignored with `-100` during cross-entropy.
- Propagated the model `bias` setting into the SwiGLU MLP.
- Normalized partial gradient-accumulation groups so their update scale does not depend on the number of batches in an epoch.
- Updated stale tests to match the current tokenizer and data configuration.
- Replaced broad exception handling in the data cleaner with specific exceptions.

### Documentation

- Reworked the README into a model/project overview with explicit architecture, status, limitations, validation philosophy, and roadmap sections.
- Removed unsupported claims about large-scale training, GGUF export, KV-cache maturity, and production readiness.

### Validation status

CI covers automated correctness and lint checks. The release does **not** claim benchmark quality or general-purpose language capability. The next milestone is validation through tiny-dataset overfitting, held-out loss/perplexity, deterministic checkpoint tests, and generation regression tests.

### Not yet implemented

- Exact mid-epoch data-loader/sampler replay.
- Mixed-precision training.
- Distributed training.
- Efficient KV-cache generation.
- Large-scale pretraining.
- Comprehensive benchmark evaluation.
- Production-grade web-scale data processing.

---

## [0.1.0] - 2026-09-06

### Initial development release

- Initial LapisLLM repository structure.
- Decoder-only Transformer implementation.
- Configurable YAML model and training configurations.
- Tokenizer training and persistence.
- Data cleaning, filtering, packing, and manifest utilities.
- Training, evaluation, generation, chat, and local serving entry points.
- Initial test and CI infrastructure.

---

For the current state of the project, see [`README.md`](README.md).
