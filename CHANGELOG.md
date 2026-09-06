# Changelog

All notable changes to LapisLLM are documented here.

## [0.1.2] - 2026-09-06

### Fixed

- Made configuration loading independent of the current working directory.
- Restored the declared `prepare_data` package entry point with deterministic UTF-8 text preparation.
- Validated model dimensions, RoPE head dimensions, token ranges, target ranges, and invalid all-padding targets before execution.
- Removed duplicate dropout application in the SwiGLU path.
- Made explicit CUDA requests fail clearly when CUDA is unavailable instead of silently switching devices.
- Removed silent CPU dtype coercion; unsupported CPU dtypes now fail early.
- Hardened checkpoint loading with required metadata, version checks, strict state loading, atomic writes, tokenizer consistency checks, and deterministic epoch/batch resume metadata.
- Hardened generation with parameter validation, greedy `temperature=0`, prompt context validation, and tokenizer/model compatibility checks.
- Removed generated checkpoint artifacts from version control.

### Tests and CI

- Added regression tests for configuration, model validation, generation edge cases, deterministic data ordering, checkpoint metadata/persistence, and the restored data-preparation CLI.
- CI now verifies bytecode compilation, Ruff, the test suite, wheel construction, installed CLI entry points, CPU training, resume, evaluation, and generation smoke tests.

### Documentation

- Updated the release status to distinguish correctness/reproducibility work from benchmark or capability claims.
- `configs/tiny.yaml` remains the canonical development profile; larger model names remain roadmap targets unless explicitly released and evaluated.

### Known limitations

- Exact replay is implemented for the current single-process deterministic data loader, but distributed and multi-worker replay is not supported.
- Mixed-precision training, distributed training, efficient KV-cache generation, and comprehensive benchmark evaluation remain out of scope for this release.

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
