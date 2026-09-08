# Changelog

All notable changes to LapisLLM are documented here.

## [Unreleased] - 2026-09-08

### Fixed

- Removed unused training-console imports that broke Ruff CI.
- Modernized Ruff configuration to use the `tool.ruff.lint` section.
- Moved `pytest` and `ruff` from runtime dependencies into the `dev` extra.
- Isolated dependency and secret audits into an independent CI security job so lint failures cannot suppress them.
- Restricted GitHub Pages write and OIDC permissions to the deploy job.
- Updated the website workflow to run on Node.js 24.
- Hardened terminal chat checkpoint loading with `weights_only=True`.
- Aligned package metadata with the current 0.2.x development line.

### Security

- CI continues to run dependency vulnerability and Git-history secret scans independently from the main test matrix.
- `tokenizers` remains an explicitly tracked security risk while the upstream fix for CVE-2026-85670 is unavailable; untrusted tokenizer artifacts must not be treated as trusted input.

### Remaining work

- Protect `main` with GitHub branch rulesets/required checks; this requires repository-administration access unavailable to the connected integration.
- Remove remaining `weights_only=False` checkpoint-loading paths from legacy training/Colab validation helpers.
- Replace the legacy `training_console_fixed.py` split with one canonical console module.
- Add a reproducible Python dependency lock and a dependency/SBOM review process.

## [0.2.0] - 2026-09-07

### Added

- Validation-focused engineering pass across the model, checkpointing, inference, configuration, testing, CI, Colab workflow, and developer platform.
- Website platform with model, skills, documentation, research, roadmap, changelog, and About surfaces.
- CPU-fast training profile for rapid local iteration.
- Dataset registry and registry-driven download pipeline with provenance metadata.
- Learning Monitor and full-screen Textual training console.

### Fixed

- Safer checkpoint loading and tokenizer/checkpoint compatibility validation across inference workflows.
- RoPE dtype handling after module conversion.
- Training and configuration consistency issues including effective batch-size validation and `prepare_data` compatibility.
- Generated training artifacts excluded from normal source commits.

### Validation

- Python 3.11, 3.12, and 3.13 CI matrix.
- Pytest, tiny CPU training smoke test, evaluation smoke test, Ruff, dependency audit, and secret scanning.

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
