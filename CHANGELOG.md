# Changelog

All notable changes to LapisLLM are documented here.

## Unreleased

### Fixed

- Removed interactive run-count input from the canonical Colab training workflow.
- Disabled in-training generated-text sampling by default so optimizer throughput is not consumed by repeated inference passes.
- Fixed Colab smoke-test corpus cache validation to use its actual 50,000-character budget.
- Distributed the Colab corpus character budget across configured sources so the first source cannot starve later sources.
- Added bounded Hugging Face Hub request timeouts to complement source retries.
- Made missing Colab GitHub authentication fail fast instead of waiting for interactive credentials.
- Separated full training/resume checkpoints from the slim inference checkpoint published for `LapisRuntime` and CHAD.

### Added

- Added regression coverage for the non-interactive Colab contract, corpus budgeting, stale-cache invalidation, smoke-test budget, and checkpoint artifact separation.
- Added a repository documentation gate requiring relevant documentation and applicable agent skills to be read before every repository change.

## [0.2.0] - 2026-09-08

### Fixed

- Isolated dependency and secret audits into a dedicated CI security job so test/lint failures cannot suppress security checks.
- Scoped GitHub Pages write and OIDC permissions to the deployment job only.
- Updated the website workflow to Node.js 24.
- Moved `pytest` and `ruff` to the development extra instead of requiring them for runtime installs.
- Packaged the `scripts` entry-point modules so wheel installations expose the same CLI targets as editable installs.

### Added

- Added a non-editable wheel build/import validation to CI, including checks for packaged CLI modules.
- Added an npm lockfile and switched website CI from `npm install` to reproducible `npm ci` installs.
- Added `SECURITY.md` with vulnerability-reporting, checkpoint-safety, and secret-handling guidance.
- Expanded generated-file and private-key patterns in `.gitignore` and CI secret scans.

### Notes

- The obsolete training-console implementation from PR #11 was not merged because `main` now uses the USER/DEV runtime split and its compatibility wrapper; the old `training_console_fixed.py` replacement would regress that architecture.
- Checkpoint loading paths that still require `weights_only=False` for training resume/validation remain explicitly tracked for a separate compatibility migration rather than being hidden by CI rules.

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
