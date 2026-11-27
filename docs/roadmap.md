# LapisLLM roadmap

This roadmap is evidence-gated, not calendar-based. Reprioritize when hardware availability, data quality or evaluation makes a stage infeasible.

## Phase 0 — Governance and foundation
- [x] Record the from-scratch coding-model direction.
- [x] Establish repository and agent instructions.
- [x] Document proposed architecture and Micro profile.
- [x] Define Colab, data, checkpoint, evaluation and security policies.
- [ ] Decide an explicit model-weight license.
- [ ] Confirm initial data/legal review and implementation issues.

**Exit gate:** reviewed project contract and testable initial configuration.

## Phase 1 — Package and CI
- [ ] Create minimal Python package and typed configuration.
- [ ] Pin supported Python/PyTorch versions.
- [ ] Add lint, unit tests, package build and docs checks.
- [ ] Add CPU-only CI and bounded Micro smoke test.
- [ ] Exclude generated artifacts and secrets.

**Exit gate:** clean install and deterministic CPU tests on a fresh runner.

## Phase 2 — Data and tokenizer
- [ ] Add dataset registry and source approval state.
- [ ] Implement normalized records and provenance manifests.
- [ ] Add license, secret and privacy checks.
- [ ] Implement cleaning, filtering and deduplication.
- [ ] Validate train/validation/test isolation.
- [ ] Train a byte-aware code BPE tokenizer.
- [ ] Save/load tokenizer with compatibility metadata.

**Exit gate:** reproducible preparation of a small reviewed corpus without raw corpus data in Git.

## Phase 3 — Lapis Micro model
- [ ] Implement decoder-only Transformer.
- [ ] Implement causal attention, RoPE, RMSNorm and SwiGLU.
- [ ] Add config validation and parameter counting.
- [ ] Add forward, causality, gradient and numerical tests.
- [ ] Confirm actual Micro parameter count.

**Exit gate:** CPU tests pass and a tiny next-token batch produces finite loss and gradients.

## Phase 4 — Training and recovery
- [ ] Implement next-token training loop and AdamW.
- [ ] Add warmup/cosine schedule, accumulation and clipping.
- [ ] Add hardware-aware precision selection.
- [ ] Add progress metrics.
- [ ] Implement resumable checkpoint state.
- [ ] Test interruption, restore and incompatibility failures.

**Exit gate:** a bounded run resumes from the last complete checkpoint with a complete manifest.

## Phase 5 — Colab
- [ ] Implement thin notebook entry point.
- [ ] Add hardware preflight and VRAM detection.
- [ ] Add explicit config and run IDs.
- [ ] Implement durable artifact-store adapter.
- [ ] Add Drive persistence for hosted notebooks.
- [ ] Test forced termination and recovery.
- [ ] Record actual hardware, throughput and cost observations.

**Exit gate:** a Micro experiment completes across one or more Colab sessions with verified recovery.

## Phase 6 — Coding evaluation
- [ ] Implement validation loss/perplexity.
- [ ] Build versioned internal coding tasks.
- [ ] Add compile/test scoring in an isolated sandbox.
- [ ] Add sample review and failure taxonomy.
- [ ] Publish reproducible experiment reports.

**Exit gate:** Micro has a measured baseline and documented failures.

## Phase 7 — Lapis Small
- [ ] Profile memory and tokens per second on actual hardware.
- [ ] Estimate token, step and accelerator budgets.
- [ ] Expand only the reviewed data mixture.
- [ ] Train the 30–50M profile if the measured budget is viable.
- [ ] Compare Micro and Small on the same evaluation harness.

**Exit gate:** scale-up yields measured benefit at an understood compute cost.

## Phase 8 — Coding post-training
- [ ] Define instruction data schema and provenance.
- [ ] Build code instruction/edit datasets.
- [ ] Run supervised fine-tuning on Lapis-owned weights.
- [ ] Evaluate instruction following, code edits and test generation.
- [ ] Consider preference optimization only with suitable data and evaluation.

**Exit gate:** post-training improves predefined coding tasks without unacceptable base-capability regression.

## Phase 9 — Inference and developer UX
- [ ] Implement local checkpoint loading and generation.
- [ ] Add developer CLI.
- [ ] Publish model metadata format and context limits.
- [ ] Add memory-safe inference.
- [ ] Document external application integration.

**Exit gate:** local coding-model runtime is usable independently.

## Phase 10 — Release and scale
- [ ] Resolve model-weight license.
- [ ] Publish model card, provenance and evaluation report.
- [ ] Publish checksummed weights in an appropriate artifact registry.
- [ ] Add release compatibility/regression gates.
- [ ] Consider distributed training only after single-GPU evidence.

**Exit gate:** a reproducible and legally reviewed model release.

## First-release non-goals

No frontier-parity claim, billion-parameter training on free Colab, unreviewed scraped corpus, persistent Colab dependency, GPU requirement for normal CI, or consumer agent orchestration inside LapisLLM.

Parameter count alone is not a reason to scale.
