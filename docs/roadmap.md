# LapisLLM roadmap

This roadmap is evidence-gated, not calendar-based. Progress depends on the actual PC hardware, data rights, measured throughput and evaluation results.

## Step 0 — Governance and project scope

- [x] Record the from-scratch coding-model direction.
- [x] Decide that training runs on the maintainer's own PC.
- [x] Establish agent and contribution instructions.
- [x] Document proposed architecture, data policy and evaluation policy.
- [ ] Decide model-weight license before public weight distribution.
- [ ] Audit the target PC and storage capacity.

**Exit gate:** documented project contract and a measured hardware report.

## Step 1 — Python package and CI foundation

- [ ] Create minimal Python package and typed configuration.
- [ ] Pin supported Python and PyTorch versions.
- [ ] Add lint, unit tests and package-build checks.
- [ ] Add CPU-only GitHub Actions CI.
- [ ] Add documentation/config validation.
- [ ] Exclude datasets, runs, checkpoints, credentials and local environments from Git.

**Exit gate:** clean install and deterministic CPU tests on a fresh runner.

## Step 2 — Local hardware discovery and runtime

- [ ] Implement hardware diagnostics for OS, CPU, RAM, accelerator, VRAM, drivers and free disk.
- [ ] Detect supported PyTorch device backends.
- [ ] Add explicit device and precision configuration.
- [ ] Fail clearly when a requested accelerator is unavailable.
- [ ] Add a bounded model throughput and peak-memory benchmark.
- [ ] Record an approved local Micro profile based on measured hardware.

**Exit gate:** a reproducible hardware report and a validated training configuration for the maintainer's PC.

## Step 3 — Dataset registry and corpus preparation

- [ ] Define dataset-source schema and approval states.
- [ ] Record canonical source, immutable revision, license, intended use and attribution.
- [ ] Implement download/ingestion with bounded retries.
- [ ] Implement normalization, secret filtering, quality filters and deduplication.
- [ ] Validate train/validation/test separation and leakage.
- [ ] Store raw and processed corpus outside Git.
- [ ] Build a small local development corpus and manifests.

**Exit gate:** reproducible, legally reviewed local corpus with provenance.

## Step 4 — Tokenizer

- [ ] Train code-oriented, byte-aware BPE on the training split only.
- [ ] Save tokenizer and metadata.
- [ ] Implement encode/decode and save/load tests.
- [ ] Evaluate tokens-per-file and language coverage.
- [ ] Bind tokenizer identity to model configs and checkpoints.

**Exit gate:** deterministic tokenizer artifact compatible with the model vocabulary.

## Step 5 — Lapis Micro architecture

- [ ] Implement decoder-only Transformer from random initialization.
- [ ] Implement causal attention, RoPE, RMSNorm and SwiGLU.
- [ ] Add parameter counting and config validation.
- [ ] Add shape, causality, gradient and numerical tests.
- [ ] Verify parameter count against the selected local Micro profile.

**Exit gate:** correct CPU forward/backward pass and finite loss on a tiny synthetic batch.

## Step 6 — Training engine

- [ ] Implement next-token prediction and loss.
- [ ] Add AdamW, warmup/cosine schedule, accumulation and clipping.
- [ ] Add device-aware FP32/BF16/FP16 behavior.
- [ ] Add progress metrics: loss, tokens, steps, step time and throughput.
- [ ] Add bounded validation and early smoke-run support.

**Exit gate:** tiny local run learns a small corpus without non-finite values.

## Step 7 — Checkpointing and recovery

- [ ] Save model, optimizer, scheduler, scaler and RNG states.
- [ ] Persist data position and experiment manifest.
- [ ] Implement safe local writes and checksum verification.
- [ ] Implement resume compatibility validation.
- [ ] Test Ctrl+C, forced process termination and restart.
- [ ] Retain previous known-good checkpoints.

**Exit gate:** a training run can be stopped and resumed without silently restarting or corrupting state.

## Step 8 — Local training workflow

- [ ] Implement local train command and config-driven run IDs.
- [ ] Implement local data/output roots and free-space checks.
- [ ] Add preflight before expensive work.
- [ ] Add graceful stop and useful console progress.
- [ ] Ensure logs and artifacts remain outside Git.
- [ ] Document the exact local commands for Windows and Linux.

**Exit gate:** the maintainer can launch, observe, stop and resume Micro training entirely on the PC.

## Step 9 — Evaluation baseline

- [ ] Implement validation loss/perplexity.
- [ ] Build versioned internal coding tasks.
- [ ] Add compile/test-based scoring in an isolated sandbox.
- [ ] Record model/tokenizer/data/config hashes.
- [ ] Publish baseline metrics and failure analysis.

**Exit gate:** Micro has measured, reproducible quality and performance baselines.

## Step 10 — First end-to-end Micro experiment

- [ ] Use the reviewed local corpus.
- [ ] Initialize model weights randomly.
- [ ] Train on the maintainer's PC.
- [ ] Save and verify checkpoints.
- [ ] Resume after a deliberate interruption.
- [ ] Evaluate held-out loss and coding tasks.
- [ ] Publish an experiment report with hardware, time, tokens, throughput, memory and limitations.

**Exit gate:** end-to-end training and recovery proven on the target PC.

## Step 11 — Coding post-training and inference

- [ ] Define instruction dataset/provenance schema.
- [ ] Fine-tune LapisLLM's own base checkpoint.
- [ ] Implement local inference and developer CLI.
- [ ] Evaluate instructions, completions, edits and generated tests.
- [ ] Compare pre-training and post-training behavior for regressions.

**Exit gate:** local model generation is usable for bounded coding tasks and limitations are documented.

## Step 12 — Model release and future scaling

- [ ] Resolve model-weight license.
- [ ] Publish model card, training lineage, provenance and evaluation.
- [ ] Publish checksummed weights only when licensing permits.
- [ ] Expand model size only after measured local hardware and compute budgets support it.
- [ ] Consider remote/distributed compute only through a new explicit project decision.

**Exit gate:** a reproducible, legally reviewed model release.

## Non-goals

No hosted notebook training, no assumed cloud GPU, no frontier-parity claim, no billion-parameter target without a compute plan, no unreviewed corpus and no GPU dependency in normal CI. Consumer chat UX and agent orchestration remain outside LapisLLM.

Parameter count alone is not a reason to scale.
