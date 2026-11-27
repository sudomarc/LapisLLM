# Agent instructions for LapisLLM

This is the canonical repository-wide policy for coding agents. Read it before inspecting or changing files. CLAUDE.md adds tool-specific guidance but must not override this document.

## 1. Project contract

LapisLLM is an independent, from-scratch coding language-model project. It owns tokenizer training, data ingestion and provenance, model architecture, pre-training and later post-training, checkpointing, evaluation, local inference, developer tooling, packaging, tests and documentation.

The base model must begin from random initialization. Do not silently introduce a third-party pretrained model, hosted inference API or external model dependency as a substitute for LapisLLM's own weights.

The initial objective is software-development assistance. This is an experimental project; do not claim parity with frontier models without strong, reproducible evidence.

Keep consumer applications and agent orchestration outside this repository. LapisLLM should expose a product-agnostic model/inference boundary.

## 2. Source of truth and current status

The current default branch is a documentation-first clean slate. At this baseline, it has no implementation code, package manifest, training scripts or CI workflows. The roadmap describes future work.

Historical commits are reference material only. Do not claim that old implementations, checkpoints or workflows are supported by the current branch. Before porting historical code, inspect it, verify dependencies and tests, and record the decision.

Repository code, committed configurations, tests and reproducible local execution are authoritative. Separate verified facts, hypotheses and unverified proposals.

## 3. Mandatory workflow

Before every change:

1. Read this file and relevant documents under docs/.
2. Read CLAUDE.md when operating through Claude Code.
3. Inspect the current tree, git status, relevant source and tests; never assume a planned file exists.
4. Read the linked issue and convert it into observable acceptance criteria.
5. Establish scope, risks and verification commands.
6. Make the smallest coherent change.
7. Add or update tests for changed behavior.
8. Update documentation when contracts, commands, configs or workflows change.
9. Run targeted checks, then the relevant regression suite.
10. Inspect final diff and status. Report actual evidence and remaining uncertainty.

If an issue, roadmap, historical document and current code disagree, stop and reconcile the conflict before implementation.

## 4. Engineering priorities

Priorities: correctness and reproducibility; data and credential safety; numerical/runtime safety; simplicity; maintainability; performance; convenience.

Trace actual behavior before modifying it. Fix root causes. Avoid unrelated refactors, speculative abstractions and blind rewrites. Never weaken tests or hide failures to obtain green CI.

## 5. Local hardware and training rules

Training is performed on the maintainer's own PC. Do not add hosted training services or rely on external GPU sessions.

- Hardware is currently unspecified. Do not assume a particular CPU, GPU, VRAM, RAM, operating system or storage capacity.
- Implement a hardware diagnostic before freezing training profiles.
- Detect available PyTorch backends at runtime. NVIDIA CUDA is a likely first accelerator path; CPU must remain supported for tests and tiny experiments. Other backends are opt-in only after verified tests.
- Never silently fall back from a requested accelerator to CPU for a long run. Fail clearly or require an explicit fallback.
- Measure peak memory and tokens/second with a bounded benchmark before recommending a model size or batch configuration.
- Keep datasets, caches, runs and checkpoints on local storage by default, outside Git.
- Check free disk space before corpus preparation and training.
- Save complete checkpoints at bounded intervals. Handle Ctrl+C and process termination safely.
- Never write credentials, private source code or private data into logs or artifacts.
- GitHub Actions is for CPU CI and bounded smoke tests, not model pre-training.

Do not promise that a useful coding assistant can be trained on the local PC until its hardware, data budget and measured throughput are known.

## 6. ML correctness

Treat tensor dimensions, masks, causal attention, RoPE, normalization, MLP semantics, tokenizer/model compatibility, dtype conversion, gradient scaling, device placement, NaN/Inf propagation, gradient clipping, random seeds, data leakage and checkpoint resume as high-risk areas.

Validate architecture combinations and fail with actionable errors. Do not silently change model semantics.

## 7. Dataset and licensing rules

Every training source needs documented origin, immutable revision or retrieval date, license evidence, intended use, redistribution constraints, field mapping, filters and a reproducible manifest.

Unknown or incompatible licensing is blocked by default. Public availability does not imply permission to train on or redistribute content. Never ingest private repositories, credentials, secrets or prohibited personal data.

Keep train, validation and test partitions isolated. Deduplicate to reduce leakage. Do not train on hidden benchmark tests or reserved evaluation answers.

## 8. Artifact safety

Prefer safetensors for distributable inference weights. Never load untrusted pickle-based checkpoints with unrestricted deserialization. Training-resume files may contain executable serialization payloads and must be loaded only from trusted, integrity-checked sources.

Every artifact must identify model config, tokenizer, source revision, dataset manifests, training config and checksums. Model weights are not automatically covered by the source-code license.

## 9. Verification and reporting

Never claim that tests, benchmarks, training, deployment or publication succeeded unless they ran and results were inspected.

A completion report must identify changes, exact commands run, results, artifacts produced, checks not run and remaining uncertainty. A lower training loss alone is not evidence of useful coding ability.
