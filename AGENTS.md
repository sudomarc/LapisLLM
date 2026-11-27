# Agent instructions for LapisLLM

This is the canonical repository-wide policy for coding agents. Read it before inspecting or changing files. CLAUDE.md adds tool-specific guidance but must not override this document.

## Project contract

LapisLLM is an independent, from-scratch coding language-model project. It owns tokenizer training, data ingestion and provenance, model architecture, pre-training and later post-training, checkpointing, evaluation, local inference, developer tooling, packaging, tests and technical documentation.

The base model must begin from random initialization. Do not silently introduce a third-party pretrained model, hosted inference API or external model dependency as a substitute for LapisLLM's own weights.

The initial objective is software-development assistance. This is an experimental project; do not claim parity with frontier models without strong, reproducible evidence.

Keep consumer applications and agent orchestration outside this repository. LapisLLM should expose a product-agnostic model/inference boundary.

## Repository status and source of truth

The current default branch is a documentation-first clean slate. At the time this baseline is written, it has no implementation code, package manifest, notebooks or CI workflows. The roadmap describes future work.

Historical commits and deleted files are reference material only. Do not claim that an old implementation, checkpoint, data manifest or workflow is supported by the current branch. Before porting historical code, inspect it, verify its dependencies and tests, and record the decision.

Repository code, committed configurations, tests and reproducible runtime evidence are authoritative. Separate verified facts, hypotheses and unverified proposals.

## Mandatory workflow

Before every change:

1. Read this file and the relevant documents under docs/.
2. Read CLAUDE.md when operating through Claude Code.
3. Inspect the current tree, git status, relevant source and tests; never assume a planned file exists.
4. Read the linked issue and convert it into observable acceptance criteria.
5. Establish scope, risk and verification commands.
6. Make the smallest coherent change.
7. Add or update tests for changed behavior.
8. Update documentation when contracts, commands, configs or workflows change.
9. Run targeted checks, then the relevant regression suite.
10. Inspect the final diff and git status. Report actual evidence and remaining uncertainty.

If an issue, roadmap, historical document and current code disagree, stop and reconcile the conflict before implementation.

## Engineering priorities

Priorities: correctness and reproducibility; data and credential safety; numerical/runtime safety; simplicity; maintainability; performance; convenience.

Trace actual behavior before modifying it. Fix root causes. Avoid unrelated refactors, speculative abstractions and blind rewrites. Never weaken tests or hide failures to obtain green CI.

## ML correctness

Treat tensor dimensions, masks, causal attention, RoPE, normalization, MLP semantics, tokenizer/model compatibility, dtype conversion, gradient scaling, device placement, NaN/Inf propagation, gradients, seed handling, data leakage, optimizer/scheduler state and checkpoint resume as high-risk areas.

Validate architecture combinations and fail with actionable errors. Do not silently change model semantics.

## Colab and training

Colab runtimes are temporary and accelerator availability is dynamic. Every training workflow must tolerate interruption.

- Save resumable checkpoints outside ephemeral runtime storage at bounded intervals.
- Include model, optimizer, scheduler, scaler, RNG, sampler/data position and experiment metadata.
- Verify checkpoint integrity and compatibility before resume.
- Keep a previous known-good checkpoint where storage permits.
- Stage intensive writes on local runtime disk, then copy complete artifacts to durable storage.
- Do not write every batch directly to mounted Drive.
- Do not claim bitwise replay across different hardware/software environments unless proven.
- GitHub Actions runs CPU CI and bounded smoke tests; normal CI is not the GPU pre-training service.
- Do not bypass Colab limits, quotas or usage policies.
- Build manual Colab training before optional API automation.

Never commit raw corpora, model checkpoints, runtime tokens, GitHub tokens, private code or notebook outputs containing credentials.

## Data and licensing

Every training source needs documented origin, immutable revision or retrieval date, license evidence, intended use, redistribution constraints, field mapping, filters and a reproducible manifest.

Unknown or incompatible licensing is blocked by default. Public availability does not imply permission to train on or redistribute content. Never ingest private repositories, credentials, secrets or prohibited personal data.

Keep train, validation and test partitions isolated. Deduplicate to reduce leakage. Do not train on hidden benchmark tests or reserved evaluation answers.

## Artifact safety

Prefer safetensors for distributable inference weights. Never load untrusted pickle-based checkpoints with unrestricted deserialization. Training-resume files may contain executable serialization payloads and must be loaded only from trusted, integrity-checked sources.

Every artifact must identify model config, tokenizer, source revision, dataset manifests, training config and checksums. Model weights are not automatically covered by the source-code license.

## Verification and reporting

Never claim that tests, benchmarks, training, deployment or publication succeeded unless they ran and results were inspected.

A completion report must identify changes, exact commands run, results, artifacts produced, checks not run and remaining uncertainty. A lower training loss alone is not evidence of useful coding ability.
