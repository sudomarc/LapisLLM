# LapisLLM Testing & Regression Policy

Testing in LapisLLM is a lifecycle that spans local development, pull requests, merge, and post-merge maintenance.

The goal is not simply to obtain a green test command. The goal is to establish evidence that a change preserves correctness across the execution paths and boundaries it can affect.

## Before a pull request

Use the smallest useful validation first, then widen the scope according to risk.

For normal Python changes:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

For a bug fix, add a regression test that fails before the fix and passes after it. Then run the affected regression suite and broader repository validation.

## Pull request CI

GitHub Actions is the authoritative automated validation layer for the PR.

After opening a PR, inspect every required check. Investigate failed, cancelled, skipped, or unexpectedly missing checks. After any meaningful change, rerun targeted tests and the affected regression suite and validate the new final commit.

Do not rely on stale CI results from an earlier commit.

## Risk-based validation

### Model and training

Validate the path `input -> tokenizer -> model -> logits -> loss -> backpropagation` and consider tensor shapes, device/dtype behavior, causal masking, RoPE, vocabulary compatibility, numerical stability, gradients, reproducibility, checkpoint compatibility, and CPU/CUDA behavior when available.

### Colab training workflow

The canonical Colab path is non-interactive and must be testable without a notebook UI. Regression coverage should prove that:

- the canonical entry point has no `input()` dependency;
- the default run count is deterministic and does not prompt for a choice;
- progress remains observable during quiet phases through logs/heartbeat output rather than keyboard input;
- in-training generation is disabled by default so monitoring cannot accidentally dominate optimizer throughput;
- the corpus budget is distributed across configured sources instead of allowing the first source to exhaust the global budget;
- a cached corpus is rejected when its manifest no longer satisfies the current builder contract;
- transient source failures retry within a bounded budget and failed partial source output is removed before retrying;
- checkpoint and paired-tokenizer validation occurs before publication;
- Git authentication failure is non-interactive and fails clearly instead of waiting indefinitely for terminal input; and
- generated preview/evaluation runs happen only after the training job reaches its verified terminal state unless an explicit developer option requests otherwise.

For performance-sensitive training changes, verify actual optimizer progress (`step`, loss, token count, and throughput) rather than treating console activity as proof of training.

### Inference and generation

Test empty prompts, long prompts, context limits, EOS/stop behavior, token limits, sampling controls, deterministic seeds, invalid settings, and checkpoint-backed inference where supported.

### Configuration

Test valid and invalid boundaries, including zero, negative, minimum/maximum values, missing paths, impossible architecture combinations, `NaN`, `+Inf`, and `-Inf` wherever relevant.

### Checkpoints and serialization

Validate save/load, missing or corrupt files, architecture/tokenizer mismatch, metadata compatibility, device/dtype behavior, and resume behavior where supported.

### Packaging and APIs

Validate build/install behavior and public imports. For runtime/API changes, test request validation, dispatch, error handling, streaming where applicable, and compatibility with external consumers such as CHAD.

## Edge cases

Consider empty and malformed input, missing/corrupt resources, boundary values, wrong dtype/device/shape, extreme values, NaN/infinities, repeated execution, deterministic execution, context boundaries, EOS behavior, and dependency/version differences where applicable.

## Before merge

The final PR state must satisfy the repository's required checks. Review the complete final diff, identify the final commit, confirm required checks are green for that commit, and explicitly disposition unresolved failures or skipped checks.

## After merge

Fast post-merge validation should protect `main`. More expensive scheduled validation may cover expanded Python compatibility, extended inference/generation tests, longer training smoke tests, property-based or fuzz tests, dependency/security audits, packaging verification, and benchmark/regression monitoring.

A post-merge failure is a repository regression signal and must be investigated.

## Test maintenance

Tests are part of the repository contract. Do not delete a real regression test, weaken assertions, silently increase numerical tolerances, skip validation without documenting why, or disable checks merely to make CI green.

Do not claim model quality, benchmark improvements, training success, or test success without reproducible evidence.