# LapisLLM Testing & Regression Policy

This file is the repository reference for testing across local development, pull requests, merge, and post-merge maintenance.

Testing is a lifecycle. A green local command is evidence for one environment and commit; it is not a substitute for validating the final PR state.

## Before a pull request

Use the smallest useful validation first, then widen the scope according to risk.

```text
single regression test
        -> targeted test module
        -> affected regression suite
        -> repository test suite
        -> CI matrix / specialized checks
```

For normal Python changes, run:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

For training-stack changes, also run the Tiny CPU training and evaluation smoke tests documented in `CONTRIBUTING.md`.

## Bug fixes

A bug fix should normally add a regression test that fails before the fix and passes after it.

```text
reproduce
  -> add regression test
  -> confirm failure
  -> implement smallest correct fix
  -> confirm same test passes
  -> run affected regression suite
  -> run broader validation
```

Do not weaken, delete, or rewrite a test only to make an implementation pass.

## Risk-based validation

### Model and training changes

Validate the path:

```text
input -> tokenizer -> model -> logits -> loss -> backpropagation
```

Also consider tensor shapes, device placement, dtype conversion, causal masks, RoPE, vocabulary compatibility, numerical stability, gradient behavior, RNG/reproducibility, checkpoint compatibility, and CPU/CUDA behavior when available.

### Inference and generation changes

Where supported, test empty prompts, long prompts, context limits, EOS/stop behavior, token limits, sampling controls, deterministic seeds, invalid settings, and checkpoint-backed inference.

### Configuration changes

Test valid values and invalid boundaries, including zero, negative, minimum/maximum limits, missing paths, impossible architecture combinations, `NaN`, `+Inf`, and `-Inf` wherever those values can reach runtime or training behavior.

### Checkpoint/serialization changes

Validate save/load, missing and corrupt files, architecture and tokenizer mismatch, metadata compatibility, device/dtype behavior, and resume behavior where supported.

### Packaging/API changes

Validate build/install behavior and public imports. For runtime/API changes, cover request validation, dispatch, errors, streaming where applicable, and compatibility with external consumers such as CHAD.

## Pull request CI

GitHub Actions is the authoritative automated layer for the PR.

After a PR is opened:

```text
CI starts
   -> inspect every required check
   -> investigate failures
   -> fix failures caused by the PR
   -> rerun targeted tests
   -> rerun affected regression tests
   -> confirm final CI state
```

Classify failures from evidence. A failure may be introduced by the PR, pre-existing, environment-specific, flaky, or caused by CI configuration.

If the PR changes after a successful run, the new final commit requires fresh validation. Do not rely on stale checks.

Unexpectedly missing or skipped required checks are themselves a validation problem.

## PR evidence

Every meaningful PR should identify:

- what behavior changed;
- what tests were added or changed;
- targeted tests executed;
- broader checks executed;
- checks not executed and why;
- relevant regression, compatibility, numerical, performance, model, training, inference, API, packaging, or security risk;
- remaining uncertainty.

Prefer actual command output and GitHub check results over statements such as “tests should pass”.

## Edge cases

Consider, where relevant:

- empty and malformed input;
- missing and corrupt resources;
- boundary lengths and values;
- zero and negative values;
- very small and very large values;
- wrong dtype;
- wrong device;
- wrong tensor shape;
- NaN and infinities;
- repeated execution;
- deterministic execution;
- context-length boundaries;
- EOS behavior;
- dependency/version differences.

## Before merge

The final PR state must satisfy the repository's required checks.

Before merge:

- review the complete final diff;
- identify the final commit;
- confirm required checks are green for that commit;
- resolve or explicitly disposition test failures;
- ensure no secrets or unintended runtime artifacts are included.

## After merge

Post-merge validation protects `main` after integration.

Fast checks should cover the core repository contract on pushes to `main`. More expensive validation may run on a schedule, including:

- expanded Python compatibility;
- extended inference/generation regression tests;
- longer training smoke tests;
- property-based tests;
- fuzz tests;
- dependency/security audits;
- packaging/install verification;
- performance and benchmark regression detection.

A post-merge failure is a repository regression signal and must be investigated.

## Test maintenance rules

Tests are part of the repository contract.

Do not:

- delete a real regression test to make CI green;
- weaken assertions solely to accommodate a bug;
- silently increase numerical tolerances to hide a regression;
- skip validation without documenting why;
- disable CI checks because they are inconvenient.

Do not claim model quality, benchmark improvements, training success, or test success without reproducible evidence.
