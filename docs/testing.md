# LapisLLM Testing & Regression Policy

Testing in LapisLLM is a lifecycle that spans local development, pull requests, merge, and post-merge maintenance.

The goal is not simply to obtain a green test command. The goal is to establish evidence that a change preserves correctness across the execution paths and boundaries it can affect.

## Testing levels

Use the smallest useful test first, then widen validation according to risk.

```text
single regression test
        ↓
targeted test module
        ↓
affected regression suite
        ↓
repository test suite
        ↓
CI matrix / specialized checks
```

Do not jump directly to expensive validation when a narrow test can expose a failure more quickly.

## Before opening a PR

Every meaningful change should have a validation plan before the PR is opened.

### Normal Python changes

Run:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

### Bug fixes

A bug fix should normally produce a regression test that fails before the fix and passes after it.

```text
reproduce
  ↓
regression test
  ↓
confirm failure
  ↓
smallest correct fix
  ↓
confirm same test passes
  ↓
affected regression suite
  ↓
broader validation
```

The test should protect the behavior that was actually broken, not merely exercise nearby code.

### Model and training changes

In addition to normal tests, validate the training path with a tiny deterministic run when practical.

```bash
python -m scripts.train --config configs/tiny.yaml --epochs 1 --device cpu
python -m scripts.evaluate --checkpoint checkpoints/latest.pt --device cpu
```

A useful micro-test covers:

```text
input → tokenizer → model → logits → loss → backpropagation
```

### Inference and generation changes

Validate checkpoint-backed inference and, where applicable:

- empty prompts;
- long prompts and context limits;
- EOS and stopping behavior;
- maximum token limits;
- sampling parameters;
- deterministic seeds;
- invalid or non-finite parameters;
- CPU and CUDA behavior when available.

### Configuration changes

Test both valid and invalid configuration values, especially boundary and non-finite values.

Where relevant, cover:

- zero and negative values;
- minimum and maximum supported values;
- `NaN`;
- `+Inf` and `-Inf`;
- impossible architecture combinations;
- invalid paths and missing files.

### Checkpoint and serialization changes

Validate:

- save and load;
- missing files;
- corrupted files;
- architecture mismatch;
- tokenizer mismatch;
- metadata compatibility;
- device and dtype behavior;
- resume behavior where supported.

### Packaging and API changes

Validate build/install behavior and public imports when packaging or public interfaces change.

For API/runtime changes, test request validation, runtime dispatch, error handling, streaming behavior where applicable, and compatibility with external consumers such as CHAD.

## Pull request CI

GitHub Actions is the authoritative automated validation layer for the PR.

A PR should be considered ready only after the actual required checks for its final commit have completed successfully.

CI failures must be investigated. A failure may be:

- introduced by the PR;
- pre-existing;
- environment-specific;
- flaky; or
- caused by a missing/incorrect CI configuration.

Do not assume the cause without evidence.

If a commit changes after CI succeeds, the new commit requires new validation.

## Required PR evidence

A PR description should state:

1. what behavior changed;
2. which tests were added or changed;
3. which targeted tests were run;
4. which broader checks were run;
5. which relevant checks were not run and why;
6. what regression or compatibility risk remains.

Use command output and GitHub check results as evidence rather than statements such as “tests should pass”.

## Edge-case expectations

When applicable, explicitly consider:

- empty input;
- malformed input;
- missing resources;
- corrupt resources;
- boundary lengths and values;
- wrong dtype;
- wrong device;
- wrong tensor shapes;
- negative values;
- zero values;
- extremely large or small values;
- `NaN` and infinities;
- repeated execution;
- deterministic execution;
- context-length boundaries;
- EOS behavior;
- dependency/version differences.

## After a PR is opened

After opening a PR:

```text
CI starts
   ↓
inspect every required check
   ↓
fix failures introduced by the PR
   ↓
rerun targeted tests
   ↓
rerun affected regression tests
   ↓
confirm final CI state
```

Do not merge based on stale results from an earlier commit.

Unexpectedly missing or skipped required checks are themselves a validation problem.

## Before merge

Before merging:

- the final diff must be reviewed;
- the final commit must be identified;
- required checks for that final commit must be green;
- unresolved test failures must have an explicit disposition;
- the PR must not contain secrets or unintended runtime artifacts.

## After merge

The same project standards continue on `main`.

Fast post-merge validation should cover the core repository contract. Expensive validation can run separately when it is not appropriate for every PR.

Useful scheduled checks include:

- expanded Python compatibility;
- extended inference/generation regression tests;
- longer training smoke tests;
- property-based tests;
- fuzz tests;
- dependency vulnerability audits;
- packaging/install verification;
- performance and benchmark regression detection.

A post-merge failure is not automatically “someone else’s problem”. It is evidence that the repository state requires investigation.

## ML-specific regression discipline

For changes to numerical or model code, prioritize correctness over speed.

Check:

- tensor shapes and broadcasting;
- causal masking;
- RoPE parameters and caches;
- vocabulary/tokenizer compatibility;
- numerical stability;
- NaN/Inf propagation;
- gradient behavior;
- RNG state and reproducibility;
- context limits;
- checkpoint compatibility.

Do not claim improved model quality, benchmark results, or training success without reproducible evidence.

## Test maintenance

When a test exposes a real regression, keep the test unless the underlying contract intentionally changes.

Do not:

- delete a failing regression test to make CI green;
- weaken assertions solely to accommodate a bug;
- skip a test without documenting the reason;
- silently increase tolerances to hide numerical regressions;
- disable CI checks because they are inconvenient.

Tests are part of the repository contract.
