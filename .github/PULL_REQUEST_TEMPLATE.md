## Summary

Describe what this pull request changes and why.

## Scope

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Model/training change
- [ ] Inference/runtime change
- [ ] CLI/developer tooling change
- [ ] Packaging/dependency change
- [ ] Documentation-only change
- [ ] Security change

## Validation

### Regression coverage

- [ ] Existing tests cover the changed behavior.
- [ ] Tests were added or updated for changed behavior.
- [ ] Bug fixes include a regression test where practical.
- [ ] Relevant edge cases were considered.

### Checks performed

- [ ] Targeted tests pass.
- [ ] Affected regression tests pass.
- [ ] Full test suite passes when applicable.
- [ ] `ruff check .` passes.
- [ ] Security/dependency checks pass when applicable.
- [ ] Packaging/import checks pass when applicable.
- [ ] CPU validation passes when applicable.
- [ ] CUDA validation passes when applicable.
- [ ] Training smoke test passes when applicable.
- [ ] Inference/generation smoke test passes when applicable.

### Evidence

List the actual commands and checks that were run, with their results.

```text
<command or GitHub check>
<result>
```

### CI state

- [ ] All required GitHub Actions checks pass.
- [ ] No required check is missing or unexpectedly skipped.
- [ ] Failures were investigated rather than ignored.
- [ ] The final commit has been validated.

### Risk / regression impact

Describe what existing behavior could be affected and any compatibility,
numerical, performance, model, training, inference, API, packaging, or security risk.

### Skipped checks / uncertainty

Document checks that could not be run, why they were skipped, and any remaining uncertainty.

### Post-merge validation

- [ ] No additional post-merge validation required.
- [ ] Scheduled regression coverage should exercise this change.
- [ ] Extended model/training validation is relevant.
- [ ] Benchmark/regression monitoring is relevant.
- [ ] Additional security review is relevant.

## Documentation

- [ ] Documentation updated when behavior or workflow changed.
- [ ] Changelog updated when required.

## Contribution rights

- [ ] I have read and agree to `CONTRIBUTOR_AGREEMENT.md`.
- [ ] I have the legal right to submit every part of this contribution.
- [ ] I have identified all third-party code, datasets, assets, and licenses used by this PR.
- [ ] I understand that acceptance into official LapisLLM may require the copyright assignment or fallback license described in `CONTRIBUTOR_AGREEMENT.md`.

## Ownership and branding

This PR changes the code/documentation of the official upstream project only.
It does not grant permission to use the LapisLLM brand, name, logos, or official
release identity outside the rules in `TRADEMARKS.md`.
