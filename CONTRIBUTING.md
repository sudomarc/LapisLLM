# Contributing to LapisLLM

LapisLLM is an open-source project with an owner-controlled upstream repository.
Anyone may propose improvements, but the official `main` branch, official
releases, project governance, and project branding remain controlled by the
project owner.

## Before contributing

Read:

- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `CONTRIBUTOR_AGREEMENT.md`
- `TRADEMARKS.md`
- `AGENTS.md`
- `.agents/bootstrap.md`

For accepted contributions, the project uses the ownership terms described in
`CONTRIBUTOR_AGREEMENT.md`. This is intended to keep the official upstream
project legally maintainable by one project owner while still allowing public
open-source participation.

## Local setup

```bash
git clone https://github.com/sudomarc/LapisLLM.git
cd LapisLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Testing policy

Testing is a lifecycle, not a single command. The full testing policy is documented in `docs/testing.md`.

Before opening a pull request, run the narrowest relevant tests first, then the affected regression suite, then the repository-wide checks required for the change.

For normal Python changes:

```bash
python -m pytest --cov=lapis --cov-report=term-missing
ruff check .
python -m pip_audit
```

For changes to the training stack, also run a Tiny CPU smoke test and evaluate
the resulting checkpoint:

```bash
python -m scripts.train --config configs/tiny.yaml --epochs 1 --device cpu
python -m scripts.evaluate --checkpoint checkpoints/latest.pt --device cpu
```

For model, inference, runtime, configuration, packaging, security, or device-sensitive changes, run the additional validation described in `docs/testing.md`.

### Regression-test rule

A bug fix should normally add a regression test that would have failed before the fix.

Preferred sequence:

```text
reproduce bug
    -> write regression test
    -> confirm failure
    -> implement smallest fix
    -> confirm the same test passes
    -> run affected regression suite
    -> run broader validation
```

Do not weaken or rewrite tests merely to make an implementation pass.

## Pull requests

All changes to the official project must arrive through a pull request.
Direct pushes to `main` are reserved for the project owner and authorized
maintainers.

A pull request should:

1. describe the reason for the change;
2. include tests for changed behavior;
3. identify the targeted and broader validation that was run;
4. document relevant risk, skipped checks, and remaining uncertainty;
5. update documentation/changelog when appropriate;
6. contain no credentials or private runtime artifacts; and
7. confirm acceptance of `CONTRIBUTOR_AGREEMENT.md` in the pull-request template.

### After opening the PR

A PR is not ready merely because local tests pass.

After CI starts, inspect the actual GitHub Actions results. Investigate failed,
cancelled, skipped, or unexpectedly missing required checks. Distinguish
pre-existing failures from regressions introduced by the PR.

After every meaningful fix, rerun the targeted tests and the affected regression
suite. Before merge, verify the final commit state and final CI results again.

Do not rely on stale CI results from an earlier commit.

### After merge

Merged code remains subject to regression validation.

Fast validation runs on pushes to `main`. Broader or expensive checks may run on
a schedule, including extended model/inference tests, packaging checks,
property-based or fuzz tests, dependency/security audits, compatibility checks,
and benchmark/regression monitoring.

A post-merge failure is a repository regression signal and must be investigated.

The project owner decides whether a contribution is accepted. Opening a PR,
being listed in commit history, or being acknowledged as a contributor does
not grant ownership or control of the LapisLLM project or brand.

## Repository control

The intended upstream workflow is:

```text
Contributor fork/branch
        |
        v
   Pull Request
        |
        v
 Owner / CODEOWNERS review
        |
   checks + review
        |
        v
 official main
        |
        v
 official release
```

Releases and other official project artifacts must originate from the official
repository and authorized maintainers.

## Datasets and models

Preserve source attribution and provenance metadata. Do not silently change
dataset licenses or source URLs; update the dataset registry and tests together.
Do not submit model weights or datasets unless their licensing and provenance
are documented and compatible with the project.

## Branding

Code may be forked under the applicable open-source license, but the LapisLLM
name, logo, and official-release identity are governed separately by
`TRADEMARKS.md`.

## Security

Do not report credentials or vulnerabilities in public issues. Use the
private security reporting process described in `SECURITY.md`.
