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

## Required checks

Run the following before opening a pull request:

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

## Pull requests

All changes to the official project must arrive through a pull request.
Direct pushes to `main` are reserved for the project owner and authorized
maintainers.

A pull request should:

1. describe the reason for the change;
2. include tests for changed behavior;
3. update documentation/changelog when appropriate;
4. contain no credentials or private runtime artifacts; and
5. confirm acceptance of `CONTRIBUTOR_AGREEMENT.md` in the pull-request template.

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
