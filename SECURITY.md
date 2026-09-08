# Security Policy

## Supported versions

Security fixes target the current development branch and the latest published release.

## Reporting vulnerabilities

Do not publish credentials, private keys, or working exploit details in a public issue. Use GitHub's private security advisory/reporting flow for `sudomarc/LapisLLM`.

## Model and checkpoint safety

Lapis checkpoints are executable input to the model runtime and must be treated as untrusted files. Current runtime code uses PyTorch `weights_only=True` for checkpoint loading. Do not bypass this protection for untrusted files.

Training checkpoints may contain optimizer, scheduler, configuration, and RNG state. Resume only from checkpoints you trust.

## Secrets

Never commit API keys, tokens, passwords, private keys, `.env` files, or credentials. Generated checkpoints, corpora, and runtime outputs are intentionally excluded by `.gitignore`.
