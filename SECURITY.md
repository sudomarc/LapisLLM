# Security policy

## Supported versions

Until the first tagged release, only the current default branch is supported.

## Reporting a vulnerability

Do not disclose credentials, exploitable vulnerabilities or sensitive dataset material in public issues or pull requests. Use GitHub private vulnerability reporting for this repository when enabled. If it is unavailable, contact the repository owner through a private channel listed on the owner's GitHub profile.

Include the affected revision and component, reproduction steps, expected and observed behavior, impact and environment. Redact secrets and personal data.

## Security-sensitive areas

- GitHub Actions permissions and untrusted pull-request code;
- Google OAuth credentials, Colab runtime proxy tokens and GitHub tokens;
- notebook outputs and persisted credentials;
- dataset ingestion and untrusted archives;
- model/checkpoint deserialization;
- generated-code execution in future evaluation harnesses;
- artifact publishing and release workflows.

## Secret handling

Never commit API keys, OAuth refresh tokens, GitHub PATs, Colab runtime tokens, private SSH keys, service-account JSON, .env files or credential-bearing notebook outputs.

Use GitHub Actions secrets or an appropriate secret manager in CI, and Colab Secrets for notebook credentials. Apply least privilege, avoid printing tokens, keep credentials out of command-line arguments and revoke exposed credentials immediately.

## Data and checkpoint safety

Treat external checkpoints and dataset archives as untrusted. Prefer safetensors for inference weights. Avoid unrestricted pickle deserialization. Validate file size, format, manifest and checksums before loading. Do not execute source code during ingestion.

## Future execution sandbox

When execution-based coding evaluation is implemented, run generated code in a disposable, network-restricted sandbox with CPU, memory, disk and time limits. Never execute generated code in the maintainer's normal shell, credentials-bearing CI job or Colab controller process.
