# Security Policy

## Supported Versions

LapisLLM is an experimental research and learning project. Security fixes are applied to the current `main` branch. Older commits and branches are not guaranteed to receive fixes.

## Reporting a Vulnerability

Do not report security vulnerabilities in public issues or pull requests.

For a private report, use the repository owner's GitHub security contact or the private vulnerability-reporting mechanism enabled for this repository, when available. Include:

- affected commit, file, and line or function;
- reproduction steps or a minimal proof of concept;
- expected and observed behavior;
- security impact;
- any required environment or dependency versions.

Do not include real API keys, passwords, private keys, or other credentials in the report.

## Checkpoint Safety

Treat model checkpoints from untrusted sources as potentially malicious. Inference paths use PyTorch's restricted `weights_only=True` loading mode. Training-resume compatibility paths may have broader serialization requirements and must only consume checkpoints from trusted sources.

## Secret Handling

Credentials must never be committed to Git. Use environment variables or local secret stores, and keep `.env` files and private key material untracked.
