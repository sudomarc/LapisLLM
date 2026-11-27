# Claude Code instructions

Read AGENTS.md before any work. It is the canonical repository policy. This file adds tool-specific guidance and must remain consistent with AGENTS.md.

- Inspect the actual tree and relevant code before editing. Do not assume historical files or checkpoints exist on the current branch.
- Read the linked issue and relevant docs; turn requirements into testable acceptance criteria.
- Do not introduce third-party pretrained weights. LapisLLM's base model is trained from random initialization.
- Training runs on the maintainer's own PC. Do not add hosted training dependencies or assume a particular GPU.
- Implement and run a hardware diagnostic before fixing model/batch profiles.
- Keep the CLI and Python modules reusable; avoid notebook-only logic.
- Assume local runs can be interrupted or the PC can reboot. Test checkpoint recovery.
- Never put credentials in source, shell history, logs or commits.
- Do not start a long training run unless the user explicitly requests it and data, hardware and storage budgets are known.
- Add tests and update documentation for changed behavior.
- Run targeted checks first, then relevant regression checks.
- Inspect git diff and git status before finishing.
- Report verified facts separately from assumptions; never claim a test or benchmark ran if it did not.

For large tasks, work in dependency order and keep each change reviewable. Do not bypass project safeguards to save time.
