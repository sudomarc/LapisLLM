# Claude Code instructions

Read AGENTS.md before any work. It is the canonical repository policy. This file adds only Claude Code operating guidance and must remain consistent with AGENTS.md.

- Inspect the actual tree and relevant code before editing. Do not assume historical files or checkpoints still exist on the current branch.
- Read the linked issue and relevant docs/ pages; turn requirements into testable acceptance criteria.
- Do not introduce third-party pretrained weights. LapisLLM's base model is trained from random initialization.
- Keep Colab notebooks thin. Reusable implementation belongs in importable Python modules.
- Assume every Colab runtime can terminate. Test checkpoint recovery before relying on long runs.
- Never put credentials in notebook cells, shell history, source files, logs or commits.
- Do not run expensive training unless the task explicitly requests it and the compute budget and artifact destination are known.
- Add tests and update documentation for changed behavior.
- Run targeted checks first, then relevant regression checks.
- Inspect git diff and git status before finishing.
- Report verified facts separately from assumptions, and never claim a command or benchmark ran if it did not.

For large tasks, work in dependency order and keep each change reviewable. Do not bypass project safeguards to save time.
