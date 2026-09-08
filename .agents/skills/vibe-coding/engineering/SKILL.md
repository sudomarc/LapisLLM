---
name: vibe-coding-engineering
description: Safely implement, debug, refactor, test, review, and release code in Lapis using evidence-first investigation and minimal verified changes.
version: 1.0.0
status: stable
category: Vibe Coding
---
# Vibe Coding Engineering
## Purpose
Provide one mandatory engineering loop for repository changes.
## When to use
Use for any coding, debugging, refactoring, dependency, CI, PR, or release task.
## Workflow
Discover repository context; reproduce failures; define contract; implement the smallest correct patch; add or update regression tests; inspect dependency and CI impact; review the complete diff; verify locally and report exact evidence.
## Verification
Run narrow tests first, then affected integration/regression checks, lint/type/static checks that exist in the repository, and inspect `git status`/`git diff`.
## Safety
Never weaken tests, hide failures, bypass hooks, force-push, expose secrets, or make unrelated cleanup changes merely because an agent can.
