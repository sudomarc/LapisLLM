# Lapis Agent System

`.agents/` is the operational control plane for coding agents working in Lapis.

## Lifecycle

```text
AGENTS.md
  -> bootstrap.md
  -> manifest.yaml
  -> skill discovery
  -> SKILL.md
  -> task execution
  -> verification
  -> evidence report
```

## Canonical source

Skills live under `.agents/skills/`. The manifest is authoritative for discoverability. Public copies or website pages are documentation and must not become independent instruction sources.

## Commands

```bash
python .agents/runtime/select.py --task "fix failing API streaming tests"
python .agents/runtime/validate.py
```

The selector always includes the core agent operating system skill and adds trigger-matching skills. The validator rejects missing, malformed, duplicate, or orphan skills.

## Design rules

- Instructions are hierarchical and task-aware.
- Skills are progressively disclosed instead of all being loaded at once.
- Deterministic checks enforce properties that should not depend on model behavior.
- Skills must define Purpose, When to use, Workflow, Verification, and Safety.
- Evidence is required for claims about tests, benchmarks, provider capabilities, and successful behavior.
