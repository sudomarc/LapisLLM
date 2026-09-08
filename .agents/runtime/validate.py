#!/usr/bin/env python3
"""Validate Lapis Agent Skills without third-party dependencies."""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".agents" / "manifest.yaml"
SKILLS = ROOT / ".agents" / "skills"
REQUIRED = {"name", "description", "version", "status", "category"}


def metadata(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("missing YAML frontmatter")
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


def manifest_paths() -> list[pathlib.Path]:
    paths = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("path: "):
            paths.append(ROOT / line.split(":", 1)[1].strip())
    return paths


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.exists():
        errors.append("missing .agents/manifest.yaml")
    if not (ROOT / ".agents/bootstrap.md").exists():
        errors.append("missing .agents/bootstrap.md")
    paths = manifest_paths() if MANIFEST.exists() else []
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path in seen:
            errors.append(f"duplicate skill path: {path}")
            continue
        seen.add(path)
        if not path.exists():
            errors.append(f"missing skill: {path}")
            continue
        try:
            meta = metadata(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        missing = REQUIRED - set(meta)
        if missing:
            errors.append(f"{path}: missing metadata: {', '.join(sorted(missing))}")
        if not re.fullmatch(r"\d+\.\d+\.\d+", meta.get("version", "")):
            errors.append(f"{path}: invalid semantic version")
        for heading in ("Purpose", "When to use", "Workflow", "Verification", "Safety"):
            if f"## {heading}" not in path.read_text(encoding="utf-8"):
                errors.append(f"{path}: missing ## {heading}")
    unregistered = []
    for path in SKILLS.glob("*/**/SKILL.md"):
        if path not in seen:
            unregistered.append(path)
    if unregistered:
        errors.extend(f"orphan skill: {p}" for p in unregistered)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Lapis agent validation passed: {len(paths)} registered skills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
