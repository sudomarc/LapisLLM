#!/usr/bin/env python3
"""Select applicable Lapis skills using the repository manifest."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".agents" / "manifest.yaml"


def load_manifest() -> list[dict[str, object]]:
    text = MANIFEST.read_text(encoding="utf-8")
    skills: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_triggers = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("  - name: "):
            if current:
                skills.append(current)
            current = {"name": line.split(":", 1)[1].strip()}
            in_triggers = False
        elif current is not None and line.startswith("    path: "):
            current["path"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("    triggers: "):
            current["triggers"] = re.findall(r"[A-Za-z0-9_-]+", line.split(":", 1)[1])
            in_triggers = True
        elif current is not None and in_triggers and line.startswith("      "):
            continue
    if current:
        skills.append(current)
    return skills


def select(task: str) -> list[dict[str, object]]:
    needle = task.lower()
    selected: list[dict[str, object]] = []
    for skill in load_manifest():
        name = str(skill["name"])
        triggers = [str(x).lower() for x in skill.get("triggers", [])]
        if name == "agent-operating-system" or any(t in needle for t in triggers):
            selected.append(skill)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    for skill in select(args.task):
        print(f"{skill['name']}\t{skill['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
