#!/usr/bin/env python3
"""Goodput weekly todo tracker — mark tasks done and show progress."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_FILE = ROOT / ".goodput-user"
TODOS_DIR = ROOT / "todos"
CURRENT_FILE = TODOS_DIR / "current.json"

OWNERS = {
    "cletus": "Cletus",
}


def load_user() -> str | None:
    if not USER_FILE.exists():
        return None
    name = USER_FILE.read_text(encoding="utf-8").strip().lower()
    return name if name in OWNERS else None


def save_user(name: str) -> None:
    USER_FILE.write_text(name.lower(), encoding="utf-8")


def resolve_todos_path() -> Path:
    if CURRENT_FILE.exists():
        target = json.loads(CURRENT_FILE.read_text(encoding="utf-8"))
        if isinstance(target, str):
            path = TODOS_DIR / target
            if path.exists():
                return path
        if isinstance(target, dict) and "file" in target:
            path = TODOS_DIR / target["file"]
            if path.exists():
                return path

    weeks = sorted(TODOS_DIR.glob("week-*.json"))
    if weeks:
        return weeks[-1]

    raise FileNotFoundError("No todo file found in todos/")


def load_todos() -> dict:
    path = resolve_todos_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def save_todos(data: dict) -> None:
    path = Path(data.pop("_path"))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def progress(data: dict) -> dict[str, tuple[int, int]]:
    totals: dict[str, tuple[int, int]] = {}
    for key in OWNERS:
        mine = [t for t in data["tasks"] if t["owner"] == key]
        done = [t for t in mine if t["done"]]
        totals[key] = (len(done), len(mine))
    return totals


def format_progress(data: dict) -> str:
    stats = progress(data)
    parts = [f"{OWNERS[k]}: {done}/{total} done" for k, (done, total) in stats.items()]
    return "  |  ".join(parts)


def find_task(tasks: list[dict], query: str) -> dict | None:
    q = query.lower().strip()
    for task in tasks:
        if task["id"].lower() == q:
            return task
    matches = [t for t in tasks if q in t["id"].lower() or q in t["title"].lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("Multiple matches — be more specific:")
        for t in matches:
            print(f"  • {t['id']}: {t['title']}")
        return None
    return None


def cmd_who(name: str) -> int:
    key = name.lower()
    if key not in OWNERS:
        print(f"Unknown owner. Use: {' or '.join(OWNERS)}")
        return 1
    save_user(key)
    print(f"You are {OWNERS[key]}. Progress: {format_progress(load_todos())}")
    return 0


def cmd_status() -> int:
    data = load_todos()
    user = load_user()
    print(f"{data['label']} — {data['phase']}")
    if goal := data.get("goal"):
        print(f"Goal: {goal}")
    print(format_progress(data))
    print()
    for task in data["tasks"]:
        mark = "✓" if task["done"] else "○"
        owner = OWNERS.get(task["owner"], task["owner"])
        you = " ← you" if user and task["owner"] == user else ""
        print(f"  {mark} [{owner}] {task['id']}: {task['title']}{you}")
    return 0


def cmd_done(query: str) -> int:
    user = load_user()
    if not user:
        print("Set who you are first:")
        print("  gdone who cletus")
        return 1

    data = load_todos()
    task = find_task(data["tasks"], query)
    if not task:
        print(f"No task matching '{query}'. Run: gdone list")
        return 1

    if task["owner"] != user:
        print(f"This task belongs to {OWNERS.get(task['owner'], task['owner'])}, not {OWNERS[user]}.")
        return 1

    if task["done"]:
        print(f"Already done: {task['id']}")
        print(format_progress(data))
        return 0

    task["done"] = True
    task["completed_by"] = user
    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_todos(data)

    stats = progress(data)
    mine_done, mine_total = stats[user]
    print(f"✓ {task['title']}")
    print(f"{OWNERS[user]}: {mine_done}/{mine_total} done")
    print(format_progress(data))
    return 0


def cmd_help() -> int:
    print(
        """Goodput todos — weekly task tracker

Setup (once per machine):
  ./scripts/setup-shell.sh
  gdone who cletus

Mark a task complete:
  gdone git-init
  gdone docs-scaffold

Other commands:
  gdone status
  gdone list

Note: `done` is a shell reserved word — we use `gdone` instead.
Without setup, run: ./scripts/done <command>
"""
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return cmd_status()

    cmd = argv[1].lower()

    if cmd in {"help", "-h", "--help"}:
        return cmd_help()
    if cmd == "who" and len(argv) >= 3:
        return cmd_who(argv[2])
    if cmd in {"status", "list"}:
        return cmd_status()
    if cmd == "done" and len(argv) >= 3:
        return cmd_done(argv[2])

    return cmd_done(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
