"""gdone must rewrite week JSON without ASCII-escaping unicode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_done():
    path = ROOT / "scripts" / "done.py"
    spec = importlib.util.spec_from_file_location("goodput_done", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_todos_preserves_unicode_em_dash(tmp_path: Path) -> None:
    done = _load_done()
    path = tmp_path / "week-04.json"
    done.save_todos(
        {
            "_path": str(path),
            "phase": "Phase 2 — Fast checkpoint + goodput curve",
            "tasks": [],
        }
    )
    text = path.read_text(encoding="utf-8")
    assert "—" in text
    assert "\\u2014" not in text
    assert "\\u" not in text


def test_committed_week_files_use_literal_unicode() -> None:
    """gdone / hand-edited week JSON must not ASCII-escape punctuation (Week 4 \u2014 bug)."""
    week_files = sorted((ROOT / "todos").glob("week-*.json"))
    assert week_files, "expected todos/week-*.json"
    for path in week_files:
        text = path.read_text(encoding="utf-8")
        assert "\\u" not in text, f"{path.name} contains JSON unicode escapes"
