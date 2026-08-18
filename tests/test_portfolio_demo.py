"""Ticket 3.5 — portfolio demo script regenerates the three-story artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_demo_script_exists_and_is_executable() -> None:
    script = ROOT / "scripts" / "portfolio-demo.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111


def test_portfolio_demo_dry_run_lists_three_stories() -> None:
    script = ROOT / "scripts" / "portfolio-demo.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "dry-run:" in out
    assert "ci-smoke.yaml" in out
    assert "--sweep experiments/sweep.yaml" in out
    assert "--latency experiments/latency.yaml" in out
    assert "--dollar experiments/dollar.yaml" in out
    assert "comparison.json" in out
    assert "latency-table" in out
    assert "dollar-impact" in out


def test_architecture_documents_portfolio_demo_and_three_stories() -> None:
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "portfolio-demo.sh" in text
    assert "Three-story portfolio" in text
    assert "kill / hang / bitflip" in text
    assert "git_sha" in text


def test_readme_documents_portfolio_demo() -> None:
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "portfolio-demo.sh" in readme
    assert "Phase 3" in readme
    assert "Demo walkthrough" in text or "Demo walkthrough" in (
        ROOT / "docs" / "what_i_learned.md"
    ).read_text(encoding="utf-8")


def test_learned_log_has_demo_walkthrough_section() -> None:
    text = (ROOT / "docs" / "what_i_learned.md").read_text(encoding="utf-8")
    assert "## Demo walkthrough" in text
    assert "./scripts/portfolio-demo.sh" in text
    assert "Three charts" in text
    assert "Honest caveats" in text
