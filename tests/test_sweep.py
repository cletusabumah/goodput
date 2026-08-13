"""Ticket 2.2 — failure-rate × ckpt-mode sweep writes comparison JSON/CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.evaluation.sweep import (
    COMPARISON_FIELDS,
    kill_at_for_rate,
    load_sweep_yaml,
    run_sweep,
    run_sweep_cell,
)
from goodput.metrics import REQUIRED_REPORT_FIELDS


def test_kill_at_none_when_rate_zero() -> None:
    assert kill_at_for_rate(steps=8, ckpt_interval=2, failure_rate=0.0) is None


def test_kill_at_snaps_to_interval_and_leaves_resume_room() -> None:
    # rate=0.5 → mean gap 2 → kill at 2 (durable ckpt, remaining=6)
    assert kill_at_for_rate(steps=8, ckpt_interval=2, failure_rate=0.5) == 2
    # rate=0.25 → mean gap 4
    assert kill_at_for_rate(steps=8, ckpt_interval=2, failure_rate=0.25) == 4
    # large gap still stays strictly before steps
    assert kill_at_for_rate(steps=8, ckpt_interval=2, failure_rate=0.1) == 6


def test_kill_at_rejects_too_short_run() -> None:
    with pytest.raises(ValueError, match="exceed ckpt_interval"):
        kill_at_for_rate(steps=2, ckpt_interval=2, failure_rate=0.5)


def test_load_committed_sweep_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_sweep_yaml(root / "experiments" / "sweep.yaml")
    assert spec.name == "phase2-sweep"
    assert spec.ckpt_modes == ("naive", "incremental")
    assert spec.failure_rates == (0.0, 0.25, 0.5)
    assert spec.settings.steps == 8
    assert spec.settings.ckpt_interval == 2


def test_load_sweep_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmode: sweep\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown sweep keys"):
        load_sweep_yaml(path)


def test_run_sweep_writes_json_and_csv(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny-sweep.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: tiny-sweep",
                "mode: sweep",
                "ckpt_modes:",
                "  - naive",
                "  - incremental",
                "failure_rates:",
                "  - 0.0",
                "  - 0.5",
                "num_workers: 1",
                "steps: 6",
                "ckpt_interval: 2",
                "ckpt_full_every: 2",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 1",
                f"ckpt_dir: {tmp_path / 'ckpts'}",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_sweep_yaml(cfg, base=Settings())
    result = run_sweep(spec)
    assert result.json_path is not None and result.csv_path is not None
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    assert len(result.rows) == 4  # 2 modes × 2 rates

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert len(data) == 4
    for row in data:
        for key in COMPARISON_FIELDS:
            assert key in row
        assert 0.0 <= row["goodput"] <= 1.0

    modes = {row["ckpt_mode"] for row in data}
    rates = {row["failure_rate"] for row in data}
    assert modes == {"naive", "incremental"}
    assert rates == {0.0, 0.5}

    crash_rows = [r for r in data if r["failure_rate"] == 0.5]
    assert all(r["kill_at"] == 2 for r in crash_rows)
    clean_rows = [r for r in data if r["failure_rate"] == 0.0]
    assert all(r["kill_at"] is None for r in clean_rows)

    with result.csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [h for h in csv_rows[0]] == list(COMPARISON_FIELDS)
    assert len(csv_rows) == 4


def test_sweep_cell_report_has_required_fields(tmp_path: Path) -> None:
    settings = Settings(
        run_name="cell",
        steps=6,
        ckpt_interval=2,
        num_workers=1,
        batch_size=4,
        input_size=8,
        hidden_size=8,
        seed=2,
        device="cpu",
        ci_mode=True,
    )
    report = run_sweep_cell(
        settings,
        ckpt_mode="naive",
        failure_rate=0.5,
        ckpt_dir=tmp_path / "ckpts",
    )
    for key in REQUIRED_REPORT_FIELDS:
        assert key in report
    assert report["kill_at"] == 2
    assert report["mode"] == "sweep_crash"


def test_zero_failure_cell_does_not_resume_on_rerun(tmp_path: Path) -> None:
    """A second sweep must not pick up leftover latest.pt as a resume."""
    settings = Settings(
        run_name="rerun",
        steps=6,
        ckpt_interval=2,
        num_workers=1,
        batch_size=4,
        input_size=8,
        hidden_size=8,
        seed=4,
        device="cpu",
        ci_mode=True,
    )
    ckpt_dir = tmp_path / "ckpts"
    first = run_sweep_cell(
        settings,
        ckpt_mode="naive",
        failure_rate=0.0,
        ckpt_dir=ckpt_dir,
    )
    assert first["mode"] == "sweep_train"
    assert first["ckpt_restore_s"] == 0.0
    assert first["steps_completed"] == 6
    assert (ckpt_dir / "latest.pt").is_file()

    second = run_sweep_cell(
        settings,
        ckpt_mode="naive",
        failure_rate=0.0,
        ckpt_dir=ckpt_dir,
    )
    assert second["mode"] == "sweep_train"
    assert second["ckpt_restore_s"] == 0.0
    assert second["steps_completed"] == 6
    assert second.get("resumed_from_step") in {None, 0}


def test_cli_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "cli-sweep.yaml"
    out = tmp_path / "artifacts"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-sweep",
                "mode: sweep",
                "ckpt_modes:",
                "  - naive",
                "failure_rates:",
                "  - 0.0",
                "num_workers: 1",
                "steps: 4",
                "ckpt_interval: 2",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 0",
                f"ckpt_dir: {tmp_path / 'ckpts'}",
                f"artifacts_dir: {out}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--sweep", str(cfg)])
    assert code == 0
    assert (out / "sweeps" / "cli-sweep" / "comparison.json").is_file()
    assert (out / "sweeps" / "cli-sweep" / "comparison.csv").is_file()
