"""Ticket 2.5 — checkpoint/restore latency vs worker count writes a table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.evaluation.latency import (
    LATENCY_FIELDS,
    load_latency_yaml,
    render_latency_table,
    run_latency,
    run_latency_cell,
)
from goodput.metrics import REQUIRED_REPORT_FIELDS


def test_load_committed_latency_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_latency_yaml(root / "experiments" / "latency.yaml")
    assert spec.name == "latency-table"
    assert spec.worker_counts == (1, 2, 4)
    assert spec.settings.steps == 6
    assert spec.settings.ckpt_interval == 2
    assert spec.settings.ckpt_mode == "naive"


def test_load_latency_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\nmode: latency\nworker_counts: [1]\nnot_a_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown latency keys"):
        load_latency_yaml(path)


def test_load_latency_rejects_empty_counts(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("name: x\nmode: latency\nworker_counts: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker_counts"):
        load_latency_yaml(path)


def test_load_latency_rejects_zero_workers(tmp_path: Path) -> None:
    path = tmp_path / "zero.yaml"
    path.write_text("name: x\nmode: latency\nworker_counts: [0]\nckpt_interval: 2\nsteps: 4\n")
    with pytest.raises(ValueError, match="worker_counts must be >= 1"):
        load_latency_yaml(path)


def test_run_latency_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny-latency.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: tiny-latency",
                "mode: latency",
                "worker_counts:",
                "  - 1",
                "  - 2",
                "steps: 4",
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
    spec = load_latency_yaml(cfg, base=Settings())
    result = run_latency(spec)
    assert result.json_path is not None and result.csv_path is not None
    assert result.table_path is not None
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    assert result.table_path.is_file()
    assert len(result.rows) == 2

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    counts = [row["num_workers"] for row in data]
    assert counts == [1, 2]
    for row in data:
        for key in LATENCY_FIELDS:
            assert key in row
        assert 0.0 <= row["goodput"] <= 1.0
        assert row["ckpt_save_s"] > 0
        assert row["ckpt_restore_s"] > 0
        assert row["ckpt_save_count"] >= 1

    markdown = result.table_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Checkpoint/restore latency vs worker count")
    assert "| num_workers |" in markdown
    assert "| 1 |" in markdown
    assert "| 2 |" in markdown

    with result.csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [h for h in csv_rows[0]] == list(LATENCY_FIELDS)
    assert len(csv_rows) == 2


def test_latency_cell_report_has_required_fields(tmp_path: Path) -> None:
    settings = Settings(
        run_name="cell",
        steps=4,
        ckpt_interval=2,
        num_workers=1,
        batch_size=4,
        input_size=8,
        hidden_size=8,
        seed=2,
        device="cpu",
        ci_mode=True,
    )
    report = run_latency_cell(
        settings,
        num_workers=1,
        ckpt_dir=tmp_path / "ckpts",
    )
    for key in REQUIRED_REPORT_FIELDS:
        assert key in report
    assert report["mode"] == "latency_cell"
    assert report["num_workers"] == 1
    assert report["ckpt_restore_s"] > 0


def test_latency_cell_wipes_ckpt_dir_on_rerun(tmp_path: Path) -> None:
    settings = Settings(
        run_name="rerun",
        steps=4,
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
    first = run_latency_cell(settings, num_workers=1, ckpt_dir=ckpt_dir)
    assert first["steps_completed"] == 4
    assert first["last_checkpoint_step"] == 4
    assert first.get("train_resumed_from_step") in {None, 0}
    assert (ckpt_dir / "latest.pt").is_file()

    second = run_latency_cell(settings, num_workers=1, ckpt_dir=ckpt_dir)
    assert second["mode"] == "latency_cell"
    assert second["steps_completed"] == 4
    assert second["last_checkpoint_step"] == 4
    assert second.get("train_resumed_from_step") in {None, 0}


def test_render_latency_table_includes_header_and_rows() -> None:
    text = render_latency_table(
        [
            {
                "num_workers": 1,
                "ckpt_mode": "naive",
                "ckpt_save_s": 0.01,
                "ckpt_restore_s": 0.02,
                "ckpt_save_count": 2,
                "goodput": 1.0,
                "wasted_gpu_hours": 0.0,
                "wall_seconds": 0.1,
                "useful_seconds": 0.1,
                "steps_completed": 4,
                "run_name": "demo-n1",
            }
        ]
    )
    assert "num_workers" in text
    assert "0.010000" in text
    assert "1.0000" in text


def test_cli_latency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "cli-latency.yaml"
    out = tmp_path / "artifacts"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-latency",
                "mode: latency",
                "worker_counts:",
                "  - 1",
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
    code = main(["--latency", str(cfg)])
    assert code == 0
    table_dir = out / "sweeps" / "cli-latency"
    assert (table_dir / "latency.json").is_file()
    assert (table_dir / "latency.csv").is_file()
    assert (table_dir / "table.md").is_file()
