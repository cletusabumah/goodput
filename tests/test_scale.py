"""Ticket 4.1 — goodput vs worker count (MTBF-at-scale) writes JSON/CSV/markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.evaluation.scale import (
    SCALE_ROW_FIELDS,
    cluster_failure_rate,
    load_scale_yaml,
    render_scale_table,
    run_scale,
)
from goodput.evaluation.sweep import kill_at_for_rate


def test_cluster_failure_rate_scales_with_n() -> None:
    assert cluster_failure_rate(1, 0.125) == pytest.approx(0.125)
    assert cluster_failure_rate(4, 0.125) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="num_workers"):
        cluster_failure_rate(0, 0.1)
    with pytest.raises(ValueError, match="per_gpu_failure_rate"):
        cluster_failure_rate(1, -0.1)


def test_larger_n_crashes_earlier_or_equal() -> None:
    """Independent failures → higher cluster rate → kill_at does not move later."""
    steps, interval = 8, 2
    kill_small = kill_at_for_rate(steps=steps, ckpt_interval=interval, failure_rate=0.125)
    kill_large = kill_at_for_rate(steps=steps, ckpt_interval=interval, failure_rate=0.5)
    assert kill_small is not None and kill_large is not None
    assert kill_large <= kill_small


def test_load_committed_scale_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_scale_yaml(root / "experiments" / "scale.yaml")
    assert spec.name == "goodput-vs-workers"
    assert spec.worker_counts == (1, 2, 4)
    assert spec.ckpt_modes == ("naive", "incremental")
    assert spec.per_gpu_failure_rate == pytest.approx(0.125)
    assert spec.settings.steps == 8


def test_load_scale_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmode: scale\nworker_counts: [1]\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown scale keys"):
        load_scale_yaml(path)


def test_load_scale_rejects_empty_counts(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("name: x\nmode: scale\nworker_counts: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker_counts"):
        load_scale_yaml(path)


def test_run_scale_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny-scale.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: tiny-scale",
                "mode: scale",
                "worker_counts:",
                "  - 1",
                "  - 2",
                "ckpt_modes:",
                "  - naive",
                "per_gpu_failure_rate: 0.25",
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
    spec = load_scale_yaml(cfg, base=Settings())
    result = run_scale(spec)
    assert result.json_path is not None and result.csv_path is not None
    assert result.table_path is not None
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    assert result.table_path.is_file()
    assert len(result.rows) == 2

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "scale"
    writeup = payload["writeup"].lower()
    assert "illustration" in writeup or "not a cluster trace" in writeup
    assert len(payload["rows"]) == 2
    ns = [row["num_workers"] for row in payload["rows"]]
    assert ns == [1, 2]
    rates = [row["cluster_failure_rate"] for row in payload["rows"]]
    assert rates[1] == pytest.approx(2 * rates[0])
    for row in payload["rows"]:
        for key in SCALE_ROW_FIELDS:
            assert key in row
        assert 0.0 <= row["goodput"] <= 1.0

    markdown = result.table_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Goodput vs worker count")
    assert "MTBF" in markdown or "interruption" in markdown.lower()
    assert "| num_workers |" in markdown

    with result.csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert list(csv_rows[0].keys()) == list(SCALE_ROW_FIELDS)


def test_render_scale_table_includes_disclaimer() -> None:
    from goodput.evaluation.scale import ScaleSpec

    spec = ScaleSpec(
        path=Path("x.yaml"),
        name="t",
        settings=Settings(),
        worker_counts=(1, 2),
        ckpt_modes=("naive",),
        per_gpu_failure_rate=0.125,
        output_dir=Path("out"),
    )
    text = render_scale_table(
        [
            {
                "num_workers": 1,
                "ckpt_mode": "naive",
                "per_gpu_failure_rate": 0.125,
                "cluster_failure_rate": 0.125,
                "kill_at": 6,
                "goodput": 0.8,
                "ckpt_save_s": 0.01,
                "ckpt_restore_s": 0.02,
                "wasted_gpu_hours": 0.0,
                "wall_seconds": 0.1,
                "useful_seconds": 0.08,
                "steps_completed": 6,
                "run_name": "t-naive-n1",
            }
        ],
        spec,
    )
    assert "Illustration" in text or "illustration" in text.lower()
    assert "0.8000" in text


def test_cli_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "cli-scale.yaml"
    out = tmp_path / "artifacts"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-scale",
                "mode: scale",
                "worker_counts:",
                "  - 1",
                "ckpt_modes:",
                "  - naive",
                "per_gpu_failure_rate: 0.25",
                "steps: 6",
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
    code = main(["--scale", str(cfg)])
    assert code == 0
    table_dir = out / "sweeps" / "cli-scale"
    assert (table_dir / "scale.json").is_file()
    assert (table_dir / "scale.csv").is_file()
    assert (table_dir / "table.md").is_file()
    data = json.loads((table_dir / "scale.json").read_text(encoding="utf-8"))
    assert data["mode"] == "scale"
    assert len(data["rows"]) == 1
    assert data["rows"][0]["num_workers"] == 1
