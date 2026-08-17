"""Ticket 1.8 — experiment YAML loader + --config runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.experiments import load_experiment_yaml
from goodput.metrics import REQUIRED_REPORT_FIELDS


def test_load_baseline_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment_yaml(root / "experiments" / "baseline.yaml")
    assert spec.name == "baseline-naive"
    assert spec.mode == "train"
    assert spec.settings.run_name == "baseline-naive"
    assert spec.settings.num_workers == 2
    assert spec.settings.steps == 20
    assert spec.settings.ckpt_interval == 5
    assert spec.settings.fault_mode == "none"
    assert spec.fault_at is None


def test_load_yaml_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nnot_a_real_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experiment keys"):
        load_experiment_yaml(path)


def test_load_fault_kill_requires_fault_at(tmp_path: Path) -> None:
    path = tmp_path / "kill.yaml"
    path.write_text("name: k\nmode: fault_kill\nsteps: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fault_at"):
        load_experiment_yaml(path)


def test_load_fault_kill_yaml(tmp_path: Path) -> None:
    path = tmp_path / "kill.yaml"
    path.write_text(
        "\n".join(
            [
                "name: kill-demo",
                "mode: fault_kill",
                "fault_at: 4",
                "fault_rank: 1",
                "num_workers: 2",
                "steps: 8",
                "ckpt_interval: 4",
                "ckpt_dir: artifacts/checkpoints/kill-demo",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_experiment_yaml(path, base=Settings())
    assert spec.mode == "fault_kill"
    assert spec.fault_at == 4
    assert spec.fault_rank == 1
    assert spec.settings.fault_at == "4"
    assert spec.settings.run_name == "kill-demo"


def test_load_fault_hang_yaml(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment_yaml(root / "experiments" / "fault-hang.yaml")
    assert spec.mode == "fault_hang"
    assert spec.fault_at == 4
    assert spec.settings.health_check_timeout_s == 1.0
    assert spec.settings.fault_mode == "hang"


def test_cli_config_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "short.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-config-smoke",
                "mode: train",
                "num_workers: 1",
                "steps: 4",
                "ckpt_interval: 2",
                f"ckpt_dir: {tmp_path / 'ckpts'}",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Avoid picking up a developer .env that could override paths.
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(cfg)])
    assert code == 0
    report_path = tmp_path / "artifacts" / "reports" / "cli-config-smoke" / "report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    for key in REQUIRED_REPORT_FIELDS:
        assert key in data
    assert 0.0 <= data["goodput"] <= 1.0
    assert data["run_name"] == "cli-config-smoke"


def test_cli_config_multiprocess_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline-shaped YAML (N=2 + ckpt_interval) must checkpoint and time saves."""
    from goodput.cli import main

    ckpt_dir = tmp_path / "ckpts"
    cfg = tmp_path / "mp.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: mp-ckpt-smoke",
                "mode: train",
                "num_workers: 2",
                "steps: 4",
                "ckpt_interval: 2",
                f"ckpt_dir: {ckpt_dir}",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(cfg)])
    assert code == 0
    assert (ckpt_dir / "step_000002.pt").is_file()
    assert (ckpt_dir / "latest.pt").is_file()
    report_path = tmp_path / "artifacts" / "reports" / "mp-ckpt-smoke" / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["ckpt_save_count"] >= 1
    assert data["ckpt_save_s"] > 0
    assert data["last_checkpoint_step"] == 4
