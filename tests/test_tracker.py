"""Ticket 4.3 — tracker provider logs a run (MLflow/WandB, JSON fallback)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from goodput.config import Settings
from goodput.providers import MLflowTracker, NullTracker, WandbTracker, build_providers
from goodput.providers.tracker import _metrics, _params

REPORT = {
    "run_name": "tracker-demo",
    "seed": 42,
    "ckpt_mode": "naive",
    "device": "cpu",
    "num_workers": 1,
    "schema_version": "1.1",
    "git_sha": "abc",
    "config_hash": "def",
    "goodput": 0.75,
    "ckpt_save_s": 0.01,
    "ckpt_restore_s": 0.02,
    "wasted_gpu_hours": 0.0,
    "wall_seconds": 1.0,
    "useful_seconds": 0.75,
    "steps_completed": 8,
    "final_loss": 1.25,
}


def test_null_tracker_logs_nothing() -> None:
    assert NullTracker().log_run(REPORT) is None


def test_params_and_metrics_extract_known_keys() -> None:
    assert _params(REPORT)["run_name"] == "tracker-demo"
    assert _metrics(REPORT)["goodput"] == 0.75
    assert "not_a_field" not in _metrics({"not_a_field": 1, "goodput": 0.5})


def test_mlflow_tracker_writes_json_without_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_mlflow() -> Any:
        raise ImportError("test: no mlflow")

    monkeypatch.setattr("goodput.providers.tracker._import_mlflow", _no_mlflow)
    tracker = MLflowTracker(tracking_uri="file:./unused", artifacts_dir=tmp_path)
    run_id = tracker.log_run(REPORT)
    assert run_id is not None
    files = list((tmp_path / "mlflow").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["backend"] == "mlflow_json"
    assert payload["run_id"] == run_id
    assert payload["metrics"]["goodput"] == 0.75
    assert payload["params"]["seed"] == "42"


def test_wandb_tracker_writes_json_without_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_wandb() -> Any:
        raise ImportError("test: no wandb")

    monkeypatch.setattr("goodput.providers.tracker._import_wandb", _no_wandb)
    tracker = WandbTracker(project="goodput", artifacts_dir=tmp_path)
    run_id = tracker.log_run(REPORT)
    assert run_id is not None
    files = list((tmp_path / "wandb").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["backend"] == "wandb_json"
    assert payload["metrics"]["goodput"] == 0.75


def test_build_providers_default_tracker_is_none(tmp_path: Path) -> None:
    settings = Settings(ci_mode=False, artifacts_dir=tmp_path, tracker="none")
    providers = build_providers(settings, artifacts_dir=tmp_path)
    assert providers.tracker.name == "none"
    assert providers.tracker.log_run(REPORT) is None


def test_build_providers_mlflow_logs_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "goodput.providers.tracker._import_mlflow",
        lambda: (_ for _ in ()).throw(ImportError("test: no mlflow")),
    )
    settings = Settings(
        ci_mode=False,
        artifacts_dir=tmp_path,
        tracker="mlflow",
        run_name="unit",
    )
    providers = build_providers(settings, artifacts_dir=tmp_path)
    assert providers.tracker.name == "mlflow"
    run_id = providers.tracker.log_run(REPORT)
    assert run_id is not None
    json_runs = list((tmp_path / "mlflow").glob("*.json"))
    mlruns = tmp_path / "mlruns"
    assert json_runs or mlruns.exists()


def test_cli_tracker_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "t.yaml"
    out = tmp_path / "artifacts"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-tracker",
                "mode: train",
                "num_workers: 1",
                "steps: 4",
                "ckpt_interval: 0",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 1",
                "tracker: mlflow",
                f"ckpt_dir: {tmp_path / 'ckpts'}",
                f"artifacts_dir: {out}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "goodput.providers.tracker._import_mlflow",
        lambda: (_ for _ in ()).throw(ImportError("test: no mlflow")),
    )
    code = main(["--config", str(cfg)])
    assert code == 0
    logged = list((out / "mlflow").glob("*.json"))
    assert logged, "expected JSON fallback run under artifacts/mlflow/"
