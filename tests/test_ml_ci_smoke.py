"""Ticket 1.9 — CI training smoke: ≤10 steps, finite loss, report fields."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from goodput.config import Settings
from goodput.data import SyntheticDataLoader
from goodput.experiments import load_experiment_yaml
from goodput.metrics import REQUIRED_REPORT_FIELDS
from goodput.models import ToyMLP
from goodput.training import train_from_settings, train_steps


def test_ci_smoke_yaml_is_short() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_experiment_yaml(root / "experiments" / "ci-smoke.yaml")
    assert spec.mode == "train"
    assert spec.settings.steps <= 10
    assert spec.settings.num_workers == 1
    assert spec.settings.device == "cpu"
    assert spec.settings.ci_mode is True


def test_train_smoke_loss_is_finite() -> None:
    settings = Settings(
        steps=8,
        num_workers=1,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        ckpt_interval=0,
        ci_mode=True,
    )
    result = train_from_settings(settings)
    assert result.ok
    assert result.steps_completed == 8
    assert math.isfinite(result.final_loss)
    assert all(math.isfinite(x) for x in result.losses)


def test_train_steps_fails_on_nan_loss() -> None:
    """Done-when: NaN loss aborts the run (CI must not stay green)."""
    device = torch.device("cpu")
    model = ToyMLP(8, 16)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loader = SyntheticDataLoader(
        num_batches=4,
        batch_size=4,
        input_size=8,
        seed=0,
        device=device,
    )

    def _nan_forward(_x: torch.Tensor) -> torch.Tensor:
        return torch.full((4, 1), float("nan"), device=device)

    model.forward = _nan_forward  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train_steps(
            model=model,
            batches=loader,
            optimizer=opt,
            steps=2,
            device=device,
        )


def test_cli_ci_smoke_writes_valid_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from goodput.cli import main

    root = Path(__file__).resolve().parents[1]
    src = (root / "experiments" / "ci-smoke.yaml").read_text(encoding="utf-8")
    cfg = tmp_path / "ci-smoke.yaml"
    # Redirect artifacts into tmp so the test is hermetic.
    cfg.write_text(
        src.replace("artifacts/checkpoints/ci-smoke", str(tmp_path / "ckpts")).replace(
            "artifacts_dir: artifacts",
            f"artifacts_dir: {tmp_path / 'artifacts'}",
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(cfg)])
    assert code == 0

    report_path = tmp_path / "artifacts" / "reports" / "ci-smoke" / "report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    for key in REQUIRED_REPORT_FIELDS:
        assert key in data
    assert 0.0 <= data["goodput"] <= 1.0
    assert math.isfinite(data["final_loss"])
    assert data["steps_completed"] <= 10
