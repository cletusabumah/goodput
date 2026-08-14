"""Ticket 2.4 — Compose file + kill script without starting Docker."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from goodput.cli import main
from goodput.experiments import load_experiment_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_yaml_has_two_default_workers() -> None:
    data = yaml.safe_load((ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8"))
    services = data["services"]
    assert "worker-0" in services
    assert "worker-1" in services
    assert services["worker-0"].get("profiles") in (None, [])
    assert services["worker-1"].get("profiles") in (None, [])
    assert services["worker-2"]["profiles"] == ["four"]
    assert services["worker-3"]["profiles"] == ["four"]
    for name in ("worker-0", "worker-1", "worker-2", "worker-3"):
        env = services[name]["environment"]
        assert env["GOODPUT_NUM_WORKERS"] == "1"
        vols = services[name]["volumes"]
        assert "../artifacts:/app/artifacts" in vols
    cmd = services["worker-0"]["command"]
    assert any("sleep infinity" in str(part) for part in cmd)


def test_compose_experiment_yaml_is_single_process() -> None:
    spec = load_experiment_yaml(ROOT / "experiments" / "compose.yaml")
    assert spec.mode == "train"
    assert spec.settings.num_workers == 1
    assert spec.settings.ckpt_interval > 0
    assert spec.settings.device == "cpu"


def test_kill_worker_script_dry_run() -> None:
    script = ROOT / "docker" / "kill-worker.sh"
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run:" in result.stdout
    assert "kill" in result.stdout
    assert "SIGKILL" in result.stdout
    assert "worker-1" in result.stdout
    assert "docker" in result.stdout


def test_rank_nonzero_skips_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "ckpts"
    artifacts = tmp_path / "artifacts"
    cfg = tmp_path / "rank1.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: rank1-node",
                "mode: train",
                "num_workers: 1",
                "rank: 1",
                "steps: 4",
                "ckpt_interval: 2",
                f"ckpt_dir: {ckpt}",
                f"artifacts_dir: {artifacts}",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 3",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(["--config", str(cfg)])
    assert code == 0
    assert not ckpt.exists() or not any(ckpt.glob("*.pt"))


def test_rank_zero_writes_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "ckpts"
    artifacts = tmp_path / "artifacts"
    cfg = tmp_path / "rank0.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: rank0-node",
                "mode: train",
                "num_workers: 1",
                "rank: 0",
                "steps: 4",
                "ckpt_interval: 2",
                f"ckpt_dir: {ckpt}",
                f"artifacts_dir: {artifacts}",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 3",
                "device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(["--config", str(cfg)])
    assert code == 0
    assert (ckpt / "latest.pt").is_file()
