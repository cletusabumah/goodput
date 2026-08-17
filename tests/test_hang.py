"""Hang fault injection + health-check detection tests (ticket 3.1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from goodput.config import Settings
from goodput.providers.faults import HANG_RANK_IDLE, MockFaultInjector, ProcessFaultInjector
from goodput.training.recovery import run_hang_and_recover, wait_for_hang_detection


def _sleep_worker(seconds: float = 30.0) -> None:
    time.sleep(seconds)


def test_mock_fault_injector_hang_records() -> None:
    inj = MockFaultInjector(inject_at=3, fault="hang")
    assert inj.maybe_inject(3, 1) == "hang"
    assert inj.injected == [(3, 1, "hang")]


def test_process_fault_injector_hang_sets_shared_rank() -> None:
    ctx = mp.get_context("spawn")
    hang_rank = ctx.Value("i", HANG_RANK_IDLE)
    inj = ProcessFaultInjector(inject_at=2, fault="hang", dry_run=False, hang_rank=hang_rank)
    assert inj.maybe_inject(2, 1) == "hang"
    assert hang_rank.value == 1


def test_wait_for_hang_detection_without_process_exit() -> None:
    """Done-when: stall detected while the worker is still alive (not via exit)."""
    ctx = mp.get_context("spawn")
    progress = ctx.Value("i", 4)
    proc = ctx.Process(target=_sleep_worker, args=(30.0,))
    proc.start()
    assert proc.pid is not None
    try:
        latency = wait_for_hang_detection(progress, 4, [proc], timeout_s=0.35)
        assert latency >= 0.3
        assert proc.is_alive()
    finally:
        proc.kill()
        proc.join(timeout=5)


def test_hang_recovery_two_workers(tmp_path: Path) -> None:
    """Integration: hang → detect → reap → resume from checkpoint step."""
    settings = Settings(
        num_workers=2,
        steps=8,
        ckpt_interval=2,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        ci_mode=False,
        health_check_timeout_s=0.5,
    )
    result = run_hang_and_recover(
        settings,
        ckpt_dir=tmp_path / "ckpts",
        hang_at_step=4,
        hang_rank=1,
        remaining_steps=2,
        health_check_timeout_s=0.5,
    )

    assert result.injected == [(4, 1, "hang")]
    assert result.fault_type == "hang"
    assert result.checkpoint_step == 4
    assert result.hang_detected
    assert result.detection_latency_s >= 0.4
    assert result.killed_pid is not None
    assert result.ok
    assert result.recovered.resumed_from_step == 4
    assert result.recovered.steps_completed == 2
    assert result.recovered.ok


def test_hang_requires_checkpoint_aligned_step(tmp_path: Path) -> None:
    settings = Settings(
        num_workers=2,
        steps=6,
        ckpt_interval=5,
        device="cpu",
        batch_size=4,
        input_size=8,
        hidden_size=16,
        health_check_timeout_s=0.5,
    )
    with pytest.raises(ValueError, match="multiple of ckpt_interval"):
        run_hang_and_recover(
            settings,
            ckpt_dir=tmp_path / "ckpts",
            hang_at_step=3,
        )


def test_cli_fault_hang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    ckpt_dir = tmp_path / "ckpts"
    cfg = tmp_path / "hang.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-hang",
                "mode: fault_hang",
                "fault_at: 4",
                "fault_rank: 1",
                "num_workers: 2",
                "steps: 8",
                "ckpt_interval: 2",
                "health_check_timeout_s: 0.5",
                f"ckpt_dir: {ckpt_dir}",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(cfg)])
    assert code == 0
    report_path = tmp_path / "artifacts" / "reports" / "cli-hang" / "report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["mode"] == "fault_hang"
    assert data["hang_detected"] == 1
    assert data["detection_latency_s"] > 0
