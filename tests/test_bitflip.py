"""Bit-flip gradient corruption + optional detector tests (ticket 3.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from goodput.config import Settings
from goodput.faults.bitflip import detect_gradient_outlier, flip_float32_bit
from goodput.providers.faults import MockFaultInjector, ProcessFaultInjector
from goodput.training.bitflip import run_bitflip_train


def test_flip_float32_bit_changes_value() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    before = t.clone()
    idx = flip_float32_bit(t)
    assert idx == 2  # default index -1 → last element
    assert not torch.equal(t, before)


def test_detect_gradient_outlier_finds_spike() -> None:
    bucket = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert detect_gradient_outlier(bucket, ratio_threshold=4.0)
    assert not detect_gradient_outlier(bucket, ratio_threshold=200.0)


def test_mock_fault_injector_bitflip_records() -> None:
    inj = MockFaultInjector(inject_at=3, fault="bitflip")
    assert inj.maybe_inject(3, 1) == "bitflip"
    assert inj.injected == [(3, 1, "bitflip")]


def test_process_fault_injector_bitflip_sets_shared_memory() -> None:
    ctx = mp.get_context("spawn")
    rank_val = ctx.Value("i", -1)
    step_val = ctx.Value("i", -1)
    inj = ProcessFaultInjector(
        inject_at=2,
        fault="bitflip",
        dry_run=False,
        bitflip_rank=rank_val,
        bitflip_at_step=step_val,
    )
    assert inj.maybe_inject(2, 1) == "bitflip"
    assert rank_val.value == 1
    assert step_val.value == 2


def test_bitflip_changes_loss_trajectory(tmp_path: Path) -> None:
    """Done-when: corrupted run diverges from clean baseline after flip step."""
    settings = Settings(
        num_workers=2,
        steps=8,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        ci_mode=True,
        bitflip_detect=True,
        bitflip_grad_ratio_threshold=4.0,
    )
    clean = run_bitflip_train(settings, flip_at_step=8, flip_rank=1)
    rank0_clean = next(w for w in clean.workers if w.rank == 0)
    assert len(rank0_clean.losses) == 8

    corrupt = run_bitflip_train(settings, flip_at_step=4, flip_rank=1)
    assert corrupt.ok
    assert any(w.bitflip_applied for w in corrupt.workers)
    losses = corrupt.rank0_losses
    assert len(losses) == 8
    # Same through flip step (exclusive); diverge afterward.
    assert losses[:4] == rank0_clean.losses[:4]
    assert losses[4:] != rank0_clean.losses[4:]


def test_bitflip_detector_flags_corruption(tmp_path: Path) -> None:
    settings = Settings(
        num_workers=2,
        steps=6,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
        ci_mode=False,
        bitflip_detect=True,
        bitflip_grad_ratio_threshold=4.0,
    )
    result = run_bitflip_train(settings, flip_at_step=3, flip_rank=1)
    assert result.corruption_detected
    assert result.corruption_detected_at == 3


def test_cli_fault_bitflip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "bitflip.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-bitflip",
                "mode: fault_bitflip",
                "fault_at: 4",
                "fault_rank: 1",
                "num_workers: 2",
                "steps: 8",
                "bitflip_detect: true",
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
    report_path = tmp_path / "artifacts" / "reports" / "cli-bitflip" / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["mode"] == "fault_bitflip"
    assert data["bitflip_applied"] == 1
    assert len(data["rank0_losses"]) == 8
