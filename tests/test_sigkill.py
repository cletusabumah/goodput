"""SIGKILL fault injection + recovery tests (ticket 1.6)."""

from __future__ import annotations

import time
from multiprocessing import Process
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.providers.faults import ProcessFaultInjector
from goodput.training.recovery import run_sigkill_and_recover


def _sleep_worker(seconds: float = 30.0) -> None:
    time.sleep(seconds)


def test_process_fault_injector_real_sigkill() -> None:
    """Unit-level: injector with dry_run=False actually kills a registered PID."""
    proc = Process(target=_sleep_worker, args=(30.0,))
    proc.start()
    assert proc.pid is not None

    inj = ProcessFaultInjector(inject_at=1, fault="kill", dry_run=False)
    inj.register_worker(0, proc.pid)
    assert inj.maybe_inject(1, 0) == "kill"

    proc.join(timeout=5)
    assert not proc.is_alive()
    # Negative exit codes / signals vary by platform; just require death.
    assert proc.exitcode != 0


def test_sigkill_recovery_two_workers(tmp_path: Path) -> None:
    """Done-when: scripted kill mid-run; recovery resumes from checkpoint step."""
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
    )
    result = run_sigkill_and_recover(
        settings,
        ckpt_dir=tmp_path / "ckpts",
        kill_at_step=4,
        kill_rank=1,
        remaining_steps=2,
    )

    assert result.injected == [(4, 1, "kill")]
    assert result.checkpoint_step == 4
    assert result.killed_pid is not None
    assert result.ok
    assert result.recovered.resumed_from_step == 4
    assert result.recovered.steps_completed == 2
    assert result.recovered.ok


def test_sigkill_requires_checkpoint_aligned_step(tmp_path: Path) -> None:
    settings = Settings(
        num_workers=2,
        steps=6,
        ckpt_interval=5,
        device="cpu",
        batch_size=4,
        input_size=8,
        hidden_size=16,
    )
    with pytest.raises(ValueError, match="multiple of ckpt_interval"):
        run_sigkill_and_recover(
            settings,
            ckpt_dir=tmp_path / "ckpts",
            kill_at_step=3,
        )
