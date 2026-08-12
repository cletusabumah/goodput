"""Multi-process launcher tests (ticket 1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.training.multiprocess import launch_workers, train_multiprocess_from_settings


def test_launch_two_workers_short_run() -> None:
    """Done-when for 1.4: N=2 workers complete a short synchronized run."""
    settings = Settings(
        num_workers=2,
        steps=5,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        ci_mode=True,
    )
    result = launch_workers(world_size=2, steps=5, settings=settings)

    assert result.world_size == 2
    assert result.ok, [w.error for w in result.workers]
    assert len(result.workers) == 2
    assert all(w.steps_completed == 5 for w in result.workers)
    assert all(w.ok for w in result.workers)


def test_launch_rejects_bad_world_size() -> None:
    settings = Settings(ci_mode=True, device="cpu")
    with pytest.raises(ValueError, match="world_size"):
        launch_workers(world_size=0, steps=1, settings=settings)


def test_launch_two_workers_with_checkpoints(tmp_path: Path) -> None:
    settings = Settings(
        num_workers=2,
        steps=4,
        ckpt_interval=2,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
        ci_mode=True,
    )
    ckpt_dir = tmp_path / "ckpts"
    result = launch_workers(
        world_size=2,
        steps=4,
        settings=settings,
        ckpt_dir=ckpt_dir,
    )
    assert result.ok, [w.error for w in result.workers]
    assert result.last_checkpoint_step == 4
    assert len(result.ckpt_save_seconds) >= 1
    assert all(t >= 0 for t in result.ckpt_save_seconds)
    assert (ckpt_dir / "step_000002.pt").is_file()
    assert (ckpt_dir / "step_000004.pt").is_file()
    assert (ckpt_dir / "latest.pt").is_file()


def test_train_multiprocess_from_settings_single_falls_back() -> None:
    settings = Settings(
        num_workers=1,
        steps=3,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=0,
        device="cpu",
        ci_mode=True,
    )
    result = train_multiprocess_from_settings(settings)
    # Single-process path returns TrainResult
    assert result.ok
    assert result.steps_completed == 3
