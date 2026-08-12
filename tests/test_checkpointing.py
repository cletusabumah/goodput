"""Naive checkpoint save/restore tests (ticket 1.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from goodput.checkpointing import capture_training_state, restore_training_state
from goodput.config import Settings
from goodput.data import SyntheticDataLoader
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore, MockCheckpointStore
from goodput.training import resume_after_crash, train_from_settings, train_steps


def _batch_pool_size(steps: int) -> int:
    """Mirror uninterrupted / resume loader sizing: max(1, min(steps, 16))."""
    return max(1, min(steps, 16))


def _tiny_settings(**overrides: object) -> Settings:
    base = dict(
        steps=10,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
        ci_mode=True,
        ckpt_interval=5,
        num_workers=1,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_capture_restore_roundtrip_preserves_weights() -> None:
    torch.manual_seed(0)
    model = ToyMLP(8, 16)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    # one train step so optimizer state is non-empty
    loader = SyntheticDataLoader(num_batches=1, batch_size=4, input_size=8, seed=0)
    train_steps(model=model, batches=loader, optimizer=opt, steps=1)

    payload = capture_training_state(model, opt, step=1)
    model2 = ToyMLP(8, 16)
    opt2 = torch.optim.SGD(model2.parameters(), lr=1e-2)
    step = restore_training_state(model2, opt2, payload, device="cpu")

    assert step == 1
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_train_writes_checkpoints_on_interval(tmp_path: Path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    settings = _tiny_settings(steps=10, ckpt_interval=5)
    result = train_from_settings(settings, checkpoint_store=store)

    assert result.ok
    assert result.last_checkpoint_step == 10
    assert store.latest() is not None
    assert store.load().step == 10
    # Interval saves: step_000005.pt and step_000010.pt
    assert (tmp_path / "ckpts" / "step_000005.pt").exists()
    assert (tmp_path / "ckpts" / "step_000010.pt").exists()


def test_resume_after_crash_matches_last_checkpoint_step(tmp_path: Path) -> None:
    """Done-when: soft-kill, restore, resume step equals last checkpoint."""
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    settings = _tiny_settings(steps=5, ckpt_interval=5)

    first = train_from_settings(settings, checkpoint_store=store)
    assert first.last_checkpoint_step == 5
    assert store.load().step == 5

    # Soft crash: drop in-memory trainer, reload from disk only
    resumed = resume_after_crash(
        settings=settings,
        checkpoint_store=store,
        remaining_steps=4,
    )
    assert resumed.ok
    assert resumed.resumed_from_step == 5
    assert resumed.steps_completed == 4
    # Latest ckpt advances with continued training (interval 5 → save at 5+4?
    # steps 5,6,7,8 → completed 6,7,8,9 — final save at end → step 9)
    assert store.load().step == 9


def test_mock_store_resume_path() -> None:
    store = MockCheckpointStore()
    settings = _tiny_settings(steps=6, ckpt_interval=3)
    train_from_settings(settings, checkpoint_store=store)
    assert store.save_count >= 2
    assert store.load().step == 6

    resumed = resume_after_crash(settings=settings, checkpoint_store=store, remaining_steps=2)
    assert resumed.resumed_from_step == 6
    assert resumed.ok


def test_resume_loader_pool_matches_uninterrupted_when_remaining_extends(
    tmp_path: Path,
) -> None:
    """
    Resume must not grow the batch cycle when remaining work past settings.steps.

    Old bug: pool = max(steps, ckpt + remaining) changed cycle length vs
    uninterrupted min(steps, 16), so skip landed on the wrong batches.
    """
    settings = _tiny_settings(steps=8, ckpt_interval=4, seed=3)
    pool = _batch_pool_size(settings.steps)
    assert pool == 8

    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    device = "cpu"
    torch.manual_seed(settings.seed)
    model = ToyMLP(settings.input_size, settings.hidden_size)
    opt = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    loader = SyntheticDataLoader(
        num_batches=pool,
        batch_size=settings.batch_size,
        input_size=settings.input_size,
        seed=settings.seed,
        device=device,
    )
    first = train_steps(
        model=model,
        batches=loader,
        optimizer=opt,
        steps=4,
        device=device,
        checkpoint_store=store,
        ckpt_interval=4,
    )
    assert first.last_checkpoint_step == 4

    # remaining=6 → ckpt+remaining=10 > settings.steps=8 (old code grew pool to 10)
    resumed = resume_after_crash(
        settings=settings,
        checkpoint_store=store,
        remaining_steps=6,
    )
    assert resumed.resumed_from_step == 4
    assert resumed.steps_completed == 6

    # Uninterrupted 10 steps with the same 8-batch pool
    torch.manual_seed(settings.seed)
    model_b = ToyMLP(settings.input_size, settings.hidden_size)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=settings.learning_rate)
    loader_b = SyntheticDataLoader(
        num_batches=pool,
        batch_size=settings.batch_size,
        input_size=settings.input_size,
        seed=settings.seed,
        device=device,
    )
    baseline = train_steps(
        model=model_b,
        batches=loader_b,
        optimizer=opt_b,
        steps=10,
        device=device,
    )
    for got, expected in zip(resumed.losses, baseline.losses[4:], strict=True):
        assert got == pytest.approx(expected, rel=1e-5, abs=1e-5)
