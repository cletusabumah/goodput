"""Incremental checkpoint path (ticket 2.1)."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from goodput.checkpointing import (
    INCREMENTAL_FORMAT,
    INCREMENTAL_FULL_FORMAT,
    IncrementalCheckpointer,
    capture_training_state,
    restore_training_state,
)
from goodput.config import Settings
from goodput.data import SyntheticDataLoader
from goodput.models import ToyMLP
from goodput.providers import LocalFsCheckpointStore, MockCheckpointStore
from goodput.training import resume_after_crash, train_from_settings, train_steps


def _adam_toy(*, hidden: int = 128) -> tuple[ToyMLP, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = ToyMLP(32, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = SyntheticDataLoader(
        num_batches=2,
        batch_size=8,
        input_size=32,
        seed=0,
    )
    # Warm optimizer state so full dumps are heavy.
    train_steps(model=model, batches=loader, optimizer=opt, steps=3)
    return model, opt


def test_incremental_checkpointer_alternates_full_and_delta() -> None:
    model, opt = _adam_toy(hidden=64)
    ckpt = IncrementalCheckpointer(full_every=3)
    kinds = []
    for step in (1, 2, 3, 4, 5, 6):
        payload = ckpt.capture(model, opt, step=step)
        kinds.append(payload.meta["kind"])
        if payload.meta["kind"] == "delta":
            assert payload.meta["format"] == INCREMENTAL_FORMAT
            assert "optimizer" not in payload.state
            assert "model" in payload.state
            assert payload.meta["base_step"] in {1, 4}
        else:
            assert payload.meta["format"] == INCREMENTAL_FULL_FORMAT
            assert "optimizer" in payload.state
    assert kinds == ["full", "delta", "delta", "full", "delta", "delta"]


def test_incremental_restore_via_store_roundtrip() -> None:
    model, opt = _adam_toy(hidden=64)
    store = MockCheckpointStore()
    ckpt = IncrementalCheckpointer(full_every=2)

    full = ckpt.capture(model, opt, step=2)
    store.save(full)
    # Train a bit more so weights diverge from the full base.
    loader = SyntheticDataLoader(num_batches=2, batch_size=8, input_size=32, seed=1)
    train_steps(model=model, batches=loader, optimizer=opt, steps=2)
    delta = ckpt.capture(model, opt, step=4)
    store.save(delta)
    assert delta.meta["format"] == INCREMENTAL_FORMAT

    model2 = ToyMLP(32, 64)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    step = restore_training_state(model2, opt2, store.load(), device="cpu", store=store)
    assert step == 4
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.allclose(a, b)


def test_train_incremental_resume_after_crash(tmp_path: Path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    settings = Settings(
        steps=8,
        ckpt_interval=2,
        ckpt_mode="incremental",
        ckpt_full_every=2,
        num_workers=1,
        batch_size=4,
        input_size=16,
        hidden_size=32,
        seed=3,
        device="cpu",
        ci_mode=True,
    )
    first = train_from_settings(settings, checkpoint_store=store)
    assert first.ok
    assert first.last_checkpoint_step == 8
    # Use full_every=3 so a delta appears: captures at 2,4,6 → full, delta, delta
    store2 = LocalFsCheckpointStore(tmp_path / "ckpts2")
    settings2 = settings.model_copy(update={"ckpt_full_every": 3, "steps": 6})
    train_from_settings(settings2, checkpoint_store=store2)
    latest2 = store2.load()
    assert latest2.step == 6
    # captures: 2(full), 4(delta), 6(delta) — latest is delta
    assert latest2.meta.get("format") == INCREMENTAL_FORMAT

    resumed = resume_after_crash(
        settings=settings2,
        checkpoint_store=store2,
        remaining_steps=2,
    )
    assert resumed.ok
    assert resumed.resumed_from_step == 6


def test_incremental_save_faster_than_naive_on_fixture(tmp_path: Path) -> None:
    """Done-when: incremental mean save time < naive on a heavy-optimizer fixture."""
    hidden = 256
    steps = 12
    interval = 1  # save every step so we get many samples
    full_every = 4

    def _time_mode(mode: str, root: Path) -> list[float]:
        store = LocalFsCheckpointStore(root)
        torch.manual_seed(7)
        model = ToyMLP(64, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = SyntheticDataLoader(
            num_batches=4,
            batch_size=16,
            input_size=64,
            seed=7,
        )
        # Warm Adam state before timed saves.
        train_steps(model=model, batches=loader, optimizer=opt, steps=2)
        result = train_steps(
            model=model,
            batches=loader,
            optimizer=opt,
            steps=steps,
            checkpoint_store=store,
            ckpt_interval=interval,
            ckpt_mode=mode,
            ckpt_full_every=full_every,
        )
        assert result.ok
        assert len(result.ckpt_save_seconds) == steps
        return result.ckpt_save_seconds

    # Stabilize once so first-run torch/import cost is paid.
    _time_mode("naive", tmp_path / "warmup")

    naive_times = _time_mode("naive", tmp_path / "naive")
    incr_times = _time_mode("incremental", tmp_path / "incr")

    naive_mean = statistics.mean(naive_times)
    incr_mean = statistics.mean(incr_times)
    # Incremental must win on average; allow a little noise on tiny CI VMs.
    assert incr_mean < naive_mean, (
        f"expected incremental mean {incr_mean:.6f} < naive mean {naive_mean:.6f}"
    )
    # Also require a clear gap when comparing median of non-full incremental saves
    # (the model-only path) against naive.
    # Captures 1,5,9 are full under full_every=4 and 12 saves → skip those indices.
    delta_times = [t for i, t in enumerate(incr_times, start=1) if (i - 1) % full_every != 0]
    assert statistics.median(delta_times) < statistics.median(naive_times)


def test_capture_naive_still_default() -> None:
    model, opt = _adam_toy(hidden=32)
    payload = capture_training_state(model, opt, step=1)
    assert "optimizer" in payload.state
    t0 = time.perf_counter()
    capture_training_state(model, opt, step=2)
    assert time.perf_counter() - t0 >= 0
