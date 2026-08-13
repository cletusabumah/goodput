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


def test_incremental_save_faster_than_naive_on_fixture() -> None:
    """Done-when: model-only incremental capture is faster than a full naive dump.

    Time ``capture()`` (clone/serialize), not ``torch.save`` to disk. GitHub
    runners make end-to-end save latency too noisy for a strict inequality —
    a single stalled write inverted the old mean-of-all-saves check.
    Interleave naive and incremental so VM noise hits both paths equally.
    """
    model, opt = _adam_toy(hidden=256)
    full_every = 4
    captures = 16

    # Pay first-call clone cost for both paths before the timed loop.
    capture_training_state(model, opt, step=0)
    IncrementalCheckpointer(full_every=full_every).capture(model, opt, step=0)

    ckpt = IncrementalCheckpointer(full_every=full_every)
    naive_times: list[float] = []
    delta_times: list[float] = []
    for step in range(1, captures + 1):
        t0 = time.perf_counter()
        capture_training_state(model, opt, step=step)
        naive_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        payload = ckpt.capture(model, opt, step=step)
        elapsed = time.perf_counter() - t0
        if payload.meta["kind"] == "delta":
            delta_times.append(elapsed)

    naive_med = statistics.median(naive_times)
    delta_med = statistics.median(delta_times)
    assert delta_times, "expected model-only incremental captures"
    assert delta_med < naive_med, (
        f"expected incremental delta median {delta_med:.6f} < naive median {naive_med:.6f}"
    )


def test_capture_naive_still_default() -> None:
    model, opt = _adam_toy(hidden=32)
    payload = capture_training_state(model, opt, step=1)
    assert "optimizer" in payload.state
    t0 = time.perf_counter()
    capture_training_state(model, opt, step=2)
    assert time.perf_counter() - t0 >= 0
