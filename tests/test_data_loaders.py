"""Synthetic data loader + fixture tests (ticket 1.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from goodput.data import (
    Batch,
    SyntheticDataLoader,
    generate_synthetic_batch,
    iter_synthetic_batches,
    load_batch_fixture,
    save_batch_fixture,
    train_val_split,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "test-fixtures"
FIXTURE = FIXTURES_DIR / "synthetic_gaussian_batch_n8.pt"


def test_generate_synthetic_batch_shape_and_determinism() -> None:
    a = generate_synthetic_batch(batch_size=8, input_size=16, seed=42)
    b = generate_synthetic_batch(batch_size=8, input_size=16, seed=42)
    c = generate_synthetic_batch(batch_size=8, input_size=16, seed=43)

    assert a.batch_size == 8
    assert a.input_size == 16
    assert a.targets.shape == (8, 1)
    assert torch.equal(a.inputs, b.inputs)
    assert torch.equal(a.targets, b.targets)
    assert not torch.equal(a.inputs, c.inputs)


def test_generate_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_batch(batch_size=0, input_size=4, seed=0)
    with pytest.raises(ValueError):
        generate_synthetic_batch(batch_size=4, input_size=0, seed=0)


def test_batch_validation() -> None:
    with pytest.raises(ValueError):
        Batch(inputs=torch.zeros(4), targets=torch.zeros(4, 1))
    with pytest.raises(ValueError):
        Batch(inputs=torch.zeros(4, 2), targets=torch.zeros(3, 1))


def test_iter_and_loader_lengths() -> None:
    batches = list(
        iter_synthetic_batches(num_batches=3, batch_size=4, input_size=8, seed=0)
    )
    assert len(batches) == 3
    assert batches[0].batch_size == 4

    loader = SyntheticDataLoader(num_batches=2, batch_size=4, input_size=8, seed=7)
    assert len(loader) == 2
    got = list(loader)
    assert len(got) == 2
    # Second pass is identical (deterministic)
    assert torch.equal(got[0].inputs, list(loader)[0].inputs)


def test_loader_rejects_invalid_num_batches() -> None:
    with pytest.raises(ValueError, match="num_batches"):
        SyntheticDataLoader(num_batches=0, batch_size=4, input_size=8, seed=0)
    with pytest.raises(ValueError, match="num_batches"):
        SyntheticDataLoader(num_batches=-1, batch_size=4, input_size=8, seed=0)


def test_fixture_roundtrip(tmp_path: Path) -> None:
    batch = generate_synthetic_batch(batch_size=4, input_size=8, seed=11)
    path = save_batch_fixture(batch, tmp_path / "tiny.pt", seed=11)
    loaded = load_batch_fixture(path)
    assert torch.allclose(loaded.inputs, batch.inputs)
    assert torch.allclose(loaded.targets, batch.targets)


def test_committed_fixture_loads() -> None:
    assert FIXTURE.exists(), f"missing committed fixture: {FIXTURE}"
    batch = load_batch_fixture(FIXTURE)
    assert batch.batch_size == 8
    assert batch.input_size == 16
    # Must match regenerator defaults (seed=42, n=8, d=16)
    expected = generate_synthetic_batch(batch_size=8, input_size=16, seed=42)
    assert torch.allclose(batch.inputs, expected.inputs)
    assert torch.allclose(batch.targets, expected.targets)


def test_train_val_split_deterministic() -> None:
    batch = generate_synthetic_batch(batch_size=8, input_size=4, seed=1)
    train_a, val_a = train_val_split(batch, val_fraction=0.25, seed=0)
    train_b, val_b = train_val_split(batch, val_fraction=0.25, seed=0)

    assert train_a.batch_size + val_a.batch_size == 8
    assert val_a.batch_size >= 1
    assert torch.equal(train_a.inputs, train_b.inputs)
    assert torch.equal(val_a.inputs, val_b.inputs)


def test_load_missing_fixture_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_batch_fixture(tmp_path / "missing.pt")
