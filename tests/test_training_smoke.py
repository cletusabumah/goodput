"""Training smoke tests (ticket 1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from goodput.config import Settings
from goodput.data import SyntheticDataLoader, generate_synthetic_batch
from goodput.models import ToyMLP
from goodput.training import train_from_settings, train_on_fixture, train_steps

FIXTURE = (
    Path(__file__).resolve().parent.parent / "test-fixtures" / "synthetic_gaussian_batch_n8.pt"
)


def test_toy_mlp_forward_shape() -> None:
    model = ToyMLP(input_size=16, hidden_size=32)
    x = torch.randn(4, 16)
    y = model(x)
    assert y.shape == (4, 1)


def test_train_steps_finite_and_completes() -> None:
    torch.manual_seed(0)
    model = ToyMLP(input_size=8, hidden_size=16)
    loader = SyntheticDataLoader(num_batches=4, batch_size=4, input_size=8, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    result = train_steps(model=model, batches=loader, optimizer=opt, steps=10, device="cpu")

    assert result.steps_completed == 10
    assert result.ok
    assert all(loss == loss and abs(loss) != float("inf") for loss in result.losses)


def test_train_steps_loss_tends_to_decrease() -> None:
    torch.manual_seed(1)
    batch = generate_synthetic_batch(batch_size=16, input_size=8, seed=1)
    model = ToyMLP(input_size=8, hidden_size=32)
    opt = torch.optim.SGD(model.parameters(), lr=5e-2)
    result = train_steps(model=model, batches=[batch], optimizer=opt, steps=40, device="cpu")

    assert result.ok
    # Average of last 5 vs first 5 — should drop on this easy linear-ish task
    early = sum(result.losses[:5]) / 5
    late = sum(result.losses[-5:]) / 5
    assert late < early


def test_train_on_committed_fixture() -> None:
    assert FIXTURE.exists()
    result = train_on_fixture(FIXTURE, steps=15, learning_rate=1e-2, seed=42)
    assert result.steps_completed == 15
    assert result.ok


def test_train_from_settings_short_run() -> None:
    settings = Settings(
        steps=8,
        batch_size=4,
        input_size=8,
        hidden_size=16,
        learning_rate=1e-2,
        seed=7,
        device="cpu",
        ci_mode=True,
    )
    result = train_from_settings(settings)
    assert result.steps_completed == 8
    assert result.ok


def test_train_steps_rejects_zero_steps() -> None:
    model = ToyMLP(input_size=4, hidden_size=8)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="steps"):
        train_steps(
            model=model,
            batches=[generate_synthetic_batch(batch_size=2, input_size=4, seed=0)],
            optimizer=opt,
            steps=0,
        )
