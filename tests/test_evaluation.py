"""Placeholder — expanded in ticket 1.7."""

from __future__ import annotations

from goodput.metrics import compute_goodput


def test_compute_goodput_clamps_at_one() -> None:
    assert compute_goodput(150.0, 100.0) == 1.0


def test_compute_goodput_rejects_bad_wall() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_goodput(1.0, 0.0)
