"""Phase 0 smoke: package importable and basic invariants hold."""

from __future__ import annotations

from goodput import __version__
from goodput.config import Settings
from goodput.metrics import compute_goodput
from goodput.providers import CheckpointPayload, MockCheckpointStore


def test_version_semver_shape() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_settings_defaults() -> None:
    s = Settings()
    assert s.device == "cpu"
    assert s.num_workers >= 1
    assert s.ckpt_mode in {"naive", "incremental"}


def test_goodput_metric() -> None:
    assert compute_goodput(80.0, 100.0) == 0.8


def test_mock_checkpoint_roundtrip(mock_checkpoint_store: MockCheckpointStore) -> None:
    payload = CheckpointPayload(step=3, state={"w": 1.0})
    loc = mock_checkpoint_store.save(payload)
    loaded = mock_checkpoint_store.load(loc)
    assert loaded.step == 3
    assert loaded.state["w"] == 1.0
