"""Ticket 3.4 — git SHA, package versions, and config hash on every report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.metrics import (
    REQUIRED_REPORT_FIELDS,
    build_run_report,
    config_hash,
    git_sha,
    package_versions,
)
from goodput.metrics.reproducibility import CONFIG_HASH_FIELDS, config_payload
from goodput.training import train_from_settings

SAME_SEED_SETTINGS = dict(
    steps=8,
    num_workers=1,
    batch_size=4,
    input_size=8,
    hidden_size=16,
    learning_rate=1e-2,
    seed=42,
    device="cpu",
    ckpt_interval=0,
    ci_mode=True,
)


def _tiny_settings(**overrides: object) -> Settings:
    payload = dict(SAME_SEED_SETTINGS)
    payload.update(overrides)
    return Settings(**payload)  # type: ignore[arg-type]


def test_config_hash_stable_for_same_knobs() -> None:
    a = _tiny_settings(artifacts_dir=Path("/tmp/a"), ckpt_dir=Path("/tmp/a/ckpts"))
    b = _tiny_settings(artifacts_dir=Path("/tmp/b"), ckpt_dir=Path("/tmp/b/ckpts"))
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 16


def test_config_hash_changes_with_seed() -> None:
    a = _tiny_settings(seed=1)
    b = _tiny_settings(seed=2)
    assert config_hash(a) != config_hash(b)


def test_config_payload_only_includes_training_knobs() -> None:
    payload = config_payload(_tiny_settings())
    assert tuple(payload) == CONFIG_HASH_FIELDS
    assert "artifacts_dir" not in payload
    assert "ckpt_dir" not in payload


def test_package_versions_include_torch_and_goodput() -> None:
    versions = package_versions()
    assert versions["torch"]
    assert versions["goodput"]
    assert versions["torch"] != "unknown"
    assert versions["goodput"] != "unknown"


def test_git_sha_looks_like_hex_when_available() -> None:
    sha = git_sha()
    if sha is None:
        pytest.skip("git not available from this checkout")
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_build_run_report_includes_repro_fields() -> None:
    settings = _tiny_settings(run_name="repro-unit")
    report = build_run_report(
        settings=settings,
        wall_seconds=1.0,
        useful_seconds=1.0,
        steps_completed=8,
    )
    for key in REQUIRED_REPORT_FIELDS:
        assert key in report
    assert report["config_hash"] == config_hash(settings)
    assert report["package_versions"]["torch"]
    assert "git_sha" in report
    assert "git_dirty" in report
    assert "versions_hash" in report
    assert report["schema_version"] == "1.1"


def test_same_seed_runs_match_within_tolerance() -> None:
    """Done-when: two runs with the same seed match within tolerance."""
    settings = _tiny_settings()
    first = train_from_settings(settings)
    second = train_from_settings(settings)
    assert first.ok and second.ok
    assert first.steps_completed == second.steps_completed == 8
    assert first.losses == pytest.approx(second.losses, rel=1e-5, abs=1e-6)
    assert first.final_loss == pytest.approx(second.final_loss, rel=1e-5, abs=1e-6)
    report_a = build_run_report(
        settings=settings,
        wall_seconds=first.wall_seconds,
        useful_seconds=first.useful_seconds,
        steps_completed=first.steps_completed,
        final_loss=first.final_loss,
    )
    report_b = build_run_report(
        settings=settings,
        wall_seconds=second.wall_seconds,
        useful_seconds=second.useful_seconds,
        steps_completed=second.steps_completed,
        final_loss=second.final_loss,
    )
    assert report_a["config_hash"] == report_b["config_hash"]
    assert report_a["seed"] == report_b["seed"] == 42
    assert report_a["final_loss"] == pytest.approx(report_b["final_loss"], rel=1e-5, abs=1e-6)
    # Wall clock is not part of the Done-when — only the seeded training outcome.


def test_different_seed_runs_diverge() -> None:
    a = train_from_settings(_tiny_settings(seed=1))
    b = train_from_settings(_tiny_settings(seed=2))
    assert a.final_loss != pytest.approx(b.final_loss, rel=1e-5, abs=1e-6)


def test_cli_report_has_repro_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    cfg = tmp_path / "repro.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-repro",
                "mode: train",
                "num_workers: 1",
                "steps: 4",
                "ckpt_interval: 0",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 3",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(cfg)])
    assert code == 0
    report_path = tmp_path / "artifacts" / "reports" / "cli-repro" / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("git_sha", "config_hash", "package_versions"):
        assert key in data
    assert data["config_hash"]
    assert data["package_versions"]["torch"]
