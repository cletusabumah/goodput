"""Ticket 1.7 — goodput math + JSON run report required fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodput.config import Settings
from goodput.metrics import (
    REQUIRED_REPORT_FIELDS,
    assert_required_fields,
    build_run_report,
    compute_goodput,
    compute_wasted_gpu_hours,
    emit_run_report,
)
from goodput.providers import LocalFsCheckpointStore, MockMetricsSink, build_providers
from goodput.training import resume_after_crash, train_from_settings


def test_compute_goodput_clamps_at_one() -> None:
    assert compute_goodput(150.0, 100.0) == 1.0


def test_compute_goodput_rejects_bad_wall() -> None:
    with pytest.raises(ValueError):
        compute_goodput(1.0, 0.0)


def test_compute_wasted_gpu_hours() -> None:
    # 100s wall, 80s useful → 20s waste × 2 workers / 3600
    assert compute_wasted_gpu_hours(100.0, 80.0, 2) == pytest.approx(20.0 * 2 / 3600.0)


def test_build_run_report_has_required_fields() -> None:
    settings = Settings(run_name="unit-report", seed=7, num_workers=2)
    report = build_run_report(
        settings=settings,
        wall_seconds=10.0,
        useful_seconds=8.0,
        steps_completed=20,
        ckpt_save_seconds=[0.01, 0.03],
        ckpt_restore_seconds=0.05,
        final_loss=0.5,
    )
    assert_required_fields(report)
    for key in REQUIRED_REPORT_FIELDS:
        assert key in report
    assert report["goodput"] == pytest.approx(0.8)
    assert report["ckpt_save_s"] == pytest.approx(0.02)
    assert report["ckpt_restore_s"] == pytest.approx(0.05)
    assert report["wasted_gpu_hours"] == pytest.approx(compute_wasted_gpu_hours(10.0, 8.0, 2))
    assert report["run_name"] == "unit-report"
    assert report["seed"] == 7


def test_emit_run_report_to_mock_sink() -> None:
    settings = Settings(run_name="mock-emit", metrics_provider="mock")
    sink = MockMetricsSink()
    report = build_run_report(
        settings=settings,
        wall_seconds=5.0,
        useful_seconds=5.0,
        steps_completed=4,
    )
    emit_run_report(sink, report)
    assert len(sink.emitted) == 1
    assert sink.emitted[0]["goodput"] == pytest.approx(1.0)


def test_emit_run_report_writes_json_file(tmp_path: Path) -> None:
    settings = Settings(
        run_name="json-emit",
        metrics_provider="json_file",
        artifacts_dir=tmp_path,
        num_workers=1,
    )
    providers = build_providers(settings, artifacts_dir=tmp_path)
    report = build_run_report(
        settings=settings,
        wall_seconds=2.0,
        useful_seconds=1.5,
        steps_completed=8,
        ckpt_save_seconds=0.01,
        ckpt_restore_seconds=0.02,
    )
    emit_run_report(providers.metrics, report)

    path = tmp_path / "reports" / "json-emit" / "report.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in REQUIRED_REPORT_FIELDS:
        assert key in data
    assert 0.0 <= data["goodput"] <= 1.0


def test_train_from_settings_records_timings(tmp_path: Path) -> None:
    settings = Settings(
        steps=6,
        ckpt_interval=3,
        num_workers=1,
        seed=1,
        batch_size=4,
        input_size=8,
        hidden_size=8,
    )
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    result = train_from_settings(settings, checkpoint_store=store)
    assert result.ok
    assert result.wall_seconds > 0
    assert result.useful_seconds > 0
    assert result.useful_seconds <= result.wall_seconds + 1e-6
    assert len(result.ckpt_save_seconds) >= 1
    assert all(t >= 0 for t in result.ckpt_save_seconds)

    report = build_run_report(
        settings=settings,
        wall_seconds=result.wall_seconds,
        useful_seconds=result.useful_seconds,
        steps_completed=result.steps_completed,
        ckpt_save_seconds=result.ckpt_save_seconds,
        ckpt_restore_seconds=result.ckpt_restore_seconds,
        final_loss=result.final_loss,
    )
    assert_required_fields(report)
    assert 0.0 < report["goodput"] <= 1.0


def test_resume_records_restore_latency(tmp_path: Path) -> None:
    settings = Settings(
        steps=6,
        ckpt_interval=3,
        num_workers=1,
        seed=2,
        batch_size=4,
        input_size=8,
        hidden_size=8,
    )
    store = LocalFsCheckpointStore(tmp_path / "ckpts")
    train_from_settings(settings, checkpoint_store=store)
    resumed = resume_after_crash(
        settings=settings,
        checkpoint_store=store,
        remaining_steps=2,
    )
    assert resumed.ok
    assert resumed.ckpt_restore_seconds > 0
    assert resumed.resumed_from_step == 6
