"""Ticket 2.3 — goodput vs failure-rate plot from a sweep comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from goodput.cli import main
from goodput.evaluation.plot import load_comparison, series_from_comparison

ROWS = [
    {"ckpt_mode": "incremental", "failure_rate": 0.5, "goodput": 0.4},
    {"ckpt_mode": "naive", "failure_rate": 0.25, "goodput": 0.7},
    {"ckpt_mode": "naive", "failure_rate": 0.0, "goodput": 0.9},
    {"ckpt_mode": "incremental", "failure_rate": 0.0, "goodput": 0.85},
    {"ckpt_mode": "naive", "failure_rate": 0.5, "goodput": 0.3},
    {"ckpt_mode": "incremental", "failure_rate": 0.25, "goodput": 0.6},
]


def test_series_groups_by_mode_and_sorts_by_rate() -> None:
    series = series_from_comparison(ROWS)
    assert list(series) == ["naive", "incremental"]
    assert series["naive"] == [(0.0, 0.9), (0.25, 0.7), (0.5, 0.3)]
    assert series["incremental"] == [(0.0, 0.85), (0.25, 0.6), (0.5, 0.4)]


def test_series_rejects_empty_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="empty"):
        series_from_comparison([])
    with pytest.raises(ValueError, match="ckpt_mode"):
        series_from_comparison([{"failure_rate": 0.0, "goodput": 1.0}])


def test_load_comparison_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "comparison.json"
    json_path.write_text(json.dumps(ROWS), encoding="utf-8")
    loaded = load_comparison(json_path)
    assert loaded[0]["ckpt_mode"] == "incremental"

    csv_path = tmp_path / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ckpt_mode", "failure_rate", "goodput"])
        writer.writeheader()
        writer.writerows(ROWS)
    from_csv = load_comparison(csv_path)
    assert series_from_comparison(from_csv) == series_from_comparison(ROWS)


def test_load_comparison_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "comparison.txt"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json or \.csv"):
        load_comparison(path)


def test_cli_plot_missing_file(tmp_path: Path) -> None:
    code = main(["--plot", str(tmp_path / "missing.json")])
    assert code == 2


def test_plot_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from goodput.evaluation.plot import plot_goodput_vs_failure_rate

    out = tmp_path / "plots" / "goodput_vs_failure_rate.png"
    written = plot_goodput_vs_failure_rate(ROWS, out)
    assert written == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_cli_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    src = tmp_path / "comparison.json"
    src.write_text(json.dumps(ROWS), encoding="utf-8")
    out = tmp_path / "out.png"
    code = main(["--plot", str(src), "--plot-out", str(out)])
    assert code == 0
    assert out.is_file()
