"""Ticket 3.3 — public $/GPU-hr × measured goodput delta writes Markdown/JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from goodput.evaluation.dollar import (
    DEFAULT_CLUSTER_SIZE,
    DEFAULT_HOURS,
    DEFAULT_USD_PER_GPU_HOUR,
    DOLLAR_ROW_FIELDS,
    DollarSpec,
    compute_dollar_impact,
    estimate_from_comparison,
    load_dollar_yaml,
    render_dollar_table,
    run_dollar,
)

ROWS = [
    {"ckpt_mode": "naive", "failure_rate": 0.0, "goodput": 0.90},
    {"ckpt_mode": "incremental", "failure_rate": 0.0, "goodput": 0.85},
    {"ckpt_mode": "naive", "failure_rate": 0.5, "goodput": 0.30},
    {"ckpt_mode": "incremental", "failure_rate": 0.5, "goodput": 0.40},
]


def test_compute_dollar_impact_matches_vision_formula() -> None:
    # 16,384 GPUs × $3.99/hr × 1,296 h × 0.10 goodput points.
    value = compute_dollar_impact(
        cluster_size=DEFAULT_CLUSTER_SIZE,
        usd_per_gpu_hour=DEFAULT_USD_PER_GPU_HOUR,
        hours=DEFAULT_HOURS,
        goodput_delta=0.10,
    )
    assert value == pytest.approx(16_384 * 3.99 * 1_296 * 0.10)


def test_compute_dollar_impact_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="cluster_size"):
        compute_dollar_impact(cluster_size=0, usd_per_gpu_hour=1.0, hours=1.0, goodput_delta=0.1)
    with pytest.raises(ValueError, match="usd_per_gpu_hour"):
        compute_dollar_impact(cluster_size=1, usd_per_gpu_hour=-1.0, hours=1.0, goodput_delta=0.1)
    with pytest.raises(ValueError, match="hours"):
        compute_dollar_impact(cluster_size=1, usd_per_gpu_hour=1.0, hours=-1.0, goodput_delta=0.1)


def test_estimate_pairs_modes_at_each_rate() -> None:
    spec = DollarSpec(
        path=Path("x.yaml"),
        name="t",
        comparison=None,
        sweep=None,
        gpu_sku="NVIDIA H100 SXM 80GB",
        usd_per_gpu_hour=3.99,
        price_source="test",
        price_url="https://lambda.ai/pricing",
        cluster_size=16384,
        hours=1296,
        baseline_mode="naive",
        improved_mode="incremental",
        output_dir=Path("out"),
    )
    rows = estimate_from_comparison(ROWS, spec)
    assert [r["failure_rate"] for r in rows] == [0.0, 0.5]
    assert rows[0]["goodput_delta"] == pytest.approx(-0.05)
    assert rows[1]["goodput_delta"] == pytest.approx(0.10)
    assert rows[1]["dollar_delta"] == pytest.approx(16_384 * 3.99 * 1_296 * 0.10)


def test_estimate_requires_paired_modes() -> None:
    spec = DollarSpec(
        path=Path("x.yaml"),
        name="t",
        comparison=None,
        sweep=None,
        gpu_sku="x",
        usd_per_gpu_hour=1.0,
        price_source="t",
        price_url="https://example.invalid",
        cluster_size=1,
        hours=1.0,
        baseline_mode="naive",
        improved_mode="incremental",
        output_dir=Path("out"),
    )
    with pytest.raises(ValueError, match="paired"):
        estimate_from_comparison(
            [{"ckpt_mode": "naive", "failure_rate": 0.0, "goodput": 1.0}],
            spec,
        )


def test_load_committed_dollar_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_dollar_yaml(root / "experiments" / "dollar.yaml")
    assert spec.name == "dollar-impact"
    assert spec.cluster_size == 16384
    assert spec.hours == pytest.approx(1296)
    assert spec.usd_per_gpu_hour == pytest.approx(3.99)
    assert spec.baseline_mode == "naive"
    assert spec.improved_mode == "incremental"
    assert spec.gpu_sku.startswith("NVIDIA H100")
    assert "not a quote" in spec.price_source.lower()
    assert spec.sweep is not None
    assert spec.sweep.name == "sweep.yaml"
    assert spec.sweep.is_file()


def test_load_dollar_unknown_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\nmode: dollar\ncomparison: c.json\nnot_a_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown dollar keys"):
        load_dollar_yaml(path)


def test_load_dollar_rejects_same_modes(tmp_path: Path) -> None:
    path = tmp_path / "same.yaml"
    path.write_text(
        "\n".join(
            [
                "name: x",
                "mode: dollar",
                "comparison: c.json",
                "baseline_mode: naive",
                "improved_mode: naive",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_dollar_yaml(path)


def test_run_dollar_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(ROWS), encoding="utf-8")
    cfg = tmp_path / "tiny-dollar.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: tiny-dollar",
                "mode: dollar",
                f"comparison: {comparison}",
                "usd_per_gpu_hour: 3.99",
                "cluster_size: 100",
                "hours: 10",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_dollar_yaml(cfg)
    result = run_dollar(spec)
    assert result.json_path is not None and result.csv_path is not None
    assert result.table_path is not None
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    assert result.table_path.is_file()
    assert len(result.rows) == 2

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["mode"] == "dollar"
    assert "not a quote" in data["disclaimer"].lower()
    assert data["cluster_size"] == 100
    assert data["usd_per_gpu_hour"] == pytest.approx(3.99)
    assert len(data["rows"]) == 2
    rates = [row["failure_rate"] for row in data["rows"]]
    assert rates == [0.0, 0.5]
    for row in data["rows"]:
        for key in DOLLAR_ROW_FIELDS:
            assert key in row
    high_rate = next(r for r in data["rows"] if r["failure_rate"] == 0.5)
    assert high_rate["dollar_delta"] == pytest.approx(100 * 3.99 * 10 * 0.10)

    markdown = result.table_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Dollar impact of a measured goodput delta")
    assert "Back-of-envelope" in markdown
    assert "| failure_rate |" in markdown
    assert "3.99" in markdown

    with result.csv_path.open(encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [h for h in csv_rows[0]] == list(DOLLAR_ROW_FIELDS)
    assert len(csv_rows) == 2


def test_render_dollar_table_includes_disclaimer_and_rows(tmp_path: Path) -> None:
    cfg = tmp_path / "d.yaml"
    comparison = tmp_path / "c.json"
    comparison.write_text("[]", encoding="utf-8")
    cfg.write_text(
        "\n".join(
            [
                "name: d",
                "mode: dollar",
                f"comparison: {comparison}",
                "cluster_size: 2",
                "hours: 3",
                "usd_per_gpu_hour: 4.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_dollar_yaml(cfg)
    text = render_dollar_table(
        [
            {
                "failure_rate": 0.5,
                "baseline_mode": "naive",
                "improved_mode": "incremental",
                "baseline_goodput": 0.3,
                "improved_goodput": 0.4,
                "goodput_delta": 0.1,
                "gpu_hours": 6.0,
                "dollar_delta": 2.4,
            }
        ],
        spec,
    )
    assert "not a quote" in text.lower()
    assert "0.1000" in text
    assert "2.40" in text


def test_cli_dollar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from goodput.cli import main

    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(ROWS), encoding="utf-8")
    out = tmp_path / "artifacts"
    cfg = tmp_path / "cli-dollar.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: cli-dollar",
                "mode: dollar",
                f"comparison: {comparison}",
                "cluster_size: 8",
                "hours: 24",
                "usd_per_gpu_hour: 3.99",
                f"artifacts_dir: {out}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = main(["--dollar", str(cfg)])
    assert code == 0
    table_dir = out / "sweeps" / "cli-dollar"
    assert (table_dir / "dollar.json").is_file()
    assert (table_dir / "dollar.csv").is_file()
    assert (table_dir / "table.md").is_file()
    data = json.loads((table_dir / "dollar.json").read_text(encoding="utf-8"))
    assert data["mode"] == "dollar"
    assert len(data["rows"]) == 2


def test_cli_dollar_missing_config(tmp_path: Path) -> None:
    from goodput.cli import main

    code = main(["--dollar", str(tmp_path / "missing.yaml")])
    assert code == 2


def test_run_dollar_runs_sweep_when_comparison_missing(tmp_path: Path) -> None:
    sweep_cfg = tmp_path / "tiny-sweep.yaml"
    sweep_cfg.write_text(
        "\n".join(
            [
                "name: tiny-sweep",
                "mode: sweep",
                "ckpt_modes:",
                "  - naive",
                "  - incremental",
                "failure_rates:",
                "  - 0.5",
                "num_workers: 1",
                "steps: 6",
                "ckpt_interval: 2",
                "ckpt_full_every: 2",
                "batch_size: 4",
                "input_size: 8",
                "hidden_size: 8",
                "seed: 1",
                f"ckpt_dir: {tmp_path / 'ckpts'}",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "dollar.yaml"
    cfg.write_text(
        "\n".join(
            [
                "name: from-sweep",
                "mode: dollar",
                f"comparison: {tmp_path / 'missing-comparison.json'}",
                f"sweep: {sweep_cfg}",
                "cluster_size: 10",
                "hours: 2",
                "usd_per_gpu_hour: 3.99",
                f"artifacts_dir: {tmp_path / 'artifacts'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_dollar_yaml(cfg)
    result = run_dollar(spec)
    assert result.json_path is not None
    assert len(result.rows) == 1
    assert result.rows[0]["failure_rate"] == pytest.approx(0.5)
    assert result.comparison_path is not None
    assert result.comparison_path.is_file()
