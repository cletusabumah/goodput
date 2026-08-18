"""Dollar-impact estimate from measured goodput deltas (ticket 3.3).

Back-of-envelope, not a quote:

    dollar_delta = cluster_size × usd_per_gpu_hour × hours × (goodput_b − goodput_a)

``goodput_a`` / ``goodput_b`` come from a sweep comparison table (naive vs
incremental at the same failure rate). Cluster size and hours are a *simulated*
job (Llama-scale GPU count × multi-day wall), not this laptop run. GPU price is
a public list rate and must stay labeled as such.

Writes JSON + CSV + a markdown table under ``artifacts/sweeps/<name>/``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from goodput.evaluation.plot import load_comparison
from goodput.evaluation.sweep import SweepResult, load_sweep_yaml, run_sweep

# Lambda Cloud 8× H100 SXM on-demand list (~Aug 2026). Public page, not a quote.
# https://lambda.ai/pricing
DEFAULT_GPU_SKU = "NVIDIA H100 SXM 80GB"
DEFAULT_USD_PER_GPU_HOUR = 3.99
DEFAULT_PRICE_SOURCE = (
    "Lambda Cloud 8x H100 SXM on-demand list, $3.99/GPU-hr "
    "(public, Aug 2026; not a quote — prices change)"
)
DEFAULT_PRICE_URL = "https://lambda.ai/pricing"
# Vision / Llama 3 405B-shaped cluster: ~16,384 GPUs over ~54 days.
DEFAULT_CLUSTER_SIZE = 16_384
DEFAULT_HOURS = 54 * 24  # 1296

DOLLAR_YAML_KEYS = frozenset(
    {
        "name",
        "mode",
        "comparison",
        "sweep",
        "gpu_sku",
        "usd_per_gpu_hour",
        "price_source",
        "price_url",
        "cluster_size",
        "hours",
        "baseline_mode",
        "improved_mode",
        "output_dir",
        "artifacts_dir",
    }
)

DOLLAR_ROW_FIELDS: tuple[str, ...] = (
    "failure_rate",
    "baseline_mode",
    "improved_mode",
    "baseline_goodput",
    "improved_goodput",
    "goodput_delta",
    "gpu_hours",
    "dollar_delta",
)

DISCLAIMER = (
    "Back-of-envelope only. Multiplies a *measured* toy-sweep goodput delta by a "
    "*simulated* cluster (GPU count × hours) and a *public* on-demand GPU list "
    "price. This is not a quote, invoice, or capacity reservation."
)


@dataclass(frozen=True)
class DollarSpec:
    """Parsed dollar YAML: pricing knobs + where measured goodput lives."""

    path: Path
    name: str
    comparison: Path | None
    sweep: Path | None
    gpu_sku: str
    usd_per_gpu_hour: float
    price_source: str
    price_url: str
    cluster_size: int
    hours: float
    baseline_mode: str
    improved_mode: str
    output_dir: Path


@dataclass
class DollarResult:
    spec: DollarSpec
    rows: list[dict[str, Any]] = field(default_factory=list)
    json_path: Path | None = None
    csv_path: Path | None = None
    table_path: Path | None = None
    comparison_path: Path | None = None


def compute_dollar_impact(
    *,
    cluster_size: int,
    usd_per_gpu_hour: float,
    hours: float,
    goodput_delta: float,
) -> float:
    """Simulated cluster GPU-hours × public $/hr × measured goodput delta."""
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if usd_per_gpu_hour < 0:
        raise ValueError("usd_per_gpu_hour must be >= 0")
    if hours < 0:
        raise ValueError("hours must be >= 0")
    return float(cluster_size) * float(usd_per_gpu_hour) * float(hours) * float(goodput_delta)


def load_dollar_yaml(path: str | Path) -> DollarSpec:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"dollar config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"dollar YAML must be a mapping: {config_path}")

    data: dict[str, Any] = dict(raw)
    unknown = sorted(k for k in data if k not in DOLLAR_YAML_KEYS)
    if unknown:
        raise ValueError(f"unknown dollar keys in {config_path}: {unknown}")

    name = str(data.pop("name", config_path.stem))
    mode = data.pop("mode", "dollar")
    if mode != "dollar":
        raise ValueError(f"dollar YAML mode must be 'dollar', got {mode!r}")

    comparison_raw = data.pop("comparison", None)
    sweep_raw = data.pop("sweep", None)
    if comparison_raw is None and sweep_raw is None:
        raise ValueError("dollar YAML needs comparison and/or sweep")

    gpu_sku = str(data.pop("gpu_sku", DEFAULT_GPU_SKU))
    usd_per_gpu_hour = float(data.pop("usd_per_gpu_hour", DEFAULT_USD_PER_GPU_HOUR))
    if usd_per_gpu_hour < 0:
        raise ValueError("usd_per_gpu_hour must be >= 0")
    price_source = str(data.pop("price_source", DEFAULT_PRICE_SOURCE))
    price_url = str(data.pop("price_url", DEFAULT_PRICE_URL))
    cluster_size = int(data.pop("cluster_size", DEFAULT_CLUSTER_SIZE))
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    hours = float(data.pop("hours", DEFAULT_HOURS))
    if hours < 0:
        raise ValueError("hours must be >= 0")
    baseline_mode = str(data.pop("baseline_mode", "naive"))
    improved_mode = str(data.pop("improved_mode", "incremental"))
    if baseline_mode == improved_mode:
        raise ValueError("baseline_mode and improved_mode must differ")

    artifacts_dir = Path(str(data.pop("artifacts_dir", "artifacts")))
    output_dir_raw = data.pop("output_dir", None)
    output_dir = Path(output_dir_raw) if output_dir_raw else artifacts_dir / "sweeps" / name

    comparison = _resolve_optional_path(comparison_raw, config_path)
    sweep = _resolve_optional_path(sweep_raw, config_path)

    return DollarSpec(
        path=config_path.resolve(),
        name=name,
        comparison=comparison,
        sweep=sweep,
        gpu_sku=gpu_sku,
        usd_per_gpu_hour=usd_per_gpu_hour,
        price_source=price_source,
        price_url=price_url,
        cluster_size=cluster_size,
        hours=hours,
        baseline_mode=baseline_mode,
        improved_mode=improved_mode,
        output_dir=output_dir,
    )


def _resolve_optional_path(raw: Any, config_path: Path) -> Path | None:
    if raw is None or raw == "":
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    from_cwd = Path.cwd() / candidate
    from_yaml = config_path.parent / candidate
    if from_cwd.is_file():
        return from_cwd.resolve()
    if from_yaml.is_file():
        return from_yaml.resolve()
    # Missing comparison is OK when sweep: will generate it; prefer cwd (repo recipes).
    return from_cwd.resolve()


def estimate_from_comparison(
    rows: list[dict[str, Any]],
    spec: DollarSpec,
) -> list[dict[str, Any]]:
    """Pair baseline vs improved goodput at each failure rate, then apply the $ model."""
    by_rate: dict[float, dict[str, float]] = {}
    for row in rows:
        if "ckpt_mode" not in row or "failure_rate" not in row or "goodput" not in row:
            raise ValueError("comparison rows need ckpt_mode, failure_rate, and goodput")
        rate = float(row["failure_rate"])
        by_rate.setdefault(rate, {})[str(row["ckpt_mode"])] = float(row["goodput"])

    gpu_hours = float(spec.cluster_size) * float(spec.hours)
    estimates: list[dict[str, Any]] = []
    for rate in sorted(by_rate):
        modes = by_rate[rate]
        if spec.baseline_mode not in modes or spec.improved_mode not in modes:
            continue
        baseline = modes[spec.baseline_mode]
        improved = modes[spec.improved_mode]
        delta = improved - baseline
        estimates.append(
            {
                "failure_rate": rate,
                "baseline_mode": spec.baseline_mode,
                "improved_mode": spec.improved_mode,
                "baseline_goodput": baseline,
                "improved_goodput": improved,
                "goodput_delta": delta,
                "gpu_hours": gpu_hours,
                "dollar_delta": compute_dollar_impact(
                    cluster_size=spec.cluster_size,
                    usd_per_gpu_hour=spec.usd_per_gpu_hour,
                    hours=spec.hours,
                    goodput_delta=delta,
                ),
            }
        )
    if not estimates:
        raise ValueError(
            f"need paired {spec.baseline_mode!r} and {spec.improved_mode!r} "
            "rows at the same failure_rate"
        )
    return estimates


def _format_cell(value: Any, field: str) -> str:
    if value is None:
        return ""
    if field == "failure_rate":
        return f"{float(value):g}"
    if field in {"baseline_goodput", "improved_goodput", "goodput_delta"}:
        return f"{float(value):.4f}"
    if field == "gpu_hours":
        return f"{float(value):.0f}"
    if field == "dollar_delta":
        return f"{float(value):,.2f}"
    return str(value)


def render_dollar_table(rows: list[dict[str, Any]], spec: DollarSpec) -> str:
    """Markdown estimate: disclaimer + pricing + one row per failure rate."""
    lines = [
        "# Dollar impact of a measured goodput delta",
        "",
        DISCLAIMER,
        "",
        f"- **GPU SKU:** {spec.gpu_sku}",
        f"- **Public list price:** ${spec.usd_per_gpu_hour:.2f} / GPU-hour",
        f"- **Price source:** {spec.price_source}",
        f"- **Price URL:** {spec.price_url}",
        f"- **Simulated cluster:** {spec.cluster_size:,} GPUs × {spec.hours:g} hours "
        f"(= {spec.cluster_size * spec.hours:,.0f} GPU-hours)",
        f"- **Comparison:** `{spec.baseline_mode}` (baseline) vs `{spec.improved_mode}` (improved)",
        "",
        "Positive `dollar_delta` means the improved checkpoint mode would buy more useful",
        "GPU-hours for the same wall clock (or waste fewer dollars) at this public rate.",
        "",
        "| " + " | ".join(DOLLAR_ROW_FIELDS) + " |",
        "| " + " | ".join("---" for _ in DOLLAR_ROW_FIELDS) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(key), key) for key in DOLLAR_ROW_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_dollar_table(
    rows: list[dict[str, Any]],
    spec: DollarSpec,
    *,
    comparison_path: Path | None,
) -> tuple[Path, Path, Path]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: row.get(k) for k in DOLLAR_ROW_FIELDS} for row in rows]
    payload = {
        "mode": "dollar",
        "name": spec.name,
        "disclaimer": DISCLAIMER,
        "gpu_sku": spec.gpu_sku,
        "usd_per_gpu_hour": spec.usd_per_gpu_hour,
        "price_source": spec.price_source,
        "price_url": spec.price_url,
        "cluster_size": spec.cluster_size,
        "hours": spec.hours,
        "baseline_mode": spec.baseline_mode,
        "improved_mode": spec.improved_mode,
        "comparison": str(comparison_path) if comparison_path is not None else None,
        "rows": slim,
    }
    json_path = output_dir / "dollar.json"
    csv_path = output_dir / "dollar.csv"
    table_path = output_dir / "table.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(DOLLAR_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(slim)
    table_path.write_text(render_dollar_table(slim, spec), encoding="utf-8")
    return json_path, csv_path, table_path


def _load_measured_rows(spec: DollarSpec) -> tuple[list[dict[str, Any]], Path | None]:
    if spec.comparison is not None and spec.comparison.is_file():
        return load_comparison(spec.comparison), spec.comparison
    if spec.sweep is not None:
        if not spec.sweep.is_file():
            raise FileNotFoundError(f"sweep config not found: {spec.sweep}")
        sweep_spec = load_sweep_yaml(spec.sweep)
        result: SweepResult = run_sweep(sweep_spec)
        comparison_path = result.json_path
        return result.rows, comparison_path
    missing = spec.comparison if spec.comparison is not None else Path("comparison.json")
    raise FileNotFoundError(
        f"comparison table not found: {missing}. Run the sweep first or set sweep: in YAML."
    )


def run_dollar(spec: DollarSpec) -> DollarResult:
    rows_in, comparison_path = _load_measured_rows(spec)
    estimates = estimate_from_comparison(rows_in, spec)
    json_path, csv_path, table_path = write_dollar_table(
        estimates, spec, comparison_path=comparison_path
    )
    return DollarResult(
        spec=spec,
        rows=estimates,
        json_path=json_path,
        csv_path=csv_path,
        table_path=table_path,
        comparison_path=comparison_path,
    )
