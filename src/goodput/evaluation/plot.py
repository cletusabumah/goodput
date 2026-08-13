"""Goodput vs failure-rate plot from a sweep comparison table (ticket 2.3)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MODE_LABELS: dict[str, str] = {
    "naive": "naive (full dump)",
    "incremental": "incremental (fast ckpt)",
}


def load_comparison(path: str | Path) -> list[dict[str, Any]]:
    """Load ``comparison.json`` or ``comparison.csv`` written by the sweep runner."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"comparison table not found: {src}")
    if src.suffix.lower() == ".json":
        data = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"comparison JSON must be a list of rows: {src}")
        return [dict(row) for row in data]
    if src.suffix.lower() == ".csv":
        with src.open(encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    raise ValueError(f"comparison table must be .json or .csv: {src}")


def series_from_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    """Group (failure_rate, goodput) points by ckpt_mode, sorted by rate."""
    by_mode: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if "ckpt_mode" not in row or "failure_rate" not in row or "goodput" not in row:
            raise ValueError("comparison rows need ckpt_mode, failure_rate, and goodput")
        by_mode[str(row["ckpt_mode"])].append(
            (float(row["failure_rate"]), float(row["goodput"]))
        )
    if not by_mode:
        raise ValueError("comparison table is empty")
    preferred = ("naive", "incremental")
    ordered: dict[str, list[tuple[float, float]]] = {}
    for mode in preferred:
        if mode in by_mode:
            ordered[mode] = sorted(by_mode[mode])
    for mode, points in by_mode.items():
        if mode not in ordered:
            ordered[mode] = sorted(points)
    return ordered


def default_plot_path() -> Path:
    """Gitignored figure path from the master-plan Done when."""
    return Path("artifacts") / "plots" / "goodput_vs_failure_rate.png"


def plot_goodput_vs_failure_rate(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """
    Draw goodput vs injected failure rate, one line per checkpoint mode.

    Requires matplotlib (``pip install -e '.[viz]'``).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. Install with: pip install -e '.[viz]'"
        ) from exc

    series = series_from_comparison(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mode, points in series.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=MODE_LABELS.get(mode, mode))
    ax.set_xlabel("Failure rate (crashes per step)")
    ax.set_ylabel("Goodput")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Goodput vs injected failure rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
