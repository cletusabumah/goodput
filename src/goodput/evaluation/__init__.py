"""Evaluation helpers — reports, sweep, plot, latency, dollar (1.7 / 2.2 / 2.3 / 2.5 / 3.3)."""

from goodput.evaluation.dollar import (
    DOLLAR_ROW_FIELDS,
    DollarResult,
    DollarSpec,
    compute_dollar_impact,
    load_dollar_yaml,
    render_dollar_table,
    run_dollar,
)
from goodput.evaluation.latency import (
    LATENCY_FIELDS,
    LatencyResult,
    LatencySpec,
    load_latency_yaml,
    render_latency_table,
    run_latency,
)
from goodput.evaluation.plot import (
    default_plot_path,
    load_comparison,
    plot_goodput_vs_failure_rate,
    series_from_comparison,
)
from goodput.evaluation.sweep import (
    COMPARISON_FIELDS,
    SweepResult,
    SweepSpec,
    kill_at_for_rate,
    load_sweep_yaml,
    run_sweep,
)
from goodput.metrics import (
    REQUIRED_REPORT_FIELDS,
    build_run_report,
    compute_goodput,
    emit_run_report,
)

__all__ = [
    "COMPARISON_FIELDS",
    "DOLLAR_ROW_FIELDS",
    "LATENCY_FIELDS",
    "REQUIRED_REPORT_FIELDS",
    "DollarResult",
    "DollarSpec",
    "LatencyResult",
    "LatencySpec",
    "SweepResult",
    "SweepSpec",
    "build_run_report",
    "compute_dollar_impact",
    "compute_goodput",
    "default_plot_path",
    "emit_run_report",
    "kill_at_for_rate",
    "load_comparison",
    "load_dollar_yaml",
    "load_latency_yaml",
    "load_sweep_yaml",
    "plot_goodput_vs_failure_rate",
    "render_dollar_table",
    "render_latency_table",
    "run_dollar",
    "run_latency",
    "run_sweep",
    "series_from_comparison",
]
