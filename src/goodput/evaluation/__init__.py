"""Evaluation helpers — reports, sweep, plot (tickets 1.7 / 2.2 / 2.3)."""

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
    "REQUIRED_REPORT_FIELDS",
    "SweepResult",
    "SweepSpec",
    "build_run_report",
    "compute_goodput",
    "default_plot_path",
    "emit_run_report",
    "kill_at_for_rate",
    "load_comparison",
    "load_sweep_yaml",
    "plot_goodput_vs_failure_rate",
    "run_sweep",
    "series_from_comparison",
]
