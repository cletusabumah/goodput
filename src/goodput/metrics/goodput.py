"""Goodput metric helpers (Phase 1 expands definitions)."""

from __future__ import annotations


def compute_goodput(useful_seconds: float, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be > 0")
    if useful_seconds < 0:
        raise ValueError("useful_seconds must be >= 0")
    return min(1.0, useful_seconds / wall_seconds)


def compute_wasted_gpu_hours(
    wall_seconds: float,
    useful_seconds: float,
    num_workers: int,
) -> float:
    """
    Proxy for wasted accelerator-hours: (wall − useful) × workers / 3600.

    On CPU-only runs this is still a useful relative cost signal for A/B
    checkpoint experiments (same worker count, same failure schedule).
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if wall_seconds < 0:
        raise ValueError("wall_seconds must be >= 0")
    if useful_seconds < 0:
        raise ValueError("useful_seconds must be >= 0")
    wasted_seconds = max(0.0, wall_seconds - useful_seconds)
    return (wasted_seconds * num_workers) / 3600.0
