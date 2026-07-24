"""Goodput metric helpers (Phase 1 expands definitions)."""

from __future__ import annotations


def compute_goodput(useful_seconds: float, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be > 0")
    if useful_seconds < 0:
        raise ValueError("useful_seconds must be >= 0")
    return min(1.0, useful_seconds / wall_seconds)
