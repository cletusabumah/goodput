"""Gradient bit-flip fault helpers (ticket 3.2)."""

from goodput.faults.bitflip import (
    BITFLIP_RANK_IDLE,
    BITFLIP_STEP_IDLE,
    detect_gradient_outlier,
    flip_float32_bit,
)

__all__ = [
    "BITFLIP_RANK_IDLE",
    "BITFLIP_STEP_IDLE",
    "detect_gradient_outlier",
    "flip_float32_bit",
]
