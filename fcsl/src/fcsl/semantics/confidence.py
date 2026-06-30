"""Confidence aggregation utilities."""
from __future__ import annotations

from statistics import mean


def mean_confidence(values: list[float], default: float = 0.0) -> float:
    return float(mean(values)) if values else default


def bounded_product(values: list[float], default: float = 1.0) -> float:
    if not values:
        return default
    out = 1.0
    for value in values:
        out *= max(0.0, min(1.0, float(value)))
    return out
