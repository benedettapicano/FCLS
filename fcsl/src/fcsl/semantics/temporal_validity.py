"""Temporal validity and freshness functions."""
from __future__ import annotations

import math

from fcsl.core import Observation, TemporalInterval


def temporal_freshness(interval: TemporalInterval, t: float, lambda_time: float = 0.05) -> float:
    """Return sigma_time using an exponential decay outside the validity interval."""
    if interval.contains(t):
        return 1.0
    return float(math.exp(-lambda_time * interval.distance(t)))


def observation_interval(observation: Observation) -> TemporalInterval:
    start, end = observation.tau
    return TemporalInterval(float(start), float(end))
