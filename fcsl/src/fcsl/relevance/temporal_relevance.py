from __future__ import annotations

from fcsl.core import Observation, TemporalInterval
from fcsl.semantics import temporal_freshness


def temporal_relevance(observation: Observation, t: float, lambda_time: float = 0.05) -> float:
    start, end = observation.tau
    return temporal_freshness(TemporalInterval(float(start), float(end)), t, lambda_time=lambda_time)
