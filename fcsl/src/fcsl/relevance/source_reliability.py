from __future__ import annotations

from fcsl.core import Observation


def source_reliability(observation: Observation) -> float:
    return max(0.0, min(1.0, float(observation.reliability)))
