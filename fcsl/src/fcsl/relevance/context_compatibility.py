from __future__ import annotations

from fcsl.core import Context, Observation


def context_compatibility(observation: Observation, context: Context) -> float:
    if observation.get("location") == context.get("location"):
        location_bonus = 1.0
    else:
        location_bonus = 0.75 if observation.get("location") in str(context.get("location")) else 0.5
    return max(0.0, min(1.0, float(observation.get("context_affinity", 0.0)) * location_bonus))
