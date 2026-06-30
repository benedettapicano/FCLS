from __future__ import annotations

from fcsl.core import Goal, Observation
from fcsl.ontology import DomainOntology


def semantic_affinity(observation: Observation, goal: Goal, ontology: DomainOntology) -> float:
    requirement = observation.get("requirement")
    if requirement is None:
        return 0.0
    if requirement not in goal.get("requirements", []):
        return 0.0
    return float(observation.get("semantic_affinity", 0.0))
