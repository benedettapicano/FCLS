from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import Agent, Context, Goal, Requirement
from fcsl.ontology import DomainOntology
from fcsl.state import CognitiveSemanticState


@dataclass
class Require:
    

    ontology: DomainOntology

    def __call__(self, agent: Agent, goal: Goal, context: Context, state: CognitiveSemanticState) -> list[Requirement]:
        requirements = [self.ontology.make_requirement(req_id) for req_id in goal.get("requirements", [])]
        for requirement in requirements:
            state.add_primitive(requirement)
        return requirements
