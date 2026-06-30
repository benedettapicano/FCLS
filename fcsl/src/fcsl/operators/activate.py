from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fcsl.core import Agent, Goal
from fcsl.ontology import DomainOntology
from fcsl.state import CognitiveSemanticState


@dataclass
class Activate:
   

    ontology: DomainOntology

    def from_query(self, query: str, agent: Agent, state: CognitiveSemanticState, tau: tuple[float, float] = (0.0, 24.0)) -> list[Goal]:
        goals = [self.ontology.make_goal(gid, agent.id, tau=tau) for gid in self.ontology.goals_for_query(query)]
        for goal in goals:
            state.add_primitive(goal)
        return goals

    def from_event(self, event: dict[str, Any], agent: Agent, state: CognitiveSemanticState, tau: tuple[float, float] = (0.0, 24.0)) -> list[Goal]:
        goals = [self.ontology.make_goal(gid, agent.id, tau=tau) for gid in self.ontology.goals_for_event(event.get("event_type", ""))]
        for goal in goals:
            state.add_primitive(goal)
        return goals
