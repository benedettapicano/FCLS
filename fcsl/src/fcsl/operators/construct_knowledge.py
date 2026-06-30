from __future__ import annotations

from dataclasses import dataclass, field

from fcsl.core import Agent, Context, FCSLStatement, Goal
from fcsl.reasoning import ClosureEngine
from fcsl.state import CognitiveSemanticState


@dataclass
class ContextualKnowledge:
    agent_id: str
    goal_id: str
    context_id: str
    statements: list[FCSLStatement] = field(default_factory=list)

    @property
    def observation_ids(self) -> set[str]:
        return {s.head.id for s in self.statements if s.head.kind == "Observation"} | {s.tail.id for s in self.statements if s.tail.kind == "Observation"}

    @property
    def belief_ids(self) -> set[str]:
        return {s.head.id for s in self.statements if s.head.kind == "Belief"} | {s.tail.id for s in self.statements if s.tail.kind == "Belief"}


@dataclass
class ConstructKnowledge:
    

    closure_engine: ClosureEngine

    def __call__(self, agent: Agent, goal: Goal, context: Context, state: CognitiveSemanticState) -> ContextualKnowledge:
        closure = self.closure_engine.close(state, seeds=[agent.id, goal.id, context.id])
        return ContextualKnowledge(agent_id=agent.id, goal_id=goal.id, context_id=context.id, statements=closure)
