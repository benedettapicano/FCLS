from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import Agent, Context, Goal, Observation
from fcsl.relevance import ContextualSemanticRelevanceModel, RelevanceBreakdown


@dataclass
class Select:
    

    relevance_model: ContextualSemanticRelevanceModel
    epsilon: float = 0.68

    def __call__(
        self,
        observations: list[Observation],
        agent: Agent,
        goal: Goal,
        context: Context,
        t: float,
    ) -> tuple[list[Observation], dict[str, RelevanceBreakdown]]:
        selected: list[Observation] = []
        scores: dict[str, RelevanceBreakdown] = {}
        for observation in observations:
            breakdown = self.relevance_model.compute(observation, agent, goal, context, t)
            scores[observation.id] = breakdown
            if self.relevance_model.is_relevant(observation, agent, goal, context, t, self.epsilon):
                selected.append(observation)
        return selected, scores
