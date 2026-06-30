from __future__ import annotations

from dataclasses import dataclass, field

from fcsl.core import Agent, Context, Goal, Observation
from fcsl.ontology import DomainOntology
from fcsl.relevance.agent_relevance import agent_relevance
from fcsl.relevance.context_compatibility import context_compatibility
from fcsl.relevance.semantic_affinity import semantic_affinity
from fcsl.relevance.source_reliability import source_reliability
from fcsl.relevance.temporal_relevance import temporal_relevance


@dataclass(frozen=True)
class RelevanceBreakdown:
    sigma_sem: float
    sigma_ctx: float
    sigma_agent: float
    sigma_time: float
    sigma_rel: float
    score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sigma_sem": self.sigma_sem,
            "sigma_ctx": self.sigma_ctx,
            "sigma_agent": self.sigma_agent,
            "sigma_time": self.sigma_time,
            "sigma_rel": self.sigma_rel,
            "rel_score": self.score,
        }


@dataclass
class ContextualSemanticRelevanceModel:
    ontology: DomainOntology
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "sem": 0.30,
            "ctx": 0.20,
            "agent": 0.15,
            "time": 0.20,
            "rel": 0.15,
        }
    )
    lambda_time: float = 0.05

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Relevance weights must have positive sum.")
        self.weights = {k: float(v) / total for k, v in self.weights.items()}

    def compute(self, observation: Observation, agent: Agent, goal: Goal, context: Context, t: float) -> RelevanceBreakdown:
        sem = semantic_affinity(observation, goal, self.ontology)
        ctx = context_compatibility(observation, context)
        ag = agent_relevance(observation, agent)
        ti = temporal_relevance(observation, t, self.lambda_time)
        rel = source_reliability(observation)
        score = (
            self.weights["sem"] * sem
            + self.weights["ctx"] * ctx
            + self.weights["agent"] * ag
            + self.weights["time"] * ti
            + self.weights["rel"] * rel
        )
        return RelevanceBreakdown(sem, ctx, ag, ti, rel, float(score))

    def is_relevant(self, observation: Observation, agent: Agent, goal: Goal, context: Context, t: float, epsilon: float) -> bool:
        requirement = observation.get("requirement")
        satisfies_goal_requirement = requirement in goal.get("requirements", [])
        return satisfies_goal_requirement and self.compute(observation, agent, goal, context, t).score >= epsilon
