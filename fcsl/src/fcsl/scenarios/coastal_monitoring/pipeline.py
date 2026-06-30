from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fcsl.core import RelationSchema
from fcsl.ontology import DomainOntology
from fcsl.operators import Activate, Contextualize, ConstructKnowledge, GroundAction, Require, Select, UpdateBelief
from fcsl.reasoning import ClosureEngine
from fcsl.relevance import ContextualSemanticRelevanceModel
from fcsl.scenarios.coastal_monitoring.generator import CoastalInstance
from fcsl.semantics import InterpretationEngine
from fcsl.state import CognitiveSemanticState


@dataclass
class PipelineResult:
    state: CognitiveSemanticState
    selected_observations: list[Any]
    relevance_scores: dict[str, Any]
    beliefs: list[Any]
    actions: list[Any]
    knowledge: Any


@dataclass
class FCSLPipeline:
    relation_schema: RelationSchema
    ontology: DomainOntology
    epsilon: float = 0.68
    kappa_min: float = 0.5

    @classmethod
    def from_paths(cls, relation_schema_path: str | Path, ontology_path: str | Path, epsilon: float = 0.68) -> "FCSLPipeline":
        return cls(RelationSchema.from_yaml(relation_schema_path), DomainOntology.from_yaml(ontology_path), epsilon=epsilon)

    def run_query(self, instance: CoastalInstance, query: str = "Is it safe to authorize swimming activities in this coastal area?") -> PipelineResult:
        state = CognitiveSemanticState(self.relation_schema, time=instance.context.get("time"), kappa_min=self.kappa_min)
        state.add_primitives([instance.agent, instance.context])
        state.add_primitives(instance.observations)

        activator = Activate(self.ontology)
        goals = activator.from_query(query, instance.agent, state, tau=(state.time, state.time + 1.0))
        goal = goals[0] if goals else instance.goal
        state.add_primitive(goal)

        requirements = Require(self.ontology)(instance.agent, goal, instance.context, state)
        relevance_model = ContextualSemanticRelevanceModel(self.ontology)
        selector = Select(relevance_model, epsilon=self.epsilon)
        selected, scores = selector(instance.observations, instance.agent, goal, instance.context, state.time)
        Contextualize(self.ontology)(selected, instance.observations, instance.agent, goal, instance.context, requirements, state, state.time, scores)
        beliefs = UpdateBelief(self.ontology)(selected, instance.agent, goal, state, state.time)
        closure_engine = ClosureEngine()
        knowledge = ConstructKnowledge(closure_engine)(instance.agent, goal, instance.context, state)
        actions = GroundAction(self.ontology)(knowledge, instance.agent, state, state.time)
        interpreter = InterpretationEngine(self.relation_schema, kappa_min=self.kappa_min)
        state.statements = interpreter.annotate(state.statements, state.time)
        return PipelineResult(state, selected, scores, beliefs, actions, knowledge)

    def run_event(self, instance: CoastalInstance, event: dict[str, Any]) -> PipelineResult:
        # For this reference implementation, event-triggered activation maps the event to the same
        # FCSL operator chain as human-triggered cognition.
        state = CognitiveSemanticState(self.relation_schema, time=instance.context.get("time"), kappa_min=self.kappa_min)
        state.add_primitives([instance.agent, instance.context])
        state.add_primitives(instance.observations)
        goals = Activate(self.ontology).from_event(event, instance.agent, state, tau=(state.time, state.time + 1.0))
        if not goals:
            return self.run_query(instance)
        # Reuse query pipeline using the event-activated goal by setting it in the instance.
        instance.goal = goals[0]
        return self.run_query(instance, query=goals[0].get("description", ""))
