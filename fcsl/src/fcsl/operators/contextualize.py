from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import Agent, BasePrimitive, Context, EnvironmentEntity, Goal, Observation, Requirement
from fcsl.ontology import DomainOntology
from fcsl.operators.common import make_statement
from fcsl.state import CognitiveSemanticState


@dataclass
class Contextualize:
   

    ontology: DomainOntology

    def __call__(
        self,
        selected_observations: list[Observation],
        all_observations: list[Observation],
        agent: Agent,
        goal: Goal,
        context: Context,
        requirements: list[Requirement],
        state: CognitiveSemanticState,
        t: float,
        relevance_scores: dict[str, object] | None = None,
    ):
        statements = []
        horizon = 1.0
        # Agent-goal-context structure.
        statements.append(make_statement(agent, "pursues", goal, t, 0.99, agent.id, "Activate", "query_mapping", horizon))
        statements.append(make_statement(agent, "locatedIn", context, t, 0.95, agent.id, "Contextualize", "context_binding", horizon))
        statements.append(make_statement(context, "constrains", goal, t, 0.90, context.id, "Contextualize", "context_rule", horizon))
        statements.append(make_statement(goal, "underCondition", context, t, 0.90, context.id, "Contextualize", "context_rule", horizon))

        for req in requirements:
            statements.append(make_statement(goal, "requires", req, t, 0.98, goal.id, "Require", "ontology_mapping", horizon))

        selected_ids = {obs.id for obs in selected_observations}
        for obs in all_observations:
            req_id = obs.get("requirement")
            req = next((r for r in requirements if r.id == req_id), None)
            source = self._source_entity(obs)
            reliability = EnvironmentEntity(
                id=f"reliability:{obs.id}",
                type="reliability_score",
                attributes={"score": obs.reliability},
                location=obs.get("location"),
                tau=obs.tau,
                label=f"rho={obs.reliability:.2f}",
            )
            time_entity = Context(
                id=f"time:{obs.id}",
                time=t,
                location=obs.get("location"),
                environment={"interval": obs.tau},
                label=f"validity interval for {obs.id}",
            )
            location_entity = EnvironmentEntity(
                id=f"env:{obs.get('location')}",
                type="coastal_area",
                attributes={"location": obs.get("location")},
                location=obs.get("location"),
                tau=obs.tau,
                label=str(obs.get("location")),
            )
            statements.extend(
                [
                    make_statement(obs, "observedBy", source, t, obs.reliability, obs.id, "Contextualize", "source_binding", horizon),
                    make_statement(obs, "observedAt", time_entity, t, obs.reliability, obs.id, "Contextualize", "time_binding", horizon),
                    make_statement(obs, "occursIn", context, t, obs.reliability, obs.id, "Contextualize", "context_binding", horizon),
                    make_statement(obs, "validUnder", context, t, obs.reliability, obs.id, "Contextualize", "context_rule", horizon),
                    make_statement(obs, "hasReliability", reliability, t, 1.0, obs.id, "Contextualize", "source_reliability", horizon),
                    make_statement(obs, "locatedIn", location_entity, t, obs.reliability, obs.id, "Contextualize", "spatial_binding", horizon),
                    make_statement(obs, "during", time_entity, t, obs.reliability, obs.id, "Contextualize", "temporal_binding", horizon),
                    make_statement(obs, "underCondition", context, t, obs.reliability, obs.id, "Contextualize", "context_rule", horizon),
                ]
            )
            if req is not None:
                statements.append(make_statement(obs, "satisfies", req, t, obs.get("semantic_affinity", 0.5), obs.id, "Contextualize", "ontology_mapping", horizon))
            if obs.id in selected_ids:
                score = None
                if relevance_scores and obs.id in relevance_scores:
                    score = getattr(relevance_scores[obs.id], "score", None)
                conf = float(score if score is not None else obs.reliability)
                statements.append(make_statement(obs, "relevantTo", goal, t, conf, obs.id, "Select", "contextual_semantic_relevance", horizon))
                statements.append(make_statement(obs, "supports", goal, t, conf, obs.id, "Select", "contextual_semantic_relevance", horizon))
        state.add_statements(statements)
        return statements

    def _source_entity(self, obs: Observation) -> EnvironmentEntity:
        source_id = obs.get("source")
        if source_id in self.ontology.sources:
            return self.ontology.make_source_entity(source_id)
        return EnvironmentEntity(id=source_id, type="observation_source", location=obs.get("location"), tau=obs.tau)
