from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from fcsl.core import Agent, Belief, Goal, Observation
from fcsl.ontology import DomainOntology
from fcsl.operators.common import make_statement
from fcsl.state import CognitiveSemanticState


@dataclass
class UpdateBelief:
   

    ontology: DomainOntology

    def __call__(
        self,
        selected_observations: list[Observation],
        agent: Agent,
        goal: Goal,
        state: CognitiveSemanticState,
        t: float,
    ) -> list[Belief]:
        required = set(goal.get("requirements", []))
        covered = {obs.get("requirement") for obs in selected_observations if obs.get("requirement")}
        high_risk = [obs for obs in selected_observations if obs.get("risk_level") in {"high", "critical"}]
        low_conf = [obs for obs in selected_observations if obs.reliability < 0.55]

        if high_risk:
            decision_state = "unsafe"
            polarity = "negative"
            content = "swimming activity is unsafe"
        elif required - covered or low_conf:
            decision_state = "uncertain"
            polarity = "uncertain"
            content = "swimming safety is uncertain"
        else:
            decision_state = "safe"
            polarity = "positive"
            content = "swimming activity is safe"

        confidence = mean([obs.reliability for obs in selected_observations]) if selected_observations else 0.35
        if high_risk:
            confidence = max(confidence, mean([obs.reliability for obs in high_risk]))
        belief = Belief(
            id=f"belief:{goal.id}:{decision_state}",
            content=content,
            source="UpdateBelief",
            confidence=float(confidence),
            tau=(t, t + 1.0),
            attributes={
                "topic": goal.id,
                "polarity": polarity,
                "decision_state": decision_state,
                "covered_requirements": sorted(x for x in covered if x),
                "missing_requirements": sorted(required - covered),
                "evidence_ids": [obs.id for obs in selected_observations],
            },
        )
        state.add_primitive(belief)
        statements = [
            make_statement(agent, "believes", belief, t, confidence, agent.id, "UpdateBelief", "evidence_aggregation"),
            make_statement(belief, "supports", goal, t, confidence, belief.id, "UpdateBelief", "belief_support"),
        ]
        for obs in selected_observations:
            statements.append(make_statement(belief, "derivedFrom", obs, t, min(confidence, obs.reliability), obs.id, "UpdateBelief", "evidence_trace"))
        state.add_statements(statements)
        return [belief]
