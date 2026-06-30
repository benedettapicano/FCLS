from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import Action, Agent, Belief, EnvironmentEntity
from fcsl.ontology import DomainOntology
from fcsl.operators.common import make_statement
from fcsl.operators.construct_knowledge import ContextualKnowledge
from fcsl.state import CognitiveSemanticState


@dataclass
class GroundAction:
    

    ontology: DomainOntology

    def __call__(self, knowledge: ContextualKnowledge, agent: Agent, state: CognitiveSemanticState, t: float) -> list[Action]:
        beliefs = [state.primitives[bid] for bid in knowledge.belief_ids if bid in state.primitives]
        if not beliefs:
            return []
        belief = max(beliefs, key=lambda b: float(b.get("confidence", 0.0)))
        decision_state = belief.get("decision_state", "uncertain")
        action_ids = self.ontology.decision_rules.get("actions_by_state", {}).get(decision_state, ["request_additional_sampling"])
        actions: list[Action] = []
        coastal_area = EnvironmentEntity(
            id="coastal_area:target",
            type="coastal_area",
            attributes={"role": "action_target"},
            location=agent.get("location"),
            tau=(t, t + 1.0),
        )
        risk = EnvironmentEntity(
            id=f"risk:{decision_state}",
            type="risk_state",
            attributes={"decision_state": decision_state},
            location=agent.get("location"),
            tau=(t, t + 1.0),
        )
        for action_id in action_ids:
            action = self.ontology.make_action(action_id, target=agent.id, tau=(t, t + 1.0))
            actions.append(action)
            state.add_primitive(action)
            state.add_statements(
                [
                    make_statement(belief, "enables", action, t, float(belief.get("confidence", 0.7)), belief.id, "GroundAction", "decision_rule"),
                    make_statement(action, "recommends", agent, t, 0.95, action.id, "GroundAction", "decision_rule"),
                    make_statement(action, "notifies", agent, t, 0.95, action.id, "GroundAction", "decision_rule"),
                    make_statement(action, "actsOn", coastal_area, t, 0.90, action.id, "GroundAction", "target_binding"),
                    make_statement(action, "mitigates", risk, t, 0.90, action.id, "GroundAction", "risk_rule"),
                ]
            )
        return actions
