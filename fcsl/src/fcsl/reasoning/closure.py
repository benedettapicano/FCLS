from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import FCSLStatement
from fcsl.operators.common import make_statement
from fcsl.state import CognitiveSemanticState


@dataclass
class ClosureEngine:
    """Apply admissible lightweight FCSL closure rules.

    This is not a full theorem prover. It is a domain-independent reference
    closure engine that makes the paper-level operator cl_Omega executable.
    """

    max_iterations: int = 5

    def close(self, state: CognitiveSemanticState, seeds: list[str] | None = None) -> list[FCSLStatement]:
        for _ in range(self.max_iterations):
            before = len(state.statements)
            self._apply_support_rule(state)
            self._apply_belief_support_rule(state)
            if len(state.statements) == before:
                break
        if seeds:
            return state.related_component(seeds, valid_only=True)
        return state.valid_statements()

    def _apply_support_rule(self, state: CognitiveSemanticState) -> None:
        # If obs satisfies req, goal requires req, and obs relevantTo goal -> obs supports goal.
        satisfies = state.relation_instances("satisfies")
        requires = state.relation_instances("requires")
        relevant = state.relation_instances("relevantTo")
        for sat in satisfies:
            obs, req = sat.head, sat.tail
            for req_stmt in requires:
                if req_stmt.tail.id != req.id:
                    continue
                goal = req_stmt.head
                if any(rel.head.id == obs.id and rel.tail.id == goal.id for rel in relevant):
                    stmt = make_statement(
                        obs,
                        "supports",
                        goal,
                        state.time,
                        min(sat.confidence, req_stmt.confidence),
                        obs.id,
                        "ClosureEngine",
                        "satisfies_requires_relevant_rule",
                    )
                    if not state.has_statement(stmt):
                        state.add_statement(stmt)

    def _apply_belief_support_rule(self, state: CognitiveSemanticState) -> None:
        # If belief derivedFrom obs and obs supports goal -> belief supports goal.
        derived = state.relation_instances("derivedFrom")
        supports = state.relation_instances("supports")
        for drv in derived:
            belief, obs = drv.head, drv.tail
            if belief.kind != "Belief" or obs.kind != "Observation":
                continue
            for sup in supports:
                if sup.head.id == obs.id and sup.tail.kind == "Goal":
                    stmt = make_statement(
                        belief,
                        "supports",
                        sup.tail,
                        state.time,
                        min(drv.confidence, sup.confidence),
                        belief.id,
                        "ClosureEngine",
                        "derived_evidence_support_rule",
                    )
                    if not state.has_statement(stmt):
                        state.add_statement(stmt)
