"""Interpretation semantics I_t(phi)=(truth_status, confidence)."""
from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import FCSLStatement, RelationSchema, TruthStatus
from fcsl.semantics.conflict_detection import statements_conflict


@dataclass
class InterpretationEngine:
    relation_schema: RelationSchema
    kappa_min: float = 0.5

    def interpret(
        self,
        statement: FCSLStatement,
        all_statements: list[FCSLStatement],
        t: float,
    ) -> tuple[TruthStatus, float]:
        if not self.relation_schema.is_valid(statement):
            return (TruthStatus.FALSE, statement.confidence)
        if not statement.tau.contains(t):
            return (TruthStatus.FALSE, statement.confidence)
        if statement.confidence < self.kappa_min:
            return (TruthStatus.UNKNOWN, statement.confidence)
        for other in all_statements:
            if other.id == statement.id:
                continue
            if other.tau.contains(t) and other.confidence >= self.kappa_min and statements_conflict(statement, other):
                return (TruthStatus.UNKNOWN, statement.confidence)
        return (TruthStatus.TRUE, statement.confidence)

    def satisfies(self, statement: FCSLStatement, all_statements: list[FCSLStatement], t: float) -> bool:
        status, confidence = self.interpret(statement, all_statements, t)
        return status == TruthStatus.TRUE and confidence >= self.kappa_min

    def annotate(self, statements: list[FCSLStatement], t: float) -> list[FCSLStatement]:
        annotated: list[FCSLStatement] = []
        for statement in statements:
            status, _ = self.interpret(statement, statements, t)
            annotated.append(statement.with_truth(status))
        return annotated
