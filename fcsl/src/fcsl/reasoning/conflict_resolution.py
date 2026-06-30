from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import FCSLStatement
from fcsl.semantics.conflict_detection import Conflict


@dataclass
class ConflictResolutionPolicy:
    w_confidence: float = 0.45
    w_freshness: float = 0.25
    w_reliability: float = 0.30

    def score(self, statement: FCSLStatement, t: float) -> float:
        freshness = 1.0 if statement.tau.contains(t) else 0.0
        reliability = float(statement.annotations.get("source_reliability", statement.confidence))
        return self.w_confidence * statement.confidence + self.w_freshness * freshness + self.w_reliability * reliability

    def resolve(self, conflict: Conflict, t: float) -> FCSLStatement:
        left_score = self.score(conflict.left, t)
        right_score = self.score(conflict.right, t)
        return conflict.left if left_score >= right_score else conflict.right
