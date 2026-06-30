"""Conflict detection for FCSL statements and beliefs."""
from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import FCSLStatement


@dataclass(frozen=True)
class Conflict:
    left: FCSLStatement
    right: FCSLStatement
    reason: str


def statements_conflict(left: FCSLStatement, right: FCSLStatement) -> bool:
    """Detect explicit and simple semantic conflicts.

    The implementation is intentionally conservative: FCSL represents conflicts,
    while applications may plug in richer domain-specific conflict detectors.
    """
    if left.relation == "conflictsWith" and left.tail.id == right.head.id:
        return True
    if right.relation == "conflictsWith" and right.tail.id == left.head.id:
        return True
    if left.relation == "contradicts" and left.tail.id == right.tail.id:
        return True
    if right.relation == "contradicts" and right.tail.id == left.tail.id:
        return True
    if left.head.kind == right.head.kind == "Belief":
        return _beliefs_conflict(left.head, right.head)
    return False


def _beliefs_conflict(left_belief, right_belief) -> bool:
    left_topic = left_belief.get("topic") or left_belief.get("content")
    right_topic = right_belief.get("topic") or right_belief.get("content")
    left_polarity = left_belief.get("polarity")
    right_polarity = right_belief.get("polarity")
    return bool(left_topic == right_topic and left_polarity and right_polarity and left_polarity != right_polarity)


def conflict_set(statements: list[FCSLStatement]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for i, left in enumerate(statements):
        for right in statements[i + 1 :]:
            if statements_conflict(left, right):
                conflicts.append(Conflict(left, right, "explicit_or_semantic_conflict"))
    return conflicts
