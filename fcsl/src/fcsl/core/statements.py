from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .primitives import BasePrimitive


class TruthStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemporalInterval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Invalid interval [{self.start}, {self.end}].")

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def distance(self, t: float) -> float:
        if self.contains(t):
            return 0.0
        return min(abs(t - self.start), abs(t - self.end))

    def as_tuple(self) -> tuple[float, float]:
        return (self.start, self.end)


@dataclass(frozen=True)
class Provenance:
    source_id: str
    generated_by: str
    method: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "generated_by": self.generated_by,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FCSLStatement:
    """A typed semantic assertion phi=<head, relation, tail, tau, kappa, mu>."""

    head: BasePrimitive
    relation: str
    tail: BasePrimitive
    tau: TemporalInterval
    confidence: float
    provenance: Provenance
    truth_status: TruthStatus = TruthStatus.UNKNOWN
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"Statement confidence must be in [0,1], got {self.confidence}.")

    @property
    def id(self) -> str:
        return f"{self.head.id}::{self.relation}::{self.tail.id}"

    def with_truth(self, status: TruthStatus) -> "FCSLStatement":
        return FCSLStatement(
            head=self.head,
            relation=self.relation,
            tail=self.tail,
            tau=self.tau,
            confidence=self.confidence,
            provenance=self.provenance,
            truth_status=status,
            annotations=dict(self.annotations),
        )

    def as_tuple(self) -> tuple[str, str, str, tuple[float, float], float, dict[str, Any]]:
        return (
            self.head.id,
            self.relation,
            self.tail.id,
            self.tau.as_tuple(),
            self.confidence,
            self.provenance.as_dict(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "head": self.head.as_dict(),
            "relation": self.relation,
            "tail": self.tail.as_dict(),
            "tau": self.tau.as_tuple(),
            "confidence": self.confidence,
            "provenance": self.provenance.as_dict(),
            "truth_status": self.truth_status.value,
            "annotations": dict(self.annotations),
        }
