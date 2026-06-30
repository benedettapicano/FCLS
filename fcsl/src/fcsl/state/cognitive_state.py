"""Dynamic cognitive semantic state for FCSL."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from fcsl.core import BasePrimitive, FCSLStatement, RelationSchema, TruthStatus


@dataclass
class CognitiveSemanticState:
    """Dynamic state C_FCSL(t)=(A,G,O,B,X,W,M,N,Phi)."""

    relation_schema: RelationSchema
    time: float = 0.0
    kappa_min: float = 0.5
    primitives: dict[str, BasePrimitive] = field(default_factory=dict)
    statements: list[FCSLStatement] = field(default_factory=list)

    def add_primitive(self, primitive: BasePrimitive) -> BasePrimitive:
        self.primitives[primitive.id] = primitive
        return primitive

    def add_primitives(self, primitives: Iterable[BasePrimitive]) -> None:
        for primitive in primitives:
            self.add_primitive(primitive)

    def get(self, primitive_id: str) -> BasePrimitive:
        return self.primitives[primitive_id]

    def by_kind(self, kind: str) -> list[BasePrimitive]:
        return [p for p in self.primitives.values() if p.kind == kind]

    @property
    def agents(self) -> list[BasePrimitive]:
        return self.by_kind("Agent")

    @property
    def goals(self) -> list[BasePrimitive]:
        return self.by_kind("Goal")

    @property
    def observations(self) -> list[BasePrimitive]:
        return self.by_kind("Observation")

    @property
    def beliefs(self) -> list[BasePrimitive]:
        return self.by_kind("Belief")

    @property
    def contexts(self) -> list[BasePrimitive]:
        return self.by_kind("Context")

    @property
    def actions(self) -> list[BasePrimitive]:
        return self.by_kind("Action")

    @property
    def requirements(self) -> list[BasePrimitive]:
        return self.by_kind("Requirement")

    @property
    def environments(self) -> list[BasePrimitive]:
        return self.by_kind("EnvironmentEntity")

    def add_statement(self, statement: FCSLStatement, validate: bool = True) -> FCSLStatement:
        if validate:
            self.relation_schema.validate(statement)
        self.add_primitive(statement.head)
        self.add_primitive(statement.tail)
        if not self.has_statement(statement):
            self.statements.append(statement)
        return statement

    def add_statements(self, statements: Iterable[FCSLStatement], validate: bool = True) -> None:
        for statement in statements:
            self.add_statement(statement, validate=validate)

    def has_statement(self, statement: FCSLStatement) -> bool:
        return any(existing.id == statement.id for existing in self.statements)

    def update(self, delta_statements: Iterable[FCSLStatement]) -> "CognitiveSemanticState":
        self.add_statements(delta_statements)
        return self

    def valid_statements(self, t: float | None = None, require_true: bool = False) -> list[FCSLStatement]:
        time = self.time if t is None else t
        valid: list[FCSLStatement] = []
        for statement in self.statements:
            if statement.tau.contains(time) and statement.confidence >= self.kappa_min:
                if require_true and statement.truth_status != TruthStatus.TRUE:
                    continue
                valid.append(statement)
        return valid

    def expired_statements(self, t: float | None = None) -> list[FCSLStatement]:
        time = self.time if t is None else t
        return [s for s in self.statements if not s.tau.contains(time)]

    def relation_instances(self, relation: str) -> list[FCSLStatement]:
        return [s for s in self.statements if s.relation == relation]

    def directly_related(self, primitive_id: str) -> list[FCSLStatement]:
        return [s for s in self.statements if s.head.id == primitive_id or s.tail.id == primitive_id]

    def related_component(self, primitive_ids: Iterable[str], valid_only: bool = True) -> list[FCSLStatement]:
        """Return statements connected to at least one primitive through the FCSL graph."""
        seeds = set(primitive_ids)
        queue: deque[str] = deque(seeds)
        seen_primitives = set(seeds)
        seen_statements: dict[str, FCSLStatement] = {}
        candidates = self.valid_statements() if valid_only else self.statements
        adjacency: dict[str, list[FCSLStatement]] = {}
        for statement in candidates:
            adjacency.setdefault(statement.head.id, []).append(statement)
            adjacency.setdefault(statement.tail.id, []).append(statement)
        while queue:
            pid = queue.popleft()
            for statement in adjacency.get(pid, []):
                if statement.id not in seen_statements:
                    seen_statements[statement.id] = statement
                for nxt in (statement.head.id, statement.tail.id):
                    if nxt not in seen_primitives:
                        seen_primitives.add(nxt)
                        queue.append(nxt)
        return list(seen_statements.values())

    def snapshot(self) -> dict[str, int | float]:
        return {
            "time": self.time,
            "agents": len(self.agents),
            "goals": len(self.goals),
            "observations": len(self.observations),
            "beliefs": len(self.beliefs),
            "contexts": len(self.contexts),
            "environment_entities": len(self.environments),
            "actions": len(self.actions),
            "requirements": len(self.requirements),
            "statements": len(self.statements),
            "valid_statements": len(self.valid_statements()),
        }
