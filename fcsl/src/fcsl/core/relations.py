from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .primitives import BasePrimitive, PRIMITIVE_KINDS
from .statements import FCSLStatement


class RelationSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class RelationSpec:
    name: str
    family: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    description: str = ""

    def accepts(self, head: BasePrimitive, tail: BasePrimitive) -> bool:
        return _kind_matches(head.kind, self.domain) and _kind_matches(tail.kind, self.range)


def _kind_matches(kind: str, allowed: Iterable[str]) -> bool:
    allowed_set = set(allowed)
    return "Primitive" in allowed_set or kind in allowed_set


class RelationSchema:
    """Container for the family decomposition R_AG, R_GO, R_OC, R_B, R_AM, R_X."""

    def __init__(self, specs: dict[str, RelationSpec]) -> None:
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RelationSchema":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RelationSchema":
        specs: dict[str, RelationSpec] = {}
        for family, relations in payload.get("relation_families", {}).items():
            for name, spec in relations.items():
                domain = spec.get("domain", "Primitive")
                range_ = spec.get("range", "Primitive")
                if isinstance(domain, str):
                    domain = [domain]
                if isinstance(range_, str):
                    range_ = [range_]
                specs[name] = RelationSpec(
                    name=name,
                    family=family,
                    domain=tuple(domain),
                    range=tuple(range_),
                    description=spec.get("description", ""),
                )
        return cls(specs)

    @property
    def relations(self) -> set[str]:
        return set(self._specs)

    @property
    def families(self) -> dict[str, list[str]]:
        fam: dict[str, list[str]] = {}
        for spec in self._specs.values():
            fam.setdefault(spec.family, []).append(spec.name)
        return {k: sorted(v) for k, v in fam.items()}

    def get(self, relation: str) -> RelationSpec:
        if relation not in self._specs:
            raise RelationSchemaError(f"Unknown FCSL relation: {relation!r}.")
        return self._specs[relation]

    def validate(self, statement: FCSLStatement) -> None:
        if statement.head.kind not in PRIMITIVE_KINDS:
            raise RelationSchemaError(f"Unknown primitive kind for head: {statement.head.kind}.")
        if statement.tail.kind not in PRIMITIVE_KINDS:
            raise RelationSchemaError(f"Unknown primitive kind for tail: {statement.tail.kind}.")
        spec = self.get(statement.relation)
        if not spec.accepts(statement.head, statement.tail):
            raise RelationSchemaError(
                f"Relation {statement.relation!r} expects {spec.domain}->{spec.range}, "
                f"got {statement.head.kind}->{statement.tail.kind}."
            )

    def is_valid(self, statement: FCSLStatement) -> bool:
        try:
            self.validate(statement)
        except RelationSchemaError:
            return False
        return True

    def coverage(self, generated_relations: Iterable[str]) -> dict[str, Any]:
        generated = set(generated_relations)
        out: dict[str, Any] = {}
        for family, rels in self.families.items():
            rel_set = set(rels)
            out[family] = {
                "generated": sorted(rel_set & generated),
                "schema_only": sorted(rel_set - generated),
                "coverage": len(rel_set & generated) / max(len(rel_set), 1),
            }
        return out
