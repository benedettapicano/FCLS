from __future__ import annotations

from fcsl.core import BasePrimitive, FCSLStatement, Provenance, TemporalInterval


def make_statement(
    head: BasePrimitive,
    relation: str,
    tail: BasePrimitive,
    t: float,
    confidence: float,
    source_id: str,
    generated_by: str,
    method: str,
    horizon: float = 1.0,
    annotations: dict | None = None,
) -> FCSLStatement:
    return FCSLStatement(
        head=head,
        relation=relation,
        tail=tail,
        tau=TemporalInterval(t, t + horizon),
        confidence=confidence,
        provenance=Provenance(source_id=source_id, generated_by=generated_by, method=method),
        annotations=dict(annotations or {}),
    )
