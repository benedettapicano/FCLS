from __future__ import annotations

from dataclasses import dataclass

from fcsl.core import RelationSchema
from fcsl.ontology.domain_ontology import DomainOntology


@dataclass
class OntologyValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_domain_ontology(ontology: DomainOntology, relation_schema: RelationSchema) -> OntologyValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    for goal_id, goal in ontology.goals.items():
        for req in goal.get("requirements", []):
            if req not in ontology.requirements:
                errors.append(f"Goal {goal_id} references unknown requirement {req}.")
        for action in goal.get("admissible_actions", []):
            if action not in ontology.actions:
                errors.append(f"Goal {goal_id} references unknown action {action}.")
    for template_id, template in ontology.observation_templates.items():
        req = template.get("requirement")
        if req is not None and req not in ontology.requirements:
            warnings.append(f"Template {template_id} references non-goal requirement {req}.")
    if not relation_schema.relations:
        errors.append("Relation schema is empty.")
    return OntologyValidationReport(ok=not errors, errors=errors, warnings=warnings)
