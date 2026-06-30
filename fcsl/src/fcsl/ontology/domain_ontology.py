from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fcsl.core import (
    Action,
    Agent,
    BasePrimitive,
    Context,
    EnvironmentEntity,
    Goal,
    Observation,
    Requirement,
)


@dataclass
class DomainOntology:
    """Lightweight ontology used to instantiate the domain-independent FCSL core."""

    payload: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DomainOntology":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.payload.get(name, {}))

    @property
    def agents(self) -> dict[str, Any]:
        return self.section("agents")

    @property
    def goals(self) -> dict[str, Any]:
        return self.section("goals")

    @property
    def contexts(self) -> dict[str, Any]:
        return self.section("contexts")

    @property
    def requirements(self) -> dict[str, Any]:
        return self.section("requirements")

    @property
    def sources(self) -> dict[str, Any]:
        return self.section("sources")

    @property
    def observation_templates(self) -> dict[str, Any]:
        return self.section("observation_templates")

    @property
    def actions(self) -> dict[str, Any]:
        return self.section("actions")

    @property
    def environment_entities(self) -> dict[str, Any]:
        return self.section("environment_entities")

    @property
    def event_triggers(self) -> dict[str, Any]:
        return self.section("event_triggers")

    @property
    def decision_rules(self) -> dict[str, Any]:
        return self.section("decision_rules")

    def make_agent(self, agent_id: str) -> Agent:
        spec = self.agents[agent_id]
        return Agent(
            id=agent_id,
            type=spec.get("type", agent_id),
            capabilities=spec.get("capabilities", []),
            preferences=spec.get("preferences", {}),
            location=spec.get("location"),
            privacy=spec.get("privacy", {}),
            label=spec.get("label", agent_id),
            attributes={"ontology_id": agent_id},
        )

    def make_goal(self, goal_id: str, agent_id: str, tau: tuple[float, float] = (0.0, 24.0)) -> Goal:
        spec = self.goals[goal_id]
        return Goal(
            id=goal_id,
            agent_id=agent_id,
            description=spec.get("description", goal_id),
            priority=spec.get("priority", "normal"),
            constraints=spec.get("constraints", {}),
            tau=tau,
            label=spec.get("label", goal_id),
            attributes={
                "ontology_id": goal_id,
                "requirements": list(spec.get("requirements", [])),
                "admissible_actions": list(spec.get("admissible_actions", [])),
                "query_patterns": list(spec.get("query_patterns", [])),
                "event_patterns": list(spec.get("event_patterns", [])),
            },
        )

    def make_context(self, context_id: str, time: float = 10.0) -> Context:
        spec = self.contexts[context_id]
        return Context(
            id=context_id,
            time=time,
            location=spec.get("location", "unknown"),
            environment=spec.get("environment", {}),
            user=spec.get("user", {}),
            operation=spec.get("operation", {}),
            label=spec.get("label", context_id),
            attributes={"ontology_id": context_id},
        )

    def make_requirement(self, req_id: str) -> Requirement:
        spec = self.requirements[req_id]
        return Requirement(
            id=req_id,
            description=spec.get("description", req_id),
            category=spec.get("category", req_id),
            priority=spec.get("priority", "normal"),
            label=spec.get("label", req_id),
            attributes={"ontology_id": req_id, "labels": list(spec.get("labels", []))},
        )

    def make_source_entity(self, source_id: str) -> EnvironmentEntity:
        spec = self.sources[source_id]
        return EnvironmentEntity(
            id=source_id,
            type="observation_source",
            attributes={
                "source_type": spec.get("type", source_id),
                "modalities": list(spec.get("modalities", [])),
                "reliability_range": list(spec.get("reliability_range", [0.5, 1.0])),
            },
            location=spec.get("location", "unknown"),
            tau=(0.0, 24.0),
            label=spec.get("label", source_id),
        )

    def make_environment_entity(self, entity_id: str) -> EnvironmentEntity:
        spec = self.environment_entities[entity_id]
        return EnvironmentEntity(
            id=entity_id,
            type=spec.get("type", entity_id),
            attributes=spec.get("attributes", {}),
            location=spec.get("location"),
            tau=tuple(spec.get("tau", [0.0, 24.0])),
            label=spec.get("label", entity_id),
        )

    def make_action(self, action_id: str, target: str | None = None, tau: tuple[float, float] = (10.0, 11.0)) -> Action:
        spec = self.actions[action_id]
        return Action(
            id=action_id,
            type=spec.get("type", action_id),
            target=target or spec.get("target", "agent"),
            preconditions=spec.get("preconditions", []),
            effects=spec.get("effects", []),
            tau=tau,
            label=spec.get("label", action_id),
            attributes={
                "ontology_id": action_id,
                "decision_state": spec.get("decision_state"),
                "priority": spec.get("priority", "normal"),
            },
        )

    def make_observation(
        self,
        observation_id: str,
        template_id: str,
        source_id: str,
        value: Any,
        tau: tuple[float, float],
        reliability: float,
        location: str,
        extra: dict[str, Any] | None = None,
    ) -> Observation:
        template = self.observation_templates[template_id]
        attrs = dict(extra or {})
        attrs.update(
            {
                "template_id": template_id,
                "semantic_label": template.get("semantic_label", template_id),
                "requirement": template.get("requirement"),
                "category": template.get("category", "unknown"),
                "physical_semantic_match": bool(template.get("physical_semantic_match", True)),
                "risk_direction": template.get("risk_direction", "none"),
                "risk_level": template.get("risk_level", "neutral"),
                "semantic_affinity": float(template.get("semantic_affinity", 0.0)),
                "context_affinity": float(template.get("context_affinity", 0.0)),
                "agent_affinity": float(template.get("agent_affinity", 0.0)),
            }
        )
        return Observation(
            id=observation_id,
            source=source_id,
            modality=template.get("modality", "unknown"),
            value=value,
            location=location,
            tau=tau,
            reliability=reliability,
            label=template.get("semantic_label", template_id),
            attributes=attrs,
        )

    def goal_requirements(self, goal_id: str) -> list[str]:
        return list(self.goals[goal_id].get("requirements", []))

    def observation_requirement(self, template_id: str) -> str | None:
        return self.observation_templates[template_id].get("requirement")

    def templates_for_goal(self, goal_id: str) -> list[str]:
        reqs = set(self.goal_requirements(goal_id))
        return [tid for tid, spec in self.observation_templates.items() if spec.get("requirement") in reqs]

    def irrelevant_templates_for_goal(self, goal_id: str) -> list[str]:
        reqs = set(self.goal_requirements(goal_id))
        return [tid for tid, spec in self.observation_templates.items() if spec.get("requirement") not in reqs]

    def goals_for_query(self, query: str) -> list[str]:
        q = query.lower()
        matched: list[str] = []
        for gid, spec in self.goals.items():
            for pattern in spec.get("query_patterns", []):
                if pattern.lower() in q:
                    matched.append(gid)
                    break
        return matched

    def goals_for_event(self, event_type: str) -> list[str]:
        matched: list[str] = []
        for gid, spec in self.goals.items():
            if event_type in spec.get("event_patterns", []):
                matched.append(gid)
        for _, trigger in self.event_triggers.items():
            if trigger.get("event_type") == event_type:
                target = trigger.get("activates_goal")
                if target and target not in matched:
                    matched.append(target)
        return matched
