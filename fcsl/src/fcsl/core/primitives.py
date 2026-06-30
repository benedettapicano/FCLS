from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PrimitiveKind = str


@dataclass(frozen=True)
class BasePrimitive:
    

    id: str
    kind: PrimitiveKind
    label: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def with_attribute(self, key: str, value: Any) -> "BasePrimitive":
        attrs = dict(self.attributes)
        attrs[key] = value
        return BasePrimitive(id=self.id, kind=self.kind, label=self.label, attributes=attrs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class Agent(BasePrimitive):
    

    def __init__(
        self,
        id: str,
        type: str,
        capabilities: list[str] | None = None,
        preferences: Mapping[str, Any] | None = None,
        location: str | None = None,
        privacy: Mapping[str, Any] | None = None,
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "type": type,
                "capabilities": list(capabilities or []),
                "preferences": dict(preferences or {}),
                "location": location,
                "privacy": dict(privacy or {}),
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Agent")
        object.__setattr__(self, "label", label or id)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class Goal(BasePrimitive):
   

    def __init__(
        self,
        id: str,
        agent_id: str,
        description: str,
        priority: str | int = "normal",
        constraints: Mapping[str, Any] | None = None,
        tau: tuple[float, float] | None = None,
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "agent_id": agent_id,
                "description": description,
                "priority": priority,
                "constraints": dict(constraints or {}),
                "tau": tau,
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Goal")
        object.__setattr__(self, "label", label or description)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class Observation(BasePrimitive):
    

    def __init__(
        self,
        id: str,
        source: str,
        modality: str,
        value: Any,
        location: str,
        tau: tuple[float, float],
        reliability: float,
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "source": source,
                "modality": modality,
                "value": value,
                "location": location,
                "tau": tau,
                "reliability": float(reliability),
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Observation")
        object.__setattr__(self, "label", label or str(value))
        object.__setattr__(self, "attributes", attrs)

    @property
    def tau(self) -> tuple[float, float]:
        return self.attributes["tau"]

    @property
    def reliability(self) -> float:
        return float(self.attributes["reliability"])


@dataclass(frozen=True)
class Belief(BasePrimitive):
   

    def __init__(
        self,
        id: str,
        content: str,
        source: str,
        confidence: float,
        tau: tuple[float, float],
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "content": content,
                "source": source,
                "confidence": float(confidence),
                "tau": tau,
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Belief")
        object.__setattr__(self, "label", label or content)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class Context(BasePrimitive):
    

    def __init__(
        self,
        id: str,
        time: float,
        location: str,
        environment: Mapping[str, Any] | None = None,
        user: Mapping[str, Any] | None = None,
        operation: Mapping[str, Any] | None = None,
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "time": float(time),
                "location": location,
                "environment": dict(environment or {}),
                "user": dict(user or {}),
                "operation": dict(operation or {}),
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Context")
        object.__setattr__(self, "label", label or id)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class EnvironmentEntity(BasePrimitive):
    

    def __init__(
        self,
        id: str,
        type: str,
        attributes: Mapping[str, Any] | None = None,
        location: str | None = None,
        tau: tuple[float, float] | None = None,
        label: str = "",
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update({"type": type, "location": location, "tau": tau})
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "EnvironmentEntity")
        object.__setattr__(self, "label", label or id)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class Action(BasePrimitive):
    

    def __init__(
        self,
        id: str,
        type: str,
        target: str,
        preconditions: list[str] | None = None,
        effects: list[str] | None = None,
        tau: tuple[float, float] | None = None,
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update(
            {
                "type": type,
                "target": target,
                "preconditions": list(preconditions or []),
                "effects": list(effects or []),
                "tau": tau,
            }
        )
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Action")
        object.__setattr__(self, "label", label or id)
        object.__setattr__(self, "attributes", attrs)


@dataclass(frozen=True)
class Requirement(BasePrimitive):
   

    def __init__(
        self,
        id: str,
        description: str,
        category: str,
        priority: str | int = "normal",
        label: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs.update({"description": description, "category": category, "priority": priority})
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "kind", "Requirement")
        object.__setattr__(self, "label", label or description)
        object.__setattr__(self, "attributes", attrs)


PRIMITIVE_KINDS = {
    "Agent",
    "Goal",
    "Observation",
    "Belief",
    "Context",
    "EnvironmentEntity",
    "Action",
    "Requirement",
}


def primitive_from_dict(payload: Mapping[str, Any]) -> BasePrimitive:
    """Deserialize a primitive while preserving its concrete kind when possible."""
    kind = payload["kind"]
    attrs = dict(payload.get("attributes", {}))
    id_ = payload["id"]
    label = payload.get("label", id_)
    return BasePrimitive(id=id_, kind=kind, label=label, attributes=attrs)
