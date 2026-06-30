from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fcsl.core import Agent, Context, Goal, Observation
from fcsl.ontology import DomainOntology
from fcsl.relevance import ContextualSemanticRelevanceModel


@dataclass
class CoastalInstance:
    instance_id: str
    agent: Agent
    goal: Goal
    context: Context
    observations: list[Observation]
    true_decision_state: str


@dataclass
class CoastalMonitoringGenerator:
    ontology: DomainOntology
    seed: int = 7
    current_time: float = 10.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed) if self.rng is None else self.rng

    def generate_instance(
        self,
        instance_id: str,
        n_observations: int,
        agent_id: str = "coast_guard_operator",
        goal_id: str = "assess_swimming_safety",
        context_id: str = "coastal_safety_regulation",
    ) -> CoastalInstance:
        agent = self.ontology.make_agent(agent_id)
        goal = self.ontology.make_goal(goal_id, agent.id, tau=(self.current_time, self.current_time + 1.0))
        context = self.ontology.make_context(context_id, time=self.current_time)
        relevant_templates = self.ontology.templates_for_goal(goal_id)
        irrelevant_templates = self.ontology.irrelevant_templates_for_goal(goal_id)
        n_relevant = max(1, int(round(0.15 * n_observations)))
        n_irrelevant = max(0, n_observations - n_relevant)
        template_ids = list(self.rng.choice(relevant_templates, size=n_relevant, replace=True))
        if irrelevant_templates and n_irrelevant:
            template_ids += list(self.rng.choice(irrelevant_templates, size=n_irrelevant, replace=True))
        self.rng.shuffle(template_ids)
        observations = [self._make_observation(instance_id, idx, tid, context) for idx, tid in enumerate(template_ids)]
        true_decision = self._true_decision(goal, observations)
        return CoastalInstance(instance_id, agent, goal, context, observations, true_decision)

    def generate_instances(self, counts: list[int], instances_per_count: int = 80) -> list[CoastalInstance]:
        out: list[CoastalInstance] = []
        for n in counts:
            for i in range(instances_per_count):
                out.append(self.generate_instance(f"N{n:03d}_i{i:04d}", n))
        return out

    def _make_observation(self, instance_id: str, idx: int, template_id: str, context: Context) -> Observation:
        template = self.ontology.observation_templates[template_id]
        sources = [sid for sid, spec in self.ontology.sources.items() if template.get("modality") in spec.get("modalities", [])]
        if not sources:
            sources = list(self.ontology.sources.keys())
        source_id = str(self.rng.choice(sources))
        source_spec = self.ontology.sources[source_id]
        rel_low, rel_high = source_spec.get("reliability_range", [0.6, 0.95])
        reliability = float(self.rng.uniform(rel_low, rel_high))
        # Mixture of fresh and stale intervals.
        if self.rng.random() < 0.82:
            start = self.current_time - float(self.rng.uniform(0.0, 0.4))
            end = self.current_time + float(self.rng.uniform(0.4, 2.0))
        else:
            start = self.current_time - float(self.rng.uniform(8.0, 96.0))
            end = start + float(self.rng.uniform(0.2, 4.0))
        value = self._sample_value(template)
        return self.ontology.make_observation(
            observation_id=f"{instance_id}_o{idx:04d}",
            template_id=template_id,
            source_id=source_id,
            value=value,
            tau=(round(start, 3), round(end, 3)),
            reliability=round(reliability, 4),
            location=context.get("location"),
            extra={"agent_types": template.get("agent_types", ["coast_guard_operator"])},
        )

    def _sample_value(self, template: dict[str, Any]) -> Any:
        value_type = template.get("value_type", "categorical")
        if value_type == "numeric":
            low, high = template.get("value_range", [0.0, 1.0])
            return round(float(self.rng.uniform(low, high)), 4)
        values = template.get("values", [template.get("risk_level", "neutral")])
        return str(self.rng.choice(values))

    def _true_decision(self, goal: Goal, observations: list[Observation]) -> str:
        """Ground-truth decision induced by the valid goal-relevant evidence.

        The downstream decision module receives different selected subsets for
        different representations. The ground truth is instead computed from
        the valid observations that actually satisfy the swimming-safety goal.
        """
        valid_relevant = [
            obs
            for obs in observations
            if obs.get("requirement") in goal.get("requirements", []) and obs.tau[0] <= self.current_time <= obs.tau[1]
        ]
        if not valid_relevant:
            return "uncertain"
        core_requirements = {"water_quality", "weather", "wave", "current"}
        covered = {obs.get("requirement") for obs in valid_relevant}
        confidence = float(np.mean([obs.reliability for obs in valid_relevant]))
        risk = float(np.mean([self._risk_level_to_score(obs.get("risk_level", "neutral")) for obs in valid_relevant]))
        if len(covered & core_requirements) < 3 or confidence < 0.45:
            return "uncertain"
        if risk >= 0.56:
            return "unsafe"
        if risk <= 0.46:
            return "safe"
        return "uncertain"

    @staticmethod
    def _risk_level_to_score(risk_level: str) -> float:
        return {
            "critical": 0.95,
            "high": 0.80,
            "moderate": 0.55,
            "low": 0.25,
            "neutral": 0.50,
        }.get(str(risk_level), 0.50)

    def _hazard_evidence(self, obs: Observation) -> float:
        risk = str(obs.get("risk_level", "neutral"))
        if risk == "critical":
            return round(float(self.rng.uniform(0.90, 1.00)), 4)
        if risk == "high":
            return round(float(self.rng.uniform(0.68, 0.92)), 4)
        if risk == "moderate":
            return round(float(self.rng.uniform(0.45, 0.64)), 4)
        if risk == "low":
            return round(float(self.rng.uniform(0.10, 0.38)), 4)
        # Neutral physical/non-physical distractors may still perturb a simple
        # downstream module that consumes all forwarded observations. This is
        # intentional: FCSL evaluates whether semantic selection protects the
        # downstream decision stage from irrelevant clutter.
        return round(float(self.rng.uniform(0.0, 1.0)), 4)

    def to_dataframe(self, instances: list[CoastalInstance]) -> pd.DataFrame:
        model = ContextualSemanticRelevanceModel(self.ontology)
        rows: list[dict[str, Any]] = []
        for inst in instances:
            for obs in inst.observations:
                breakdown = model.compute(obs, inst.agent, inst.goal, inst.context, self.current_time)
                rows.append(
                    {
                        "instance_id": inst.instance_id,
                        "n_observations": len(inst.observations),
                        "agent_id": inst.agent.id,
                        "goal_id": inst.goal.id,
                        "context_id": inst.context.id,
                        "current_time": self.current_time,
                        "observation_id": obs.id,
                        "template_id": obs.get("template_id"),
                        "semantic_label": obs.get("semantic_label"),
                        "template_category": obs.get("category"),
                        "source_id": obs.get("source"),
                        "modality": obs.get("modality"),
                        "location": obs.get("location"),
                        "value": obs.get("value"),
                        "risk_level": obs.get("risk_level"),
                        "satisfies_requirement": obs.get("requirement"),
                        "physical_semantic_match": obs.get("physical_semantic_match"),
                        "tau_start": obs.tau[0],
                        "tau_end": obs.tau[1],
                        "is_temporally_valid": obs.tau[0] <= self.current_time <= obs.tau[1],
                        "source_reliability": obs.reliability,
                        "hazard_evidence": self._hazard_evidence(obs),
                        **breakdown.as_dict(),
                        "is_ground_truth_relevant": bool(
                            obs.get("requirement") in inst.goal.get("requirements", [])
                            and obs.tau[0] <= self.current_time <= obs.tau[1]
                            and breakdown.score >= 0.68
                        ),
                        "true_decision_state": inst.true_decision_state,
                    }
                )
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        return df

    def save_dataset(self, instances: list[CoastalInstance], path: str | Path) -> pd.DataFrame:
        df = self.to_dataframe(instances)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
