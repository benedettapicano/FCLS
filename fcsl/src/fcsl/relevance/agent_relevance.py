from __future__ import annotations

from fcsl.core import Agent, Observation


def agent_relevance(observation: Observation, agent: Agent) -> float:
    agent_type = agent.get("type")
    preferred = observation.get("agent_types", [])
    base = float(observation.get("agent_affinity", 0.0))
    if preferred and agent_type not in preferred:
        return 0.5 * base
    return base
