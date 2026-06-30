from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fcsl.state import CognitiveSemanticState


Rule = Callable[[CognitiveSemanticState], None]


@dataclass
class RuleEngine:
    rules: list[Rule] = field(default_factory=list)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def run(self, state: CognitiveSemanticState) -> CognitiveSemanticState:
        for rule in self.rules:
            rule(state)
        return state
