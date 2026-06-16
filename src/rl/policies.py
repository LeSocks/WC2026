from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.engine.tactics import AttackStyle, PressStyle
from src.rl.actions import ActionSpace, ActionType, DEFENSIVE_ACTIONS, POSSESSION_ACTIONS
from src.rl.observations import AgentObservation


class Policy(Protocol):
    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        ...


@dataclass(frozen=True)
class RandomPolicy:
    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        return action_space.sample(rng, has_possession=observation.has_possession)


@dataclass(frozen=True)
class ConstantPolicy:
    action: ActionType

    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        return action_space.validate(self.action)


@dataclass(frozen=True)
class TacticalPresetPolicy:
    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        if not observation.has_possession:
            return self._defensive_action(observation, rng)

        if observation.current_zone == "att_center":
            if observation.role_group == "attack" and rng.random() < 0.42:
                return ActionType.SHOOT
            if observation.attack_style == AttackStyle.POSSESSION.value:
                return ActionType.SAFE_PASS
            return ActionType.SHOOT if rng.random() < 0.30 else ActionType.PROGRESSIVE_PASS

        if observation.current_zone.startswith("att_"):
            if observation.attack_style == AttackStyle.DIRECT.value:
                return ActionType.PROGRESSIVE_PASS if rng.random() < 0.58 else ActionType.SHOOT
            return ActionType.DRIBBLE if rng.random() < 0.35 else ActionType.SAFE_PASS

        if observation.current_zone.startswith("def_"):
            return ActionType.CLEAR if observation.nearest_zone_pressure > 0.72 else ActionType.SAFE_PASS

        if observation.attack_style == AttackStyle.COUNTER_ATTACK.value:
            return ActionType.DRIBBLE if rng.random() < 0.45 else ActionType.PROGRESSIVE_PASS
        if observation.attack_style == AttackStyle.POSSESSION.value:
            return ActionType.SAFE_PASS if rng.random() < 0.62 else ActionType.SWITCH_PLAY
        return ActionType.PROGRESSIVE_PASS

    def _defensive_action(self, observation: AgentObservation, rng: np.random.Generator) -> ActionType:
        if observation.press_style in {PressStyle.HIGH_PRESS.value, PressStyle.COUNTER_PRESS.value}:
            return ActionType.PRESS if rng.random() < 0.70 else ActionType.MARK
        if observation.press_style == PressStyle.LOW_BLOCK.value:
            return ActionType.DROP if rng.random() < 0.68 else ActionType.MARK
        return ActionType.MARK if rng.random() < 0.55 else ActionType.PRESS


@dataclass
class EpsilonGreedyTabularPolicy:
    q_values: dict[tuple[object, ...], dict[ActionType, float]] = field(default_factory=dict)
    epsilon: float = 0.10

    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        available = POSSESSION_ACTIONS if observation.has_possession else DEFENSIVE_ACTIONS
        if rng.random() < self.epsilon:
            return available[int(rng.integers(0, len(available)))]

        state_values = self.q_values.get(observation.state_key(), {})
        if not state_values:
            return TacticalPresetPolicy().select_action(observation, action_space, rng)
        return max(available, key=lambda action: state_values.get(action, 0.0))


@dataclass
class RoleGroupPolicy:
    role_policies: dict[str, EpsilonGreedyTabularPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for role in ("defense", "midfield", "attack"):
            self.role_policies.setdefault(role, EpsilonGreedyTabularPolicy())

    def select_action(
        self,
        observation: AgentObservation,
        action_space: ActionSpace,
        rng: np.random.Generator,
    ) -> ActionType:
        return self.policy_for_role(observation.role_group).select_action(observation, action_space, rng)

    def policy_for_role(self, role_group: str) -> EpsilonGreedyTabularPolicy:
        canonical = canonical_role_group(role_group)
        return self.role_policies[canonical]

    @property
    def q_state_count(self) -> int:
        return sum(len(policy.q_values) for policy in self.role_policies.values())

    @property
    def epsilon(self) -> float:
        values = [policy.epsilon for policy in self.role_policies.values()]
        return float(sum(values) / len(values))

    def set_epsilon(self, epsilon: float) -> None:
        for policy in self.role_policies.values():
            policy.epsilon = epsilon


def canonical_role_group(role_group: str) -> str:
    return "defense" if role_group == "goalkeeper" else role_group
