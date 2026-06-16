from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ActionType(Enum):
    HOLD = "hold"
    SAFE_PASS = "safe_pass"
    PROGRESSIVE_PASS = "progressive_pass"
    SWITCH_PLAY = "switch_play"
    DRIBBLE = "dribble"
    SHOOT = "shoot"
    PRESS = "press"
    DROP = "drop"
    MARK = "mark"
    CLEAR = "clear"


POSSESSION_ACTIONS: tuple[ActionType, ...] = (
    ActionType.HOLD,
    ActionType.SAFE_PASS,
    ActionType.PROGRESSIVE_PASS,
    ActionType.SWITCH_PLAY,
    ActionType.DRIBBLE,
    ActionType.SHOOT,
    ActionType.CLEAR,
)

DEFENSIVE_ACTIONS: tuple[ActionType, ...] = (
    ActionType.PRESS,
    ActionType.DROP,
    ActionType.MARK,
)


@dataclass(frozen=True)
class ActionSpace:
    actions: tuple[ActionType, ...] = tuple(ActionType)

    def normalize(self, action: ActionType | str) -> ActionType:
        if isinstance(action, ActionType):
            return action
        try:
            return ActionType(str(action))
        except ValueError as error:
            valid = ", ".join(item.value for item in self.actions)
            raise ValueError(f"Unsupported action '{action}'. Expected one of: {valid}") from error

    def validate(self, action: ActionType | str) -> ActionType:
        normalized = self.normalize(action)
        if normalized not in self.actions:
            valid = ", ".join(item.value for item in self.actions)
            raise ValueError(f"Action '{normalized.value}' is not available. Expected one of: {valid}")
        return normalized

    def sample(self, rng: np.random.Generator, has_possession: bool = True) -> ActionType:
        pool = POSSESSION_ACTIONS if has_possession else DEFENSIVE_ACTIONS
        return pool[int(rng.integers(0, len(pool)))]
