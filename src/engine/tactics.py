from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PressStyle(Enum):
    HIGH_PRESS = "high_press"
    MID_BLOCK = "mid_block"
    LOW_BLOCK = "low_block"
    COUNTER_PRESS = "counter_press"


class AttackStyle(Enum):
    POSSESSION = "possession"
    DIRECT = "direct"
    COUNTER_ATTACK = "counter_attack"
    HIGH_LINE = "high_line"


@dataclass(frozen=True)
class TacticalConfig:
    formation: str
    press_style: PressStyle
    attack_style: AttackStyle
    pressing_intensity: float
    defensive_line_height: float
    width: float
    tempo: float

    def __post_init__(self) -> None:
        for field_name in ["pressing_intensity", "defensive_line_height", "width", "tempo"]:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")


TACTICAL_PRESETS: dict[str, TacticalConfig] = {
    "Argentina": TacticalConfig(
        formation="4-3-3",
        press_style=PressStyle.MID_BLOCK,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.66,
        defensive_line_height=0.58,
        width=0.58,
        tempo=0.58,
    ),
    "Spain": TacticalConfig(
        formation="4-3-3",
        press_style=PressStyle.HIGH_PRESS,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.82,
        defensive_line_height=0.75,
        width=0.65,
        tempo=0.55,
    ),
    "France": TacticalConfig(
        formation="4-2-3-1",
        press_style=PressStyle.MID_BLOCK,
        attack_style=AttackStyle.DIRECT,
        pressing_intensity=0.68,
        defensive_line_height=0.62,
        width=0.72,
        tempo=0.72,
    ),
    "England": TacticalConfig(
        formation="4-2-3-1",
        press_style=PressStyle.MID_BLOCK,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.64,
        defensive_line_height=0.60,
        width=0.62,
        tempo=0.58,
    ),
    "Portugal": TacticalConfig(
        formation="4-3-3",
        press_style=PressStyle.COUNTER_PRESS,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.76,
        defensive_line_height=0.70,
        width=0.66,
        tempo=0.66,
    ),
    "Brazil": TacticalConfig(
        formation="4-2-3-1",
        press_style=PressStyle.MID_BLOCK,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.62,
        defensive_line_height=0.60,
        width=0.75,
        tempo=0.65,
    ),
    "Morocco": TacticalConfig(
        formation="4-5-1",
        press_style=PressStyle.LOW_BLOCK,
        attack_style=AttackStyle.COUNTER_ATTACK,
        pressing_intensity=0.40,
        defensive_line_height=0.25,
        width=0.45,
        tempo=0.60,
    ),
    "Netherlands": TacticalConfig(
        formation="3-4-3",
        press_style=PressStyle.HIGH_PRESS,
        attack_style=AttackStyle.DIRECT,
        pressing_intensity=0.74,
        defensive_line_height=0.72,
        width=0.70,
        tempo=0.70,
    ),
    "Belgium": TacticalConfig(
        formation="4-2-3-1",
        press_style=PressStyle.MID_BLOCK,
        attack_style=AttackStyle.POSSESSION,
        pressing_intensity=0.58,
        defensive_line_height=0.54,
        width=0.62,
        tempo=0.55,
    ),
    "Germany": TacticalConfig(
        formation="4-2-3-1",
        press_style=PressStyle.HIGH_PRESS,
        attack_style=AttackStyle.HIGH_LINE,
        pressing_intensity=0.78,
        defensive_line_height=0.80,
        width=0.70,
        tempo=0.70,
    ),
}
