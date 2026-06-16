from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.engine.events import PitchZone


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


class TacticalEngine:
    def compute_possession_tendency(self, home_team, away_team) -> float:
        home_quality = self._possession_quality(home_team)
        away_quality = self._possession_quality(away_team)
        return float(np.clip(home_quality / (home_quality + away_quality), 0.35, 0.65))

    def pressure_modifier(self, pressing_team, zone: PitchZone) -> float:
        style_multiplier = {
            PressStyle.HIGH_PRESS: 1.20,
            PressStyle.COUNTER_PRESS: 1.12,
            PressStyle.MID_BLOCK: 0.92,
            PressStyle.LOW_BLOCK: 0.70,
        }[pressing_team.tactical_config.press_style]
        zone_multiplier = {
            PitchZone.DEF_LEFT: 0.78,
            PitchZone.DEF_CENTER: 0.78,
            PitchZone.DEF_RIGHT: 0.78,
            PitchZone.MID_LEFT: 1.00,
            PitchZone.MID_CENTER: 1.06,
            PitchZone.MID_RIGHT: 1.00,
            PitchZone.ATT_LEFT: 1.12,
            PitchZone.ATT_CENTER: 1.16,
            PitchZone.ATT_RIGHT: 1.12,
        }[zone]
        line_bonus = 0.84 + pressing_team.tactical_config.defensive_line_height * 0.32
        return float(pressing_team.tactical_config.pressing_intensity * style_multiplier * zone_multiplier * line_bonus)

    def transition_zone(self, zone: PitchZone, team, action: str) -> PitchZone:
        direct = team.tactical_config.attack_style in {AttackStyle.DIRECT, AttackStyle.COUNTER_ATTACK}
        if action == "dribble" and zone in {PitchZone.ATT_LEFT, PitchZone.ATT_RIGHT}:
            return PitchZone.ATT_CENTER

        transitions = {
            PitchZone.DEF_LEFT: PitchZone.MID_LEFT,
            PitchZone.DEF_CENTER: PitchZone.MID_CENTER,
            PitchZone.DEF_RIGHT: PitchZone.MID_RIGHT,
            PitchZone.MID_LEFT: PitchZone.ATT_CENTER if direct else PitchZone.ATT_LEFT,
            PitchZone.MID_CENTER: PitchZone.ATT_CENTER,
            PitchZone.MID_RIGHT: PitchZone.ATT_CENTER if direct else PitchZone.ATT_RIGHT,
            PitchZone.ATT_LEFT: PitchZone.ATT_CENTER,
            PitchZone.ATT_CENTER: PitchZone.ATT_CENTER,
            PitchZone.ATT_RIGHT: PitchZone.ATT_CENTER,
        }
        return transitions[zone]

    def action_probabilities(self, player, team, zone: PitchZone) -> dict[str, float]:
        zone_probs = {
            PitchZone.DEF_LEFT: {"pass": 0.82, "dribble": 0.15, "shoot": 0.03},
            PitchZone.DEF_CENTER: {"pass": 0.88, "dribble": 0.11, "shoot": 0.01},
            PitchZone.DEF_RIGHT: {"pass": 0.82, "dribble": 0.15, "shoot": 0.03},
            PitchZone.MID_LEFT: {"pass": 0.70, "dribble": 0.25, "shoot": 0.05},
            PitchZone.MID_CENTER: {"pass": 0.70, "dribble": 0.23, "shoot": 0.07},
            PitchZone.MID_RIGHT: {"pass": 0.70, "dribble": 0.25, "shoot": 0.05},
            PitchZone.ATT_LEFT: {"pass": 0.52, "dribble": 0.30, "shoot": 0.18},
            PitchZone.ATT_CENTER: {"pass": 0.44, "dribble": 0.24, "shoot": 0.32},
            PitchZone.ATT_RIGHT: {"pass": 0.52, "dribble": 0.30, "shoot": 0.18},
        }
        probs = dict(zone_probs[zone])

        playstyle = getattr(player, "playstyle", None)
        playstyle_value = getattr(playstyle, "value", "")
        if playstyle_value == "poacher" and zone == PitchZone.ATT_CENTER:
            probs["shoot"] *= 1.35
        elif playstyle_value == "dribbler":
            probs["dribble"] *= 1.25
        elif playstyle_value in {"deep_lying_pm", "creator"}:
            probs["pass"] *= 1.18

        if team.tactical_config.attack_style == AttackStyle.DIRECT:
            probs["pass"] *= 0.88
            probs["shoot"] *= 1.18
        elif team.tactical_config.attack_style == AttackStyle.POSSESSION:
            probs["pass"] *= 1.18
            probs["shoot"] *= 0.90
        elif team.tactical_config.attack_style == AttackStyle.COUNTER_ATTACK:
            probs["dribble"] *= 1.10
            probs["shoot"] *= 1.08

        total = sum(probs.values())
        return {action: value / total for action, value in probs.items()}

    def pass_success_modifier(self, team, opponent, zone: PitchZone, home_possession_tendency: float, is_home: bool) -> float:
        pressure = self.pressure_modifier(opponent, zone)
        possession_bonus = (home_possession_tendency - 0.5) * 0.08 if is_home else (0.5 - home_possession_tendency) * 0.08
        tempo_penalty = max(0.0, team.tactical_config.tempo - 0.72) * 0.07
        return float(possession_bonus - pressure * 0.10 - tempo_penalty)

    def dribble_success_modifier(self, opponent, zone: PitchZone) -> float:
        return float(-(self.pressure_modifier(opponent, zone) * 0.085))

    def _possession_quality(self, team) -> float:
        quality = team.passing_quality + (team.tactical_config.tempo * 6)
        if team.tactical_config.attack_style == AttackStyle.POSSESSION:
            quality += 4
        elif team.tactical_config.attack_style == AttackStyle.COUNTER_ATTACK:
            quality -= 2

        if team.tactical_config.press_style in {PressStyle.HIGH_PRESS, PressStyle.COUNTER_PRESS}:
            quality += 2
        elif team.tactical_config.press_style == PressStyle.LOW_BLOCK:
            quality -= 1

        return max(1.0, quality)
