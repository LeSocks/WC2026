from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.engine.tactics import TacticalConfig
from src.models.formation import Formation, get_formation
from src.models.player import Player, Position


ZONE_POSITION_WEIGHTS: dict[str, tuple[Position, ...]] = {
    "def_left": (Position.LB, Position.CB, Position.CDM, Position.GK),
    "def_center": (Position.CB, Position.CDM, Position.GK, Position.CM),
    "def_right": (Position.RB, Position.CB, Position.CDM, Position.GK),
    "mid_left": (Position.LW, Position.LB, Position.CM, Position.CDM),
    "mid_center": (Position.CM, Position.CAM, Position.CDM, Position.ST),
    "mid_right": (Position.RW, Position.RB, Position.CM, Position.CDM),
    "att_left": (Position.LW, Position.ST, Position.CAM, Position.LB),
    "att_center": (Position.ST, Position.CAM, Position.CM, Position.RW, Position.LW),
    "att_right": (Position.RW, Position.ST, Position.CAM, Position.RB),
}


@dataclass
class Team:
    name: str
    squad: list[Player]
    tactical_config: TacticalConfig
    fifa_rank: int | None = None

    def __post_init__(self) -> None:
        if len(self.starters) != 11:
            raise ValueError(f"{self.name} must have exactly 11 starters, got {len(self.starters)}")

    @property
    def formation(self) -> Formation:
        return get_formation(self.tactical_config.formation)

    @property
    def starters(self) -> list[Player]:
        return [player for player in self.squad if player.is_starter]

    @property
    def ordered_starters(self) -> list[Player]:
        remaining = list(self.starters)
        ordered: list[Player] = []

        for slot in self.formation.slots:
            match_index = next((index for index, player in enumerate(remaining) if player.position == slot), None)
            if match_index is None:
                match_index = 0
            ordered.append(remaining.pop(match_index))

        return ordered

    @property
    def average_overall(self) -> float:
        values = [player.stats.overall for player in self.starters if player.stats.overall is not None]
        if not values:
            return 75.0
        return float(sum(values) / len(values))

    @property
    def attacking_quality(self) -> float:
        attackers = [player for player in self.starters if player.position in {Position.LW, Position.RW, Position.ST, Position.CAM}]
        pool = attackers or self.starters
        return float(sum((player.stats.shooting + player.stats.dribbling) / 2 for player in pool) / len(pool))

    @property
    def defensive_quality(self) -> float:
        defenders = [player for player in self.starters if player.position in {Position.GK, Position.CB, Position.LB, Position.RB, Position.CDM}]
        pool = defenders or self.starters
        return float(sum((player.stats.defending + player.stats.physical) / 2 for player in pool) / len(pool))

    @property
    def passing_quality(self) -> float:
        return float(sum(player.stats.passing for player in self.starters) / len(self.starters))

    def get_player_in_zone(self, zone: object, rng: np.random.Generator) -> Player:
        zone_key = getattr(zone, "value", str(zone))
        preferred_positions = ZONE_POSITION_WEIGHTS.get(zone_key, (Position.CM, Position.CDM, Position.CAM))
        candidates = [player for player in self.starters if player.position in preferred_positions]
        if not candidates:
            candidates = self.starters

        weights = np.array([preferred_positions.count(player.position) + 1 for player in candidates], dtype=float)
        weights = weights / weights.sum()
        return candidates[int(rng.choice(len(candidates), p=weights))]

    def lineup_text(self) -> str:
        lines = [f"{self.name} ({self.tactical_config.formation})"]
        for player, position in zip(self.ordered_starters, self.formation.slots):
            lines.append(f"{position.name}: {player.name} ({player.position.name}, {player.stats.overall})")
        return "\n".join(lines)
