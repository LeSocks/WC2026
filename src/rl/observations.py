from __future__ import annotations

from dataclasses import dataclass

from src.engine.events import MatchState, PitchZone
from src.engine.tactics import TacticalEngine
from src.models.player import Player, Position
from src.models.team import Team


ROLE_GROUPS: dict[Position, str] = {
    Position.GK: "goalkeeper",
    Position.CB: "defense",
    Position.LB: "defense",
    Position.RB: "defense",
    Position.CDM: "midfield",
    Position.CM: "midfield",
    Position.CAM: "midfield",
    Position.LW: "attack",
    Position.RW: "attack",
    Position.ST: "attack",
}

ZONE_INDEX: dict[PitchZone, int] = {
    PitchZone.DEF_LEFT: 0,
    PitchZone.DEF_CENTER: 1,
    PitchZone.DEF_RIGHT: 2,
    PitchZone.MID_LEFT: 3,
    PitchZone.MID_CENTER: 4,
    PitchZone.MID_RIGHT: 5,
    PitchZone.ATT_LEFT: 6,
    PitchZone.ATT_CENTER: 7,
    PitchZone.ATT_RIGHT: 8,
}


@dataclass(frozen=True)
class AgentObservation:
    agent_id: str
    team: str
    side: str
    player_name: str
    position: str
    role_group: str
    minute: int
    score_for: int
    score_against: int
    has_possession: bool
    current_zone: str
    player_zone: str
    formation: str
    press_style: str
    attack_style: str
    overall: int
    pace: int
    shooting: int
    passing: int
    dribbling: int
    defending: int
    physical: int
    relative_team_strength: float
    nearest_zone_pressure: float
    local_teammates: int
    local_opponents: int

    @property
    def score_diff(self) -> int:
        return self.score_for - self.score_against

    def state_key(self) -> tuple[object, ...]:
        minute_bucket = min(5, self.minute // 15)
        score_bucket = max(-2, min(2, self.score_diff))
        strength_bucket = round(self.relative_team_strength, 1)
        pressure_bucket = round(self.nearest_zone_pressure, 1)
        return (
            minute_bucket,
            score_bucket,
            self.has_possession,
            self.current_zone,
            self.player_zone,
            self.role_group,
            self.attack_style,
            self.press_style,
            strength_bucket,
            pressure_bucket,
        )

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.minute / 90,
            self.score_diff / 5,
            1.0 if self.has_possession else 0.0,
            ZONE_INDEX[PitchZone(self.current_zone)] / 8,
            ZONE_INDEX[PitchZone(self.player_zone)] / 8,
            self.overall / 100,
            self.pace / 100,
            self.shooting / 100,
            self.passing / 100,
            self.dribbling / 100,
            self.defending / 100,
            self.physical / 100,
            self.relative_team_strength,
            self.nearest_zone_pressure,
            self.local_teammates / 11,
            self.local_opponents / 11,
        )


class ObservationBuilder:
    def __init__(self) -> None:
        self.tactical_engine = TacticalEngine()

    def build(self, home: Team, away: Team, state: MatchState) -> dict[str, AgentObservation]:
        home_zones = self._player_zones(home)
        away_zones = self._player_zones(away)
        observations: dict[str, AgentObservation] = {}

        observations.update(
            self._build_side(
                side="home",
                team=home,
                opponent=away,
                own_zones=home_zones,
                opponent_zones=away_zones,
                state=state,
            )
        )
        observations.update(
            self._build_side(
                side="away",
                team=away,
                opponent=home,
                own_zones=away_zones,
                opponent_zones=home_zones,
                state=state,
            )
        )
        return observations

    def _build_side(
        self,
        side: str,
        team: Team,
        opponent: Team,
        own_zones: list[PitchZone],
        opponent_zones: list[PitchZone],
        state: MatchState,
    ) -> dict[str, AgentObservation]:
        result: dict[str, AgentObservation] = {}
        score_for = state.home_goals if side == "home" else state.away_goals
        score_against = state.away_goals if side == "home" else state.home_goals
        relative_strength = (team.average_overall - opponent.average_overall) / 100
        pressure = self.tactical_engine.pressure_modifier(opponent, state.current_zone)

        for index, player in enumerate(team.ordered_starters):
            player_zone = own_zones[index]
            agent_id = f"{side}:{index}"
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                team=team.name,
                side=side,
                player_name=player.name,
                position=player.position.name,
                role_group=ROLE_GROUPS[player.position],
                minute=state.minute,
                score_for=score_for,
                score_against=score_against,
                has_possession=state.possession_team == team.name,
                current_zone=state.current_zone.value,
                player_zone=player_zone.value,
                formation=team.tactical_config.formation,
                press_style=team.tactical_config.press_style.value,
                attack_style=team.tactical_config.attack_style.value,
                overall=player.stats.overall or 75,
                pace=player.stats.pace,
                shooting=player.stats.shooting,
                passing=player.stats.passing,
                dribbling=player.stats.dribbling,
                defending=player.stats.defending,
                physical=player.stats.physical,
                relative_team_strength=round(relative_strength, 3),
                nearest_zone_pressure=round(pressure, 3),
                local_teammates=own_zones.count(player_zone),
                local_opponents=opponent_zones.count(player_zone),
            )
        return result

    def _player_zones(self, team: Team) -> list[PitchZone]:
        zones: list[PitchZone] = []
        for x, y in team.formation.coordinates:
            zones.append(self._coordinate_to_zone(x, y))
        return zones

    def _coordinate_to_zone(self, x: float, y: float) -> PitchZone:
        if y < 27:
            lane = "LEFT"
        elif y > 53:
            lane = "RIGHT"
        else:
            lane = "CENTER"

        if x < 40:
            third = "DEF"
        elif x < 70:
            third = "MID"
        else:
            third = "ATT"

        return PitchZone[f"{third}_{lane}"]


def role_group_for_player(player: Player) -> str:
    return ROLE_GROUPS[player.position]
