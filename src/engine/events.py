from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PitchZone(Enum):
    DEF_LEFT = "def_left"
    DEF_CENTER = "def_center"
    DEF_RIGHT = "def_right"
    MID_LEFT = "mid_left"
    MID_CENTER = "mid_center"
    MID_RIGHT = "mid_right"
    ATT_LEFT = "att_left"
    ATT_CENTER = "att_center"
    ATT_RIGHT = "att_right"


class EventType(Enum):
    KICKOFF = "kickoff"
    PASS = "pass"
    DRIBBLE = "dribble"
    SHOT = "shot"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    GOAL = "goal"
    SAVE = "save"
    MISS = "miss"
    BLOCKED = "blocked"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"


@dataclass(frozen=True)
class MatchEvent:
    minute: int
    event_type: EventType
    team: str
    player_name: str
    zone: PitchZone
    success: bool
    description: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class MatchState:
    minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    possession_team: str = ""
    current_zone: PitchZone = PitchZone.MID_CENTER
    events: list[MatchEvent] = field(default_factory=list)
    home_possession_pct: float = 50.0
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
