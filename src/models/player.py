from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Position(Enum):
    GK = "goalkeeper"
    CB = "center_back"
    LB = "left_back"
    RB = "right_back"
    CDM = "defensive_midfielder"
    CM = "central_midfielder"
    CAM = "attacking_midfielder"
    LW = "left_winger"
    RW = "right_winger"
    ST = "striker"

    @classmethod
    def from_code(cls, value: str) -> Position:
        normalized = value.strip().upper()
        try:
            return cls[normalized]
        except KeyError as error:
            valid = ", ".join(member.name for member in cls)
            raise ValueError(f"Unsupported position '{value}'. Expected one of: {valid}") from error


class PlayStyle(Enum):
    TARGET_MAN = "target_man"
    POACHER = "poacher"
    FALSE_NINE = "false_nine"
    BOX_TO_BOX = "box_to_box"
    DEEP_LYING_PM = "deep_lying_pm"
    BALL_WINNER = "ball_winner"
    COMPLETE_FORWARD = "complete_fwd"
    DRIBBLER = "dribbler"
    SWEEPER_KEEPER = "sweeper_keeper"
    TRADITIONAL_GK = "traditional_gk"
    BALL_PLAYING_CB = "ball_playing_cb"
    CREATOR = "creator"
    FULL_BACK = "full_back"
    INVERTED_FULL_BACK = "inverted_full_back"
    PRESSING_FORWARD = "pressing_forward"
    STOPPER = "stopper"
    WING_BACK = "wing_back"

    @classmethod
    def from_value(cls, value: str) -> PlayStyle:
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported playstyle '{value}'. Expected one of: {valid}")


@dataclass(frozen=True)
class PlayerStats:
    """FIFA-style stats, scale 0-100."""

    pace: int
    shooting: int
    passing: int
    dribbling: int
    defending: int
    physical: int
    overall: int | None = None
    xg_per_shot: float = 0.0
    press_intensity: float = 0.0
    progressive_passes_pct: float = 0.0

    def __post_init__(self) -> None:
        stat_names = [
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physical",
            "overall",
        ]

        for stat_name in stat_names:
            value = getattr(self, stat_name)
            if value is None:
                continue
            if not 0 <= value <= 100:
                raise ValueError(f"{stat_name} must be between 0 and 100, got {value}")


@dataclass
class Player:
    name: str
    position: Position
    playstyle: PlayStyle
    stats: PlayerStats
    age: int = 25
    is_starter: bool = True
    team: str | None = None
    fifa_rank: int | None = None
    ratings_source: str | None = None

    _shot_accuracy_mean: float = field(init=False)
    _pass_success_rate: float = field(init=False)
    _press_trigger_distance: float = field(init=False)
    _dribble_success_rate: float = field(init=False)
    _aerial_win_rate: float = field(init=False)

    def __post_init__(self) -> None:
        if self.age <= 0:
            raise ValueError(f"age must be positive, got {self.age}")

        self._compute_behavioral_params()

    @property
    def shot_accuracy_mean(self) -> float:
        return self._shot_accuracy_mean

    @property
    def pass_success_rate(self) -> float:
        return self._pass_success_rate

    @property
    def press_trigger_distance(self) -> float:
        return self._press_trigger_distance

    @property
    def dribble_success_rate(self) -> float:
        return self._dribble_success_rate

    @property
    def aerial_win_rate(self) -> float:
        return self._aerial_win_rate

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> Player:
        stats = PlayerStats(
            pace=int(row["pace"]),
            shooting=int(row["shooting"]),
            passing=int(row["passing"]),
            dribbling=int(row["dribbling"]),
            defending=int(row["defending"]),
            physical=int(row["physical"]),
            overall=_optional_int(row.get("overall")),
        )

        return cls(
            name=str(row["player_name"]),
            team=_optional_str(row.get("team")),
            fifa_rank=_optional_int(row.get("fifa_rank")),
            position=Position.from_code(str(row["position"])),
            playstyle=PlayStyle.from_value(str(row["playstyle"])),
            stats=stats,
            age=int(row.get("age", 25)),
            is_starter=_to_bool(row.get("is_starter", True)),
            ratings_source=_optional_str(row.get("ratings_source")),
        )

    def _compute_behavioral_params(self) -> None:
        stats = self.stats

        self._shot_accuracy_mean = 0.25 + (stats.shooting / 100) * 0.55
        self._pass_success_rate = 0.60 + (stats.passing / 100) * 0.30 + (stats.dribbling / 100) * 0.05
        self._press_trigger_distance = 3.0 + (stats.pace / 100) * 5.0 + (stats.physical / 100) * 2.0
        self._dribble_success_rate = 0.35 + (stats.dribbling / 100) * 0.45
        self._aerial_win_rate = 0.35 + (stats.physical / 100) * 0.40 + (stats.shooting / 100) * 0.10

        self._apply_playstyle_modifiers()
        self._clamp_rates()

    def _apply_playstyle_modifiers(self) -> None:
        if self.playstyle == PlayStyle.POACHER:
            self._shot_accuracy_mean *= 1.12
        elif self.playstyle == PlayStyle.DRIBBLER:
            self._dribble_success_rate *= 1.15
        elif self.playstyle == PlayStyle.TARGET_MAN:
            self._aerial_win_rate *= 1.20
        elif self.playstyle == PlayStyle.BALL_WINNER:
            self._press_trigger_distance *= 1.10
        elif self.playstyle == PlayStyle.DEEP_LYING_PM:
            self._pass_success_rate *= 1.08
        elif self.playstyle == PlayStyle.COMPLETE_FORWARD:
            self._shot_accuracy_mean *= 1.06
            self._dribble_success_rate *= 1.04
        elif self.playstyle == PlayStyle.CREATOR:
            self._pass_success_rate *= 1.05
        elif self.playstyle == PlayStyle.PRESSING_FORWARD:
            self._press_trigger_distance *= 1.08
            self._shot_accuracy_mean *= 1.03
        elif self.playstyle == PlayStyle.STOPPER:
            self._aerial_win_rate *= 1.08
        elif self.playstyle == PlayStyle.BALL_PLAYING_CB:
            self._pass_success_rate *= 1.04
        elif self.playstyle in {PlayStyle.WING_BACK, PlayStyle.FULL_BACK, PlayStyle.INVERTED_FULL_BACK}:
            self._press_trigger_distance *= 1.04
            self._pass_success_rate *= 1.02
        elif self.playstyle == PlayStyle.SWEEPER_KEEPER:
            self._pass_success_rate *= 1.03

    def _clamp_rates(self) -> None:
        self._shot_accuracy_mean = _clamp(self._shot_accuracy_mean, 0.0, 0.95)
        self._pass_success_rate = _clamp(self._pass_success_rate, 0.0, 0.98)
        self._dribble_success_rate = _clamp(self._dribble_success_rate, 0.0, 0.95)
        self._aerial_win_rate = _clamp(self._aerial_win_rate, 0.0, 0.95)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
