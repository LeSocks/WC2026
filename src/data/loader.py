from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.engine.tactics import AttackStyle, PressStyle, TACTICAL_PRESETS, TacticalConfig
from src.models.player import Player
from src.models.team import Team


DEFAULT_PLAYER_RATINGS_PATH = Path("data/raw/fifa_style_player_ratings_top10.csv")
DEFAULT_ALL_PLAYERS_PATH = Path("data/processed/all_players_48.csv")
DEFAULT_ALL_TEAMS_PATH = Path("data/processed/all_teams.json")


def load_players(path: str | Path | None = None) -> list[Player]:
    path = Path(path) if path is not None else _default_player_path()
    dataframe = pd.read_csv(path)
    return [Player.from_mapping(row) for row in dataframe.to_dict(orient="records")]


def load_team_players(team: str, path: str | Path | None = None) -> list[Player]:
    normalized_team = team.casefold()
    return [player for player in load_players(path) if player.team and player.team.casefold() == normalized_team]


def load_team(team: str, path: str | Path | None = None) -> Team:
    if path is None and DEFAULT_ALL_TEAMS_PATH.exists():
        teams = load_all_teams(DEFAULT_ALL_TEAMS_PATH)
        try:
            return teams[team]
        except KeyError as error:
            raise ValueError(f"No team found for '{team}'") from error

    players = load_team_players(team, path)
    if not players:
        raise ValueError(f"No players found for team '{team}'")
    if team not in TACTICAL_PRESETS:
        raise ValueError(f"No tactical preset found for team '{team}'")

    fifa_rank = players[0].fifa_rank
    return Team(
        name=team,
        squad=players,
        fifa_rank=fifa_rank,
        tactical_config=TACTICAL_PRESETS[team],
    )


def load_all_teams(path: str | Path | None = None) -> dict[str, Team]:
    if path is None:
        path = DEFAULT_ALL_TEAMS_PATH if DEFAULT_ALL_TEAMS_PATH.exists() else DEFAULT_PLAYER_RATINGS_PATH

    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {team["name"]: _team_from_payload(team) for team in payload["teams"]}

    players = load_players(path)
    teams = sorted({player.team for player in players if player.team})
    return {team: load_team(team, path) for team in teams}


def _team_from_payload(payload: dict[str, object]) -> Team:
    tactics = payload["tactics"]
    if not isinstance(tactics, dict):
        raise ValueError("team payload tactics must be a dict")

    squad_payload = payload["squad"]
    if not isinstance(squad_payload, list):
        raise ValueError("team payload squad must be a list")

    return Team(
        name=str(payload["name"]),
        squad=[Player.from_mapping(row) for row in squad_payload],
        fifa_rank=int(payload["fifa_rank"]) if payload.get("fifa_rank") is not None else None,
        tactical_config=TacticalConfig(
            formation=str(tactics["formation"]),
            press_style=PressStyle(str(tactics["press_style"])),
            attack_style=AttackStyle(str(tactics["attack_style"])),
            pressing_intensity=float(tactics["pressing_intensity"]),
            defensive_line_height=float(tactics["defensive_line_height"]),
            width=float(tactics["width"]),
            tempo=float(tactics["tempo"]),
        ),
    )


def _default_player_path() -> Path:
    return DEFAULT_ALL_PLAYERS_PATH if DEFAULT_ALL_PLAYERS_PATH.exists() else DEFAULT_PLAYER_RATINGS_PATH
