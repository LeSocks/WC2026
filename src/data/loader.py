from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.engine.tactics import TACTICAL_PRESETS
from src.models.player import Player
from src.models.team import Team


DEFAULT_PLAYER_RATINGS_PATH = Path("data/raw/fifa_style_player_ratings_top10.csv")


def load_players(path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> list[Player]:
    dataframe = pd.read_csv(path)
    return [Player.from_mapping(row) for row in dataframe.to_dict(orient="records")]


def load_team_players(team: str, path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> list[Player]:
    normalized_team = team.casefold()
    return [player for player in load_players(path) if player.team and player.team.casefold() == normalized_team]


def load_team(team: str, path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> Team:
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


def load_all_teams(path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> dict[str, Team]:
    players = load_players(path)
    teams = sorted({player.team for player in players if player.team})
    return {team: load_team(team, path) for team in teams}
