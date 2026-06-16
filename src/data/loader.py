from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.player import Player


DEFAULT_PLAYER_RATINGS_PATH = Path("data/raw/fifa_style_player_ratings_top10.csv")


def load_players(path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> list[Player]:
    dataframe = pd.read_csv(path)
    return [Player.from_mapping(row) for row in dataframe.to_dict(orient="records")]


def load_team_players(team: str, path: str | Path = DEFAULT_PLAYER_RATINGS_PATH) -> list[Player]:
    normalized_team = team.casefold()
    return [player for player in load_players(path) if player.team and player.team.casefold() == normalized_team]
