import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_all_teams, load_players, load_team
from src.data.seed_48_teams import build_all_players_dataframe, build_all_teams_payload
from src.data.worldcup_2026 import FIFA_RANKINGS_2026_06_11, WORLD_CUP_GROUPS, all_teams


def test_world_cup_group_list_has_48_teams() -> None:
    teams = all_teams()

    assert len(teams) == 48
    assert len(set(teams)) == 48
    assert len(WORLD_CUP_GROUPS) == 12
    assert all(len(group_teams) == 4 for group_teams in WORLD_CUP_GROUPS.values())


def test_generated_all_players_dataframe_is_complete() -> None:
    dataframe = build_all_players_dataframe()

    assert len(dataframe) == 48 * 11
    assert dataframe.groupby("team").size().eq(11).all()
    assert set(dataframe["team"]) == set(all_teams())
    assert dataframe["overall"].between(35, 95).all()
    assert dataframe["fifa_rank"].isna().sum() == 0
    assert dataframe["fifa_rank"].astype(int).between(1, 211).all()


def test_all_teams_payload_schema_is_complete() -> None:
    payload = build_all_teams_payload()

    assert payload["schema_version"] == 1
    assert len(payload["teams"]) == 48
    assert {team["name"] for team in payload["teams"]} == set(all_teams())
    assert all(len(team["squad"]) == 11 for team in payload["teams"])
    assert all(isinstance(team["fifa_rank"], int) for team in payload["teams"])


def test_processed_files_exist_and_load_default_dataset() -> None:
    teams_path = Path("data/processed/all_teams.json")
    players_path = Path("data/processed/all_players_48.csv")

    assert teams_path.exists()
    assert players_path.exists()

    payload = json.loads(teams_path.read_text(encoding="utf-8"))
    players = pd.read_csv(players_path)

    assert len(payload["teams"]) == 48
    assert len(players) == 528
    assert "NaN" not in teams_path.read_text(encoding="utf-8")
    assert players["fifa_rank"].isna().sum() == 0
    assert len(load_players()) == 528
    assert len(load_all_teams()) == 48
    assert len(load_team("Japan").starters) == 11
    assert load_team("Japan").fifa_rank == 18


def test_official_fifa_rank_examples_are_present() -> None:
    assert len(FIFA_RANKINGS_2026_06_11) == 48
    assert set(FIFA_RANKINGS_2026_06_11) == set(all_teams())
    assert FIFA_RANKINGS_2026_06_11["Argentina"] == 1
    assert FIFA_RANKINGS_2026_06_11["United States"] == 17
    assert FIFA_RANKINGS_2026_06_11["Ivory Coast"] == 33
    assert FIFA_RANKINGS_2026_06_11["Curacao"] == 82
    assert FIFA_RANKINGS_2026_06_11["New Zealand"] == 85
