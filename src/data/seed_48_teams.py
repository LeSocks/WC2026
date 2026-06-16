from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.seed_top10_ratings import build_player_ratings_dataframe
from src.data.worldcup_2026 import CONFEDERATIONS, TEAM_STRENGTHS, WORLD_CUP_GROUPS, all_teams, team_group
from src.engine.tactics import AttackStyle, PressStyle, TacticalConfig, TACTICAL_PRESETS


OUTPUT_TEAMS_PATH = Path("data/processed/all_teams.json")
OUTPUT_PLAYERS_PATH = Path("data/processed/all_players_48.csv")
DATA_QUALITY = "tier1_generated_from_team_profile"

POSITION_TEMPLATES = [
    ("GK", "Goalkeeper", {"pace": -24, "shooting": -48, "passing": -8, "dribbling": -18, "defending": 8, "physical": 5, "overall": 0, "playstyle": "traditional_gk"}),
    ("RB", "Right Back", {"pace": 6, "shooting": -13, "passing": 0, "dribbling": 1, "defending": 2, "physical": 1, "overall": -1, "playstyle": "full_back"}),
    ("CB", "Centre Back 1", {"pace": -4, "shooting": -28, "passing": -6, "dribbling": -8, "defending": 8, "physical": 9, "overall": 0, "playstyle": "stopper"}),
    ("CB", "Centre Back 2", {"pace": -6, "shooting": -30, "passing": -3, "dribbling": -7, "defending": 7, "physical": 8, "overall": -1, "playstyle": "ball_playing_cb"}),
    ("LB", "Left Back", {"pace": 5, "shooting": -14, "passing": 0, "dribbling": 1, "defending": 2, "physical": 1, "overall": -1, "playstyle": "full_back"}),
    ("CDM", "Defensive Midfielder", {"pace": -2, "shooting": -10, "passing": 4, "dribbling": 0, "defending": 6, "physical": 6, "overall": 0, "playstyle": "ball_winner"}),
    ("CM", "Central Midfielder 1", {"pace": 0, "shooting": -3, "passing": 7, "dribbling": 4, "defending": 1, "physical": 1, "overall": 1, "playstyle": "box_to_box"}),
    ("CM", "Central Midfielder 2", {"pace": -1, "shooting": -5, "passing": 8, "dribbling": 3, "defending": 0, "physical": 0, "overall": 0, "playstyle": "deep_lying_pm"}),
    ("LW", "Left Winger", {"pace": 8, "shooting": 5, "passing": 3, "dribbling": 8, "defending": -26, "physical": -5, "overall": 1, "playstyle": "dribbler"}),
    ("ST", "Striker", {"pace": 4, "shooting": 10, "passing": -3, "dribbling": 3, "defending": -29, "physical": 5, "overall": 1, "playstyle": "complete_fwd"}),
    ("RW", "Right Winger", {"pace": 8, "shooting": 4, "passing": 3, "dribbling": 7, "defending": -26, "physical": -5, "overall": 0, "playstyle": "dribbler"}),
]


TACTIC_BY_CONFEDERATION = {
    "AFC": (PressStyle.MID_BLOCK, AttackStyle.DIRECT, 0.56, 0.52, 0.58, 0.62),
    "CAF": (PressStyle.MID_BLOCK, AttackStyle.COUNTER_ATTACK, 0.60, 0.54, 0.62, 0.66),
    "CONCACAF": (PressStyle.MID_BLOCK, AttackStyle.DIRECT, 0.58, 0.55, 0.60, 0.62),
    "CONMEBOL": (PressStyle.COUNTER_PRESS, AttackStyle.POSSESSION, 0.68, 0.62, 0.66, 0.64),
    "OFC": (PressStyle.LOW_BLOCK, AttackStyle.DIRECT, 0.44, 0.38, 0.52, 0.56),
    "UEFA": (PressStyle.HIGH_PRESS, AttackStyle.POSSESSION, 0.70, 0.66, 0.64, 0.62),
}


def build_all_players_dataframe() -> pd.DataFrame:
    manual_players = build_player_ratings_dataframe()
    rows = []
    manual_teams = set(manual_players["team"])

    for team in all_teams():
        if team in manual_teams:
            team_rows = manual_players[manual_players["team"] == team].copy()
            team_rows["group"] = team_group(team)
            team_rows["confederation"] = CONFEDERATIONS[team]
            team_rows["data_quality"] = "tier1_manual_seed"
            rows.extend(team_rows.to_dict(orient="records"))
            continue

        rows.extend(_generated_team_players(team))

    dataframe = pd.DataFrame(rows)
    return dataframe[
        [
            "team",
            "group",
            "confederation",
            "fifa_rank",
            "player_name",
            "position",
            "age",
            "is_starter",
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physical",
            "overall",
            "playstyle",
            "ratings_source",
            "ranking_date",
            "data_quality",
        ]
    ].sort_values(["group", "team", "position", "player_name"])


def build_all_teams_payload() -> dict[str, object]:
    players = build_all_players_dataframe()
    teams = []

    for team in all_teams():
        tactic = tactical_config_for_team(team)
        team_players = players[players["team"] == team]
        teams.append(
            {
                "name": team,
                "group": team_group(team),
                "confederation": CONFEDERATIONS[team],
                "team_strength": TEAM_STRENGTHS[team],
                "data_quality": "tier1_manual_seed" if team in set(build_player_ratings_dataframe()["team"]) else DATA_QUALITY,
                "tactics": {
                    "formation": tactic.formation,
                    "press_style": tactic.press_style.value,
                    "attack_style": tactic.attack_style.value,
                    "pressing_intensity": tactic.pressing_intensity,
                    "defensive_line_height": tactic.defensive_line_height,
                    "width": tactic.width,
                    "tempo": tactic.tempo,
                },
                "squad": team_players.to_dict(orient="records"),
            }
        )

    return {
        "schema_version": 1,
        "source_note": "WC2026 groups verified from current tournament schedule; non-top-10 squads are generated Tier 1 placeholders.",
        "groups": WORLD_CUP_GROUPS,
        "teams": teams,
    }


def write_seed_data(
    teams_path: str | Path = OUTPUT_TEAMS_PATH,
    players_path: str | Path = OUTPUT_PLAYERS_PATH,
) -> dict[str, Path]:
    teams_output = Path(teams_path)
    players_output = Path(players_path)
    teams_output.parent.mkdir(parents=True, exist_ok=True)
    players_output.parent.mkdir(parents=True, exist_ok=True)

    players = build_all_players_dataframe()
    players.to_csv(players_output, index=False)
    teams_output.write_text(json.dumps(build_all_teams_payload(), indent=2), encoding="utf-8")

    return {"teams": teams_output, "players": players_output}


def tactical_config_for_team(team: str) -> TacticalConfig:
    if team in TACTICAL_PRESETS:
        return TACTICAL_PRESETS[team]

    confederation = CONFEDERATIONS[team]
    strength = TEAM_STRENGTHS[team]
    press_style, attack_style, press, line, width, tempo = TACTIC_BY_CONFEDERATION[confederation]
    strength_delta = (strength - 75) / 100

    if strength >= 84:
        formation = "4-3-3"
    elif attack_style == AttackStyle.COUNTER_ATTACK:
        formation = "4-5-1"
    else:
        formation = "4-2-3-1"

    return TacticalConfig(
        formation=formation,
        press_style=press_style,
        attack_style=attack_style,
        pressing_intensity=_clamp(press + strength_delta, 0.30, 0.84),
        defensive_line_height=_clamp(line + strength_delta, 0.25, 0.82),
        width=_clamp(width, 0.42, 0.78),
        tempo=_clamp(tempo + strength_delta / 2, 0.45, 0.76),
    )


def _generated_team_players(team: str) -> list[dict[str, object]]:
    strength = TEAM_STRENGTHS[team]
    group = team_group(team)
    confederation = CONFEDERATIONS[team]
    rows = []

    for index, (position, role_name, offsets) in enumerate(POSITION_TEMPLATES, start=1):
        rows.append(
            {
                "team": team,
                "group": group,
                "confederation": confederation,
                "fifa_rank": None,
                "player_name": f"{team} {role_name}",
                "position": position,
                "age": _age_for_slot(index),
                "is_starter": True,
                "pace": _stat(strength, offsets["pace"]),
                "shooting": _stat(strength, offsets["shooting"]),
                "passing": _stat(strength, offsets["passing"]),
                "dribbling": _stat(strength, offsets["dribbling"]),
                "defending": _stat(strength, offsets["defending"]),
                "physical": _stat(strength, offsets["physical"]),
                "overall": _stat(strength, offsets["overall"]),
                "playstyle": offsets["playstyle"],
                "ratings_source": "generated_team_profile",
                "ranking_date": "2026-06-11",
                "data_quality": DATA_QUALITY,
            }
        )

    return rows


def _stat(base: int, offset: object) -> int:
    return int(_clamp(base + int(offset), 35, 95))


def _age_for_slot(index: int) -> int:
    ages = [29, 27, 28, 30, 27, 28, 26, 29, 25, 27, 25]
    return ages[index - 1]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


if __name__ == "__main__":
    for label, path in write_seed_data().items():
        print(f"{label}: {path}")
