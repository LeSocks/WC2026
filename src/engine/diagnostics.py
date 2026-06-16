from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.loader import load_all_teams
from src.engine.match import MatchSimulator
from src.models.team import Team


def summarize_matchup(
    home_team: Team,
    away_team: Team,
    *,
    runs: int = 100,
    seed_offset: int = 0,
    high_score_threshold: int = 7,
) -> dict[str, object]:
    if runs <= 0:
        raise ValueError("runs must be positive")

    home_wins = 0
    draws = 0
    away_wins = 0
    total_goals = 0
    total_shots = 0
    total_shots_on_target = 0
    total_xg = 0.0
    total_goal_diff = 0
    high_score_outliers = 0
    max_total_goals = 0

    for index in range(runs):
        state = MatchSimulator(home_team, away_team, rng_seed=seed_offset + index).simulate()
        match_goals = state.home_goals + state.away_goals

        if state.home_goals > state.away_goals:
            home_wins += 1
        elif state.home_goals == state.away_goals:
            draws += 1
        else:
            away_wins += 1

        total_goals += match_goals
        total_shots += state.home_shots + state.away_shots
        total_shots_on_target += state.home_shots_on_target + state.away_shots_on_target
        total_xg += state.home_xg + state.away_xg
        total_goal_diff += abs(state.home_goals - state.away_goals)
        high_score_outliers += int(match_goals >= high_score_threshold)
        max_total_goals = max(max_total_goals, match_goals)

    return {
        "home_team": home_team.name,
        "away_team": away_team.name,
        "simulations": runs,
        "avg_goals": round(total_goals / runs, 3),
        "avg_shots": round(total_shots / runs, 3),
        "avg_shots_on_target": round(total_shots_on_target / runs, 3),
        "avg_xg": round(total_xg / runs, 3),
        "avg_goal_diff": round(total_goal_diff / runs, 3),
        "home_win_rate": round(home_wins / runs, 3),
        "draw_rate": round(draws / runs, 3),
        "away_win_rate": round(away_wins / runs, 3),
        "high_score_outliers": high_score_outliers,
        "max_total_goals": max_total_goals,
    }


def run_pair_grid(
    teams: dict[str, Team],
    *,
    runs_per_pair: int = 20,
    team_names: Iterable[str] | None = None,
    high_score_threshold: int = 7,
) -> pd.DataFrame:
    selected_names = sorted(team_names or teams)
    rows: list[dict[str, object]] = []

    for pair_index, (home_name, away_name) in enumerate(combinations(selected_names, 2)):
        rows.append(
            summarize_matchup(
                teams[home_name],
                teams[away_name],
                runs=runs_per_pair,
                seed_offset=pair_index * 10_000,
                high_score_threshold=high_score_threshold,
            )
        )

    return pd.DataFrame(rows)


def aggregate_diagnostics(results: pd.DataFrame) -> dict[str, float | int]:
    if results.empty:
        raise ValueError("results must contain at least one matchup")

    total_simulations = int(results["simulations"].sum())
    total_outliers = int(results["high_score_outliers"].sum())

    return {
        "matchups": int(len(results)),
        "simulations": total_simulations,
        "avg_goals": round(float(results["avg_goals"].mean()), 3),
        "avg_shots": round(float(results["avg_shots"].mean()), 3),
        "avg_shots_on_target": round(float(results["avg_shots_on_target"].mean()), 3),
        "avg_xg": round(float(results["avg_xg"].mean()), 3),
        "high_score_outliers": total_outliers,
        "high_score_outlier_rate": round(total_outliers / total_simulations, 4),
        "max_total_goals": int(results["max_total_goals"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run match-engine distribution diagnostics.")
    parser.add_argument("--runs-per-pair", type=int, default=20)
    parser.add_argument("--teams", nargs="*", help="Optional subset of team names.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    teams = load_all_teams()
    results = run_pair_grid(teams, runs_per_pair=args.runs_per_pair, team_names=args.teams)
    summary = aggregate_diagnostics(results)

    print(pd.DataFrame([summary]).to_string(index=False))
    print()
    print(results.sort_values("avg_goals", ascending=False).head(10).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nWrote diagnostics CSV to {args.output}")


if __name__ == "__main__":
    main()
