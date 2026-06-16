import pytest

from src.data.loader import load_all_teams, load_team
from src.engine.diagnostics import aggregate_diagnostics, run_pair_grid, summarize_matchup


def test_summarize_matchup_returns_distribution_metrics() -> None:
    summary = summarize_matchup(load_team("France"), load_team("Morocco"), runs=5, seed_offset=100)

    assert summary["home_team"] == "France"
    assert summary["away_team"] == "Morocco"
    assert summary["simulations"] == 5
    assert summary["avg_goals"] >= 0
    assert summary["avg_shots"] > 0
    assert summary["avg_shots_on_target"] <= summary["avg_shots"]
    assert summary["avg_xg"] > 0
    assert summary["home_win_rate"] + summary["draw_rate"] + summary["away_win_rate"] == pytest.approx(1.0)


def test_pair_grid_and_aggregate_diagnostics() -> None:
    teams = load_all_teams()
    results = run_pair_grid(
        teams,
        runs_per_pair=3,
        team_names=["France", "Morocco", "Spain"],
    )
    summary = aggregate_diagnostics(results)

    assert len(results) == 3
    assert summary["matchups"] == 3
    assert summary["simulations"] == 9
    assert summary["avg_goals"] >= 0
    assert summary["high_score_outlier_rate"] >= 0
