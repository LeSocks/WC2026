import pytest

from src.data.loader import load_team_players
from src.models.player import Player, PlayerStats, PlayStyle, Position


def test_player_computes_behavioral_params_from_stats() -> None:
    player = Player(
        name="Example Forward",
        position=Position.ST,
        playstyle=PlayStyle.POACHER,
        stats=PlayerStats(
            pace=80,
            shooting=90,
            passing=70,
            dribbling=75,
            defending=35,
            physical=78,
        ),
    )

    baseline_shot_accuracy = 0.25 + 0.90 * 0.55

    assert player.shot_accuracy_mean == pytest.approx(baseline_shot_accuracy * 1.12)
    assert player.pass_success_rate == pytest.approx(0.60 + 0.70 * 0.30 + 0.75 * 0.05)
    assert player.press_trigger_distance == pytest.approx(3.0 + 0.80 * 5.0 + 0.78 * 2.0)
    assert player.dribble_success_rate == pytest.approx(0.35 + 0.75 * 0.45)


def test_playstyle_modifiers_differentiate_players() -> None:
    stats = PlayerStats(
        pace=82,
        shooting=80,
        passing=76,
        dribbling=88,
        defending=42,
        physical=71,
    )
    plain = Player("Plain", Position.LW, PlayStyle.COMPLETE_FORWARD, stats)
    dribbler = Player("Dribbler", Position.LW, PlayStyle.DRIBBLER, stats)

    assert dribbler.dribble_success_rate > plain.dribble_success_rate


def test_player_stats_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="pace must be between 0 and 100"):
        PlayerStats(
            pace=101,
            shooting=80,
            passing=80,
            dribbling=80,
            defending=80,
            physical=80,
        )


def test_player_from_mapping_parses_seed_row() -> None:
    row = {
        "team": "France",
        "fifa_rank": "3",
        "player_name": "Kylian Mbappe",
        "position": "LW",
        "age": "27",
        "is_starter": "True",
        "pace": "97",
        "shooting": "90",
        "passing": "83",
        "dribbling": "93",
        "defending": "39",
        "physical": "78",
        "overall": "91",
        "playstyle": "complete_fwd",
        "ratings_source": "manual_seed_estimate",
    }

    player = Player.from_mapping(row)

    assert player.team == "France"
    assert player.fifa_rank == 3
    assert player.position == Position.LW
    assert player.playstyle == PlayStyle.COMPLETE_FORWARD
    assert player.stats.overall == 91
    assert player.shot_accuracy_mean > 0.75


def test_loader_reads_day2_seed_players() -> None:
    france_players = load_team_players("France")

    assert len(france_players) == 11
    assert {player.name for player in france_players} >= {"Kylian Mbappe", "Mike Maignan"}
    assert all(player.is_starter for player in france_players)
