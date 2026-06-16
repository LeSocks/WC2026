from src.data.loader import DEFAULT_PLAYER_RATINGS_PATH, load_all_teams, load_team
from src.engine.tactics import TACTICAL_PRESETS
from src.models.formation import FORMATIONS
from src.models.player import Position


def test_tactical_presets_cover_seeded_top10_teams() -> None:
    teams = load_all_teams(DEFAULT_PLAYER_RATINGS_PATH)

    assert set(teams) == set(TACTICAL_PRESETS)
    assert len(teams) == 10


def test_formations_have_eleven_slots_and_coordinates() -> None:
    for formation in FORMATIONS.values():
        assert len(formation.slots) == 11
        assert len(formation.coordinates) == 11
        assert formation.slots[0] == Position.GK


def test_expanded_formation_catalog_is_available() -> None:
    expected_formations = {
        "4-3-3",
        "4-2-3-1",
        "4-5-1",
        "3-4-3",
        "4-4-2",
        "4-1-4-1",
        "4-3-2-1",
        "4-3-1-2",
        "4-2-2-2",
        "4-1-2-1-2",
        "3-5-2",
        "3-4-2-1",
        "3-4-1-2",
        "5-3-2",
        "5-4-1",
        "5-2-3",
    }

    assert expected_formations.issubset(FORMATIONS)


def test_team_lineup_can_be_printed() -> None:
    france = load_team("France")
    lineup = france.lineup_text()

    assert france.name in lineup
    assert "4-2-3-1" in lineup
    assert "GK: Mike Maignan" in lineup
    assert len(lineup.splitlines()) == 12


def test_team_quality_metrics_are_reasonable() -> None:
    spain = load_team("Spain")

    assert spain.average_overall > 80
    assert spain.passing_quality > 75
    assert spain.attacking_quality > 75
    assert spain.defensive_quality > 75
