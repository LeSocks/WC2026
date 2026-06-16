from src.data.loader import load_team
from src.engine.events import EventType
from src.engine.match import MatchSimulator


def test_match_simulator_runs_full_match() -> None:
    france = load_team("France")
    morocco = load_team("Morocco")

    state = MatchSimulator(france, morocco, rng_seed=42).simulate()

    assert state.minute == 90
    assert state.home_goals >= 0
    assert state.away_goals >= 0
    assert state.home_shots + state.away_shots > 0
    assert 0 <= state.home_possession_pct <= 100
    assert state.events[0].event_type == EventType.KICKOFF
    assert state.events[-1].event_type == EventType.FULL_TIME


def test_match_simulator_is_seed_deterministic() -> None:
    first = MatchSimulator(load_team("Argentina"), load_team("Germany"), rng_seed=7).simulate()
    second = MatchSimulator(load_team("Argentina"), load_team("Germany"), rng_seed=7).simulate()

    assert (first.home_goals, first.away_goals, first.home_shots, first.away_shots) == (
        second.home_goals,
        second.away_goals,
        second.home_shots,
        second.away_shots,
    )
    assert [event.description for event in first.events[:10]] == [event.description for event in second.events[:10]]


def test_match_event_log_contains_terminal_scoreline() -> None:
    state = MatchSimulator(load_team("Spain"), load_team("Brazil"), rng_seed=11).simulate()

    assert f"Spain {state.home_goals} - {state.away_goals} Brazil" in state.events[-1].description
