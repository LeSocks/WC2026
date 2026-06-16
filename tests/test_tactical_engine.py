from dataclasses import replace

from src.data.loader import load_team
from src.engine.events import PitchZone
from src.engine.tactics import AttackStyle, PressStyle, TacticalEngine
from src.models.team import Team


def test_high_press_creates_more_pressure_than_low_block() -> None:
    engine = TacticalEngine()
    germany = load_team("Germany")
    morocco = load_team("Morocco")

    assert engine.pressure_modifier(germany, PitchZone.ATT_CENTER) > engine.pressure_modifier(morocco, PitchZone.ATT_CENTER)


def test_attack_style_changes_action_probabilities() -> None:
    engine = TacticalEngine()
    france = load_team("France")
    player = next(player for player in france.starters if player.position.name == "ST")

    possession_team = Team(
        name="Possession France",
        squad=france.squad,
        tactical_config=replace(france.tactical_config, attack_style=AttackStyle.POSSESSION),
    )
    direct_team = Team(
        name="Direct France",
        squad=france.squad,
        tactical_config=replace(france.tactical_config, attack_style=AttackStyle.DIRECT),
    )

    possession_probs = engine.action_probabilities(player, possession_team, PitchZone.ATT_CENTER)
    direct_probs = engine.action_probabilities(player, direct_team, PitchZone.ATT_CENTER)

    assert possession_probs["pass"] > direct_probs["pass"]
    assert direct_probs["shoot"] > possession_probs["shoot"]


def test_transition_logic_moves_direct_teams_centrally() -> None:
    engine = TacticalEngine()
    australia = load_team("Australia")

    direct_team = Team(
        name="Direct Australia",
        squad=australia.squad,
        tactical_config=replace(australia.tactical_config, attack_style=AttackStyle.DIRECT),
    )

    assert engine.transition_zone(PitchZone.MID_LEFT, direct_team, "pass") == PitchZone.ATT_CENTER


def test_possession_tendency_changes_with_style_override() -> None:
    engine = TacticalEngine()
    spain = load_team("Spain")
    morocco = load_team("Morocco")
    low_block_spain = Team(
        name="Low Block Spain",
        squad=spain.squad,
        tactical_config=replace(
            spain.tactical_config,
            attack_style=AttackStyle.COUNTER_ATTACK,
            press_style=PressStyle.LOW_BLOCK,
            tempo=0.52,
        ),
    )

    assert engine.compute_possession_tendency(spain, morocco) > engine.compute_possession_tendency(low_block_spain, morocco)
