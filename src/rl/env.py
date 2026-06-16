from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from src.engine.events import EventType, MatchEvent, MatchState, PitchZone
from src.engine.tactics import TacticalEngine
from src.models.player import PlayStyle, Position
from src.models.team import Team
from src.rl.actions import ActionSpace, ActionType, POSSESSION_ACTIONS
from src.rl.observations import AgentObservation, ObservationBuilder, role_group_for_player
from src.rl.policies import Policy, TacticalPresetPolicy
from src.rl.rewards import RewardFunction


class FootballEnv:
    """Lightweight football MARL environment with 11v11-compatible agent slots."""

    def __init__(
        self,
        home_team: Team,
        away_team: Team,
        rng_seed: Optional[int] = None,
        max_minutes: int = 90,
        default_policy: Policy | None = None,
    ) -> None:
        self.home = home_team
        self.away = away_team
        self.rng = np.random.default_rng(rng_seed)
        self.max_minutes = max_minutes
        self.action_space = ActionSpace()
        self.observation_builder = ObservationBuilder()
        self.reward_function = RewardFunction()
        self.tactical_engine = TacticalEngine()
        self.default_policy = default_policy or TacticalPresetPolicy()
        self.state = MatchState()
        self._home_possession_minutes = 0
        self._away_possession_minutes = 0
        self._full_time_recorded = False
        self._half_time_recorded = False

    def reset(self, match_config: Mapping[str, object] | None = None) -> dict[str, AgentObservation]:
        if match_config and "max_minutes" in match_config:
            self.max_minutes = int(match_config["max_minutes"])

        self.state = MatchState(
            minute=1,
            possession_team=self._coin_toss(),
            current_zone=PitchZone.MID_CENTER,
        )
        self._home_possession_minutes = 0
        self._away_possession_minutes = 0
        self._full_time_recorded = False
        self._half_time_recorded = False
        self.state.events.append(
            MatchEvent(
                minute=1,
                event_type=EventType.KICKOFF,
                team=self.state.possession_team,
                player_name="",
                zone=PitchZone.MID_CENTER,
                success=True,
                description=f"RL kickoff: {self.state.possession_team} starts with possession",
            )
        )
        return self._observe()

    def step(
        self,
        actions: Mapping[str, ActionType | str] | None,
    ) -> tuple[dict[str, AgentObservation], dict[str, float], bool, dict[str, object]]:
        if self.state.minute <= 0:
            self.reset()
        if self._full_time_recorded:
            return self._observe(), {self.home.name: 0.0, self.away.name: 0.0}, True, {"event": None}

        observations = self._observe()
        possession_team = self._team_by_name(self.state.possession_team)
        defending_team = self.away if possession_team.name == self.home.name else self.home
        side = self._side_for_team(possession_team)
        player = possession_team.get_player_in_zone(self.state.current_zone, self.rng)
        player_index = self._player_index(possession_team, player)
        agent_id = f"{side}:{player_index}"
        observation = observations[agent_id]

        action = self._select_action(actions or {}, observation, possession_team, player_index)
        defensive_action = self._select_defensive_action(actions or {}, observations, defending_team)
        event, progressed = self._resolve_action(action, defensive_action, player, possession_team, defending_team)

        self.state.events.append(event)
        rewards = self.reward_function.score_step(event, possession_team.name, defending_team.name, progressed=progressed)
        self._record_possession_time(possession_team.name, 1)
        self._advance_clock()
        self._update_possession_stats()

        info = {
            "event": event,
            "acting_team": possession_team.name,
            "defending_team": defending_team.name,
            "action": action.value,
            "defensive_action": defensive_action.value,
            "agent_id": agent_id,
            "progressed": progressed,
        }
        return self._observe(), rewards, self._full_time_recorded, info

    def _resolve_action(
        self,
        action: ActionType,
        defensive_action: ActionType,
        player,
        team: Team,
        opponent: Team,
    ) -> tuple[MatchEvent, bool]:
        if action in {ActionType.SAFE_PASS, ActionType.PROGRESSIVE_PASS, ActionType.SWITCH_PLAY, ActionType.HOLD, ActionType.CLEAR}:
            return self._resolve_pass_like(action, defensive_action, player, team, opponent)
        if action == ActionType.DRIBBLE:
            return self._resolve_dribble(defensive_action, player, team, opponent)
        if action == ActionType.SHOOT:
            return self._resolve_shot(defensive_action, player, team, opponent)

        event = MatchEvent(
            minute=self.state.minute,
            event_type=EventType.PASS,
            team=team.name,
            player_name=player.name,
            zone=self.state.current_zone,
            success=True,
            description=f"{player.name} holds team shape",
            extra={"action": action.value},
        )
        return event, False

    def _resolve_pass_like(
        self,
        action: ActionType,
        defensive_action: ActionType,
        player,
        team: Team,
        opponent: Team,
    ) -> tuple[MatchEvent, bool]:
        current_zone = self.state.current_zone
        home_tendency = self.tactical_engine.compute_possession_tendency(self.home, self.away)
        modifier = self.tactical_engine.pass_success_modifier(
            team,
            opponent,
            current_zone,
            home_tendency,
            is_home=team.name == self.home.name,
        )
        action_modifier = {
            ActionType.HOLD: 0.08,
            ActionType.SAFE_PASS: 0.05,
            ActionType.SWITCH_PLAY: -0.02,
            ActionType.PROGRESSIVE_PASS: -0.08,
            ActionType.CLEAR: -0.04,
        }[action]
        defensive_modifier = self._defensive_action_modifier(defensive_action)
        probability = float(np.clip(player.pass_success_rate + modifier + action_modifier - defensive_modifier, 0.25, 0.96))
        success = bool(self.rng.random() < probability)

        if success:
            next_zone = self._next_zone_for_action(current_zone, team, action)
            progressed = self._is_progression(current_zone, next_zone)
            self.state.current_zone = next_zone
            event_type = EventType.PASS
            description = f"{player.name} plays {action.value} from {current_zone.value}"
        else:
            progressed = False
            self.state.current_zone = self._flip_zone(current_zone)
            self.state.possession_team = opponent.name
            event_type = EventType.INTERCEPTION
            description = f"{opponent.name} intercepts {player.name}'s {action.value}"

        return (
            MatchEvent(
                minute=self.state.minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=current_zone,
                success=success,
                description=description,
                extra={"action": action.value, "probability": round(probability, 3)},
            ),
            progressed,
        )

    def _resolve_dribble(
        self,
        defensive_action: ActionType,
        player,
        team: Team,
        opponent: Team,
    ) -> tuple[MatchEvent, bool]:
        current_zone = self.state.current_zone
        pressure = self.tactical_engine.pressure_modifier(opponent, current_zone) * 0.085
        physical_bonus = (player.stats.physical - 70) / 1000
        defensive_modifier = self._defensive_action_modifier(defensive_action)
        probability = float(np.clip(player.dribble_success_rate - pressure + physical_bonus - defensive_modifier, 0.18, 0.88))
        success = bool(self.rng.random() < probability)

        if success:
            next_zone = self.tactical_engine.transition_zone(current_zone, team, "dribble")
            progressed = self._is_progression(current_zone, next_zone)
            self.state.current_zone = next_zone
            event_type = EventType.DRIBBLE
            description = f"{player.name} carries through {current_zone.value}"
        else:
            progressed = False
            self.state.possession_team = opponent.name
            event_type = EventType.TACKLE
            description = f"{player.name} is tackled in {current_zone.value}"

        return (
            MatchEvent(
                minute=self.state.minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=current_zone,
                success=success,
                description=description,
                extra={"action": ActionType.DRIBBLE.value, "probability": round(probability, 3)},
            ),
            progressed,
        )

    def _resolve_shot(
        self,
        defensive_action: ActionType,
        player,
        team: Team,
        opponent: Team,
    ) -> tuple[MatchEvent, bool]:
        current_zone = self.state.current_zone
        xg = self._expected_goal_value(player, team, opponent, current_zone)
        if defensive_action in {ActionType.DROP, ActionType.MARK}:
            xg *= 0.92
        elif defensive_action == ActionType.PRESS:
            xg *= 0.96

        on_target_probability = self._shot_on_target_probability(player, current_zone)
        on_target = bool(self.rng.random() < on_target_probability)
        save_rate = self._goalkeeper_save_rate(opponent)
        goal_probability_if_on_target = float(np.clip((xg / max(on_target_probability, 0.01)) * (0.90 / save_rate), 0.025, 0.55))
        scored = bool(on_target and self.rng.random() < goal_probability_if_on_target)
        blocked = bool(not on_target and self.rng.random() < self._blocked_shot_probability(opponent, current_zone))

        if team.name == self.home.name:
            self.state.home_shots += 1
            self.state.home_xg += xg
            if on_target:
                self.state.home_shots_on_target += 1
            if scored:
                self.state.home_goals += 1
        else:
            self.state.away_shots += 1
            self.state.away_xg += xg
            if on_target:
                self.state.away_shots_on_target += 1
            if scored:
                self.state.away_goals += 1

        if scored:
            event_type = EventType.GOAL
            description = f"{player.name} scores from {current_zone.value}"
        elif on_target:
            event_type = EventType.SAVE
            description = f"{player.name}'s shot is saved"
        elif blocked:
            event_type = EventType.BLOCKED
            description = f"{player.name}'s shot is blocked"
        else:
            event_type = EventType.MISS
            description = f"{player.name} misses from {current_zone.value}"

        self.state.current_zone = PitchZone.MID_CENTER if scored else self._flip_zone(current_zone)
        self.state.possession_team = opponent.name

        return (
            MatchEvent(
                minute=self.state.minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=current_zone,
                success=scored,
                description=description,
                extra={
                    "action": ActionType.SHOOT.value,
                    "xg": round(float(xg), 3),
                    "on_target_probability": round(on_target_probability, 3),
                    "on_target": on_target,
                    "save_rate": round(save_rate, 3),
                },
            ),
            False,
        )

    def _select_action(
        self,
        actions: Mapping[str, ActionType | str],
        observation: AgentObservation,
        team: Team,
        player_index: int,
    ) -> ActionType:
        role_key = f"{observation.side}:{observation.role_group}"
        candidate_keys = (
            observation.agent_id,
            role_key,
            observation.side,
            team.name,
        )
        for key in candidate_keys:
            if key in actions:
                candidate = self.action_space.validate(actions[key])
                if candidate in POSSESSION_ACTIONS:
                    return candidate
                break
        return self.default_policy.select_action(observation, self.action_space, self.rng)

    def _select_defensive_action(
        self,
        actions: Mapping[str, ActionType | str],
        observations: dict[str, AgentObservation],
        defending_team: Team,
    ) -> ActionType:
        side = self._side_for_team(defending_team)
        role_key = f"{side}:defense"
        candidate_keys = (role_key, side, defending_team.name)
        for key in candidate_keys:
            if key in actions:
                candidate = self.action_space.validate(actions[key])
                if candidate in {ActionType.PRESS, ActionType.DROP, ActionType.MARK}:
                    return candidate

        defense_obs = next(obs for obs in observations.values() if obs.side == side and obs.role_group == "defense")
        candidate = self.default_policy.select_action(defense_obs, self.action_space, self.rng)
        return candidate if candidate in {ActionType.PRESS, ActionType.DROP, ActionType.MARK} else ActionType.MARK

    def _next_zone_for_action(self, zone: PitchZone, team: Team, action: ActionType) -> PitchZone:
        if action == ActionType.HOLD:
            return zone
        if action == ActionType.SWITCH_PLAY:
            return self._switch_zone(zone)
        if action == ActionType.CLEAR:
            return PitchZone.MID_CENTER if zone in DEFENSIVE_ZONES else self.tactical_engine.transition_zone(zone, team, "pass")
        if action == ActionType.SAFE_PASS and zone in ATTACKING_ZONES:
            return zone
        return self.tactical_engine.transition_zone(zone, team, "pass")

    def _defensive_action_modifier(self, action: ActionType) -> float:
        return {
            ActionType.PRESS: 0.055,
            ActionType.MARK: 0.035,
            ActionType.DROP: 0.015,
        }.get(action, 0.025)

    def _expected_goal_value(self, player, team: Team, opponent: Team, zone: PitchZone) -> float:
        base_xg = {
            PitchZone.ATT_CENTER: 0.15,
            PitchZone.ATT_LEFT: 0.08,
            PitchZone.ATT_RIGHT: 0.08,
            PitchZone.MID_CENTER: 0.035,
            PitchZone.MID_LEFT: 0.018,
            PitchZone.MID_RIGHT: 0.018,
            PitchZone.DEF_LEFT: 0.006,
            PitchZone.DEF_CENTER: 0.006,
            PitchZone.DEF_RIGHT: 0.006,
        }[zone]
        shooter_factor = 0.72 + (player.stats.shooting / 220)
        matchup_factor = float(np.clip(team.attacking_quality / max(opponent.defensive_quality, 1), 0.82, 1.18))
        style_factor = 1.0
        if player.playstyle == PlayStyle.POACHER and zone == PitchZone.ATT_CENTER:
            style_factor = 1.16
        elif player.playstyle == PlayStyle.COMPLETE_FORWARD:
            style_factor = 1.08
        elif player.playstyle == PlayStyle.TARGET_MAN:
            style_factor = 1.05
        return float(np.clip(base_xg * shooter_factor * matchup_factor * style_factor, 0.003, 0.34))

    def _shot_on_target_probability(self, player, zone: PitchZone) -> float:
        zone_bonus = {
            PitchZone.ATT_CENTER: 0.10,
            PitchZone.ATT_LEFT: 0.04,
            PitchZone.ATT_RIGHT: 0.04,
            PitchZone.MID_CENTER: -0.07,
            PitchZone.MID_LEFT: -0.11,
            PitchZone.MID_RIGHT: -0.11,
            PitchZone.DEF_LEFT: -0.18,
            PitchZone.DEF_CENTER: -0.18,
            PitchZone.DEF_RIGHT: -0.18,
        }[zone]
        probability = 0.18 + (player.stats.shooting / 100) * 0.26 + zone_bonus
        return float(np.clip(probability, 0.06, 0.62))

    def _blocked_shot_probability(self, opponent: Team, zone: PitchZone) -> float:
        defensive_shape = opponent.defensive_quality / 100
        central_bonus = 0.08 if zone == PitchZone.ATT_CENTER else 0.03
        return float(np.clip(0.16 + defensive_shape * 0.18 + central_bonus, 0.16, 0.42))

    def _goalkeeper_save_rate(self, team: Team) -> float:
        goalkeeper = next((player for player in team.starters if player.position == Position.GK), None)
        if goalkeeper is None:
            return 0.70
        return float(np.clip(0.48 + goalkeeper.stats.defending / 250 + goalkeeper.stats.physical / 500, 0.55, 0.86))

    def _advance_clock(self) -> None:
        if self.state.minute == 45 and not self._half_time_recorded and self.max_minutes > 45:
            self._half_time_recorded = True
            self.state.events.append(
                MatchEvent(
                    minute=45,
                    event_type=EventType.HALF_TIME,
                    team="",
                    player_name="",
                    zone=PitchZone.MID_CENTER,
                    success=True,
                    description=f"RL half-time: {self.home.name} {self.state.home_goals} - {self.state.away_goals} {self.away.name}",
                )
            )

        self.state.minute += 1
        if self.state.minute > self.max_minutes:
            self.state.minute = self.max_minutes
            self._full_time_recorded = True
            self.state.events.append(
                MatchEvent(
                    minute=self.max_minutes,
                    event_type=EventType.FULL_TIME,
                    team="",
                    player_name="",
                    zone=PitchZone.MID_CENTER,
                    success=True,
                    description=f"RL full-time: {self.home.name} {self.state.home_goals} - {self.state.away_goals} {self.away.name}",
                )
            )

    def _observe(self) -> dict[str, AgentObservation]:
        return self.observation_builder.build(self.home, self.away, self.state)

    def _team_by_name(self, team_name: str) -> Team:
        return self.home if team_name == self.home.name else self.away

    def _side_for_team(self, team: Team) -> str:
        return "home" if team.name == self.home.name else "away"

    def _player_index(self, team: Team, player) -> int:
        for index, starter in enumerate(team.ordered_starters):
            if starter.name == player.name:
                return index
        return 0

    def _coin_toss(self) -> str:
        return self.home.name if self.rng.random() < 0.5 else self.away.name

    def _record_possession_time(self, team_name: str, duration: int) -> None:
        if team_name == self.home.name:
            self._home_possession_minutes += duration
        else:
            self._away_possession_minutes += duration

    def _update_possession_stats(self) -> None:
        total = self._home_possession_minutes + self._away_possession_minutes
        if total > 0:
            self.state.home_possession_pct = round((self._home_possession_minutes / total) * 100, 1)

    def _flip_zone(self, zone: PitchZone) -> PitchZone:
        mapping = {
            PitchZone.DEF_LEFT: PitchZone.ATT_RIGHT,
            PitchZone.DEF_CENTER: PitchZone.ATT_CENTER,
            PitchZone.DEF_RIGHT: PitchZone.ATT_LEFT,
            PitchZone.MID_LEFT: PitchZone.MID_RIGHT,
            PitchZone.MID_CENTER: PitchZone.MID_CENTER,
            PitchZone.MID_RIGHT: PitchZone.MID_LEFT,
            PitchZone.ATT_LEFT: PitchZone.DEF_RIGHT,
            PitchZone.ATT_CENTER: PitchZone.DEF_CENTER,
            PitchZone.ATT_RIGHT: PitchZone.DEF_LEFT,
        }
        return mapping[zone]

    def _switch_zone(self, zone: PitchZone) -> PitchZone:
        mapping = {
            PitchZone.DEF_LEFT: PitchZone.DEF_RIGHT,
            PitchZone.DEF_CENTER: PitchZone.DEF_CENTER,
            PitchZone.DEF_RIGHT: PitchZone.DEF_LEFT,
            PitchZone.MID_LEFT: PitchZone.MID_RIGHT,
            PitchZone.MID_CENTER: PitchZone.MID_CENTER,
            PitchZone.MID_RIGHT: PitchZone.MID_LEFT,
            PitchZone.ATT_LEFT: PitchZone.ATT_RIGHT,
            PitchZone.ATT_CENTER: PitchZone.ATT_CENTER,
            PitchZone.ATT_RIGHT: PitchZone.ATT_LEFT,
        }
        return mapping[zone]

    def _is_progression(self, before: PitchZone, after: PitchZone) -> bool:
        return ZONE_PROGRESS[after] > ZONE_PROGRESS[before]


DEFENSIVE_ZONES: set[PitchZone] = {
    PitchZone.DEF_LEFT,
    PitchZone.DEF_CENTER,
    PitchZone.DEF_RIGHT,
}

ATTACKING_ZONES: set[PitchZone] = {
    PitchZone.ATT_LEFT,
    PitchZone.ATT_CENTER,
    PitchZone.ATT_RIGHT,
}

ZONE_PROGRESS: dict[PitchZone, int] = {
    PitchZone.DEF_LEFT: 0,
    PitchZone.DEF_CENTER: 0,
    PitchZone.DEF_RIGHT: 0,
    PitchZone.MID_LEFT: 1,
    PitchZone.MID_CENTER: 1,
    PitchZone.MID_RIGHT: 1,
    PitchZone.ATT_LEFT: 2,
    PitchZone.ATT_CENTER: 2,
    PitchZone.ATT_RIGHT: 2,
}
