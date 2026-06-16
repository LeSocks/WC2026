from __future__ import annotations

from typing import Optional

import numpy as np

from src.engine.events import EventType, MatchEvent, MatchState, PitchZone
from src.engine.tactics import TacticalEngine
from src.models.player import PlayStyle, Position
from src.models.team import Team


class MatchSimulator:
    def __init__(self, home_team: Team, away_team: Team, rng_seed: Optional[int] = None):
        self.home = home_team
        self.away = away_team
        self.rng = np.random.default_rng(rng_seed)
        self.state = MatchState()
        self.tactical_engine = TacticalEngine()
        self._home_possession_minutes = 0
        self._away_possession_minutes = 0
        self._half_time_recorded = False

    def simulate(self) -> MatchState:
        self.state.possession_team = self._coin_toss()
        self.state.minute = 1
        self.state.events.append(
            MatchEvent(
                minute=1,
                event_type=EventType.KICKOFF,
                team=self.state.possession_team,
                player_name="",
                zone=PitchZone.MID_CENTER,
                success=True,
                description=f"Kickoff: {self.state.possession_team} memulai pertandingan",
            )
        )

        home_possession_tendency = self.tactical_engine.compute_possession_tendency(self.home, self.away)

        self._simulate_period(period_end=45, home_possession_tendency=home_possession_tendency)
        self._half_time_recorded = True
        self.state.events.append(
            MatchEvent(
                minute=45,
                event_type=EventType.HALF_TIME,
                team="",
                player_name="",
                zone=PitchZone.MID_CENTER,
                success=True,
                description=f"Babak pertama: {self.home.name} {self.state.home_goals} - {self.state.away_goals} {self.away.name}",
            )
        )

        self.state.minute = 46
        self.state.current_zone = PitchZone.MID_CENTER
        self._simulate_period(period_end=90, home_possession_tendency=home_possession_tendency)

        self.state.minute = 90
        self.state.events.append(
            MatchEvent(
                minute=90,
                event_type=EventType.FULL_TIME,
                team="",
                player_name="",
                zone=PitchZone.MID_CENTER,
                success=True,
                description=f"Pertandingan berakhir: {self.home.name} {self.state.home_goals} - {self.state.away_goals} {self.away.name}",
            )
        )
        return self.state

    def _simulate_period(self, period_end: int, home_possession_tendency: float) -> None:
        while self.state.minute < period_end:
            possession_team = self.state.possession_team
            chain_duration, events = self._simulate_possession_chain(
                possession_team,
                self.state.current_zone,
                home_possession_tendency,
                max_minute=period_end - 1,
            )

            self.state.events.extend(events)
            self._record_possession_time(possession_team, chain_duration)
            self.state.minute = min(period_end, self.state.minute + chain_duration)
            self._update_possession_stats()

    def _simulate_possession_chain(
        self,
        possession_team: str,
        start_zone: PitchZone,
        home_possession_tendency: float,
        max_minute: int,
    ) -> tuple[int, list[MatchEvent]]:
        events: list[MatchEvent] = []
        current_zone = start_zone
        chain_minute = self.state.minute
        team = self._team_by_name(possession_team)
        opponent = self.away if team.name == self.home.name else self.home

        for _ in range(8):
            if chain_minute > max_minute:
                break

            player = team.get_player_in_zone(current_zone, self.rng)
            action = self._choose_action(player, team, current_zone)

            if action == "shoot":
                event, scored = self._resolve_shot(player, team, opponent, current_zone, chain_minute)
                events.append(event)
                if scored:
                    if team.name == self.home.name:
                        self.state.home_goals += 1
                    else:
                        self.state.away_goals += 1
                    self.state.current_zone = PitchZone.MID_CENTER
                    self.state.possession_team = opponent.name
                else:
                    self.state.current_zone = self._flip_zone(current_zone)
                    self.state.possession_team = opponent.name
                break

            if action == "pass":
                event, success = self._resolve_pass(player, team, opponent, current_zone, chain_minute, home_possession_tendency)
                events.append(event)
                if success:
                    current_zone = self.tactical_engine.transition_zone(current_zone, team, "pass")
                    self.state.current_zone = current_zone
                else:
                    self.state.possession_team = opponent.name
                    self.state.current_zone = self._flip_zone(current_zone)
                    break

            if action == "dribble":
                event, success = self._resolve_dribble(player, team, opponent, current_zone, chain_minute)
                events.append(event)
                if success:
                    current_zone = self.tactical_engine.transition_zone(current_zone, team, "dribble")
                    self.state.current_zone = current_zone
                else:
                    self.state.possession_team = opponent.name
                    self.state.current_zone = current_zone
                    break

            chain_minute += int(self.rng.integers(1, 3))

        chain_duration = max(1, chain_minute - self.state.minute)
        return chain_duration, events

    def _choose_action(self, player, team: Team, zone: PitchZone) -> str:
        probs = self.tactical_engine.action_probabilities(player, team, zone)
        actions = list(probs)
        weights = [probs[action] for action in actions]
        return str(self.rng.choice(actions, p=weights))

    def _resolve_pass(
        self,
        player,
        team: Team,
        opponent: Team,
        zone: PitchZone,
        minute: int,
        home_possession_tendency: float,
    ) -> tuple[MatchEvent, bool]:
        modifier = self.tactical_engine.pass_success_modifier(
            team,
            opponent,
            zone,
            home_possession_tendency,
            is_home=team.name == self.home.name,
        )
        probability = player.pass_success_rate + modifier
        probability = float(np.clip(probability, 0.35, 0.95))
        success = bool(self.rng.random() < probability)
        event_type = EventType.PASS if success else EventType.INTERCEPTION
        description = (
            f"{player.name} mengalirkan bola dari {zone.value}"
            if success
            else f"Umpan {player.name} dipotong {opponent.name}"
        )
        return (
            MatchEvent(
                minute=minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=zone,
                success=success,
                description=description,
                extra={"probability": round(probability, 3)},
            ),
            success,
        )

    def _resolve_dribble(self, player, team: Team, opponent: Team, zone: PitchZone, minute: int) -> tuple[MatchEvent, bool]:
        pressure = self.tactical_engine.pressure_modifier(opponent, zone) * 0.085
        physical_bonus = (player.stats.physical - 70) / 1000
        probability = float(np.clip(player.dribble_success_rate - pressure + physical_bonus, 0.25, 0.86))
        success = bool(self.rng.random() < probability)
        event_type = EventType.DRIBBLE if success else EventType.TACKLE
        description = (
            f"{player.name} melewati tekanan di {zone.value}"
            if success
            else f"{player.name} kehilangan bola saat dribble"
        )
        return (
            MatchEvent(
                minute=minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=zone,
                success=success,
                description=description,
                extra={"probability": round(probability, 3)},
            ),
            success,
        )

    def _resolve_shot(self, player, team: Team, opponent: Team, zone: PitchZone, minute: int) -> tuple[MatchEvent, bool]:
        xg = self._expected_goal_value(player, team, opponent, zone)
        on_target_probability = self._shot_on_target_probability(player, zone)
        on_target = bool(self.rng.random() < on_target_probability)
        save_rate = self._goalkeeper_save_rate(opponent)
        goal_probability_if_on_target = float(np.clip((xg / max(on_target_probability, 0.01)) * (0.90 / save_rate), 0.03, 0.55))
        scored = bool(on_target and self.rng.random() < goal_probability_if_on_target)
        blocked = bool(not on_target and self.rng.random() < self._blocked_shot_probability(opponent, zone))

        if team.name == self.home.name:
            self.state.home_shots += 1
            self.state.home_xg += xg
            if on_target:
                self.state.home_shots_on_target += 1
        else:
            self.state.away_shots += 1
            self.state.away_xg += xg
            if on_target:
                self.state.away_shots_on_target += 1

        if scored:
            event_type = EventType.GOAL
            description = f"{player.name} mencetak gol!"
        elif on_target:
            event_type = EventType.SAVE
            description = f"Tembakan {player.name} diselamatkan"
        elif blocked:
            event_type = EventType.BLOCKED
            description = f"Tembakan {player.name} diblok"
        else:
            event_type = EventType.MISS
            description = f"Tembakan {player.name} melenceng"

        return (
            MatchEvent(
                minute=minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=zone,
                success=scored,
                description=description,
                extra={
                    "xg": round(xg, 3),
                    "on_target_probability": round(on_target_probability, 3),
                    "on_target": on_target,
                    "save_rate": round(save_rate, 3),
                },
            ),
            scored,
        )

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

    def _goalkeeper_save_rate(self, team: Team) -> float:
        goalkeeper = next((player for player in team.starters if player.position == Position.GK), None)
        if goalkeeper is None:
            return 0.70
        return float(np.clip(0.48 + goalkeeper.stats.defending / 250 + goalkeeper.stats.physical / 500, 0.55, 0.86))

    def _coin_toss(self) -> str:
        return self.home.name if self.rng.random() < 0.5 else self.away.name

    def _team_by_name(self, team_name: str) -> Team:
        return self.home if team_name == self.home.name else self.away

    def _record_possession_time(self, team_name: str, duration: int) -> None:
        if team_name == self.home.name:
            self._home_possession_minutes += duration
        else:
            self._away_possession_minutes += duration

    def _update_possession_stats(self) -> None:
        total = self._home_possession_minutes + self._away_possession_minutes
        if total > 0:
            self.state.home_possession_pct = round((self._home_possession_minutes / total) * 100, 1)
