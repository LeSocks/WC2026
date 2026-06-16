from __future__ import annotations

from typing import Optional

import numpy as np

from src.engine.events import EventType, MatchEvent, MatchState, PitchZone
from src.engine.tactics import AttackStyle, PressStyle
from src.models.player import PlayStyle, Position
from src.models.team import Team


class MatchSimulator:
    def __init__(self, home_team: Team, away_team: Team, rng_seed: Optional[int] = None):
        self.home = home_team
        self.away = away_team
        self.rng = np.random.default_rng(rng_seed)
        self.state = MatchState()
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

        home_possession_tendency = self._compute_possession_tendency()

        while self.state.minute < 90:
            chain_duration, events = self._simulate_possession_chain(
                self.state.possession_team,
                self.state.current_zone,
                home_possession_tendency,
            )

            self.state.events.extend(events)
            self._record_possession_time(self.state.possession_team, chain_duration)
            self.state.minute = min(90, self.state.minute + chain_duration)
            self._update_possession_stats()

            if self.state.minute >= 45 and not self._half_time_recorded:
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

    def _simulate_possession_chain(
        self,
        possession_team: str,
        start_zone: PitchZone,
        home_possession_tendency: float,
    ) -> tuple[int, list[MatchEvent]]:
        events: list[MatchEvent] = []
        current_zone = start_zone
        chain_minute = self.state.minute
        team = self._team_by_name(possession_team)
        opponent = self.away if team.name == self.home.name else self.home

        for _ in range(8):
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
                    current_zone = self._advance_zone(current_zone, team)
                    self.state.current_zone = current_zone
                else:
                    self.state.possession_team = opponent.name
                    self.state.current_zone = self._flip_zone(current_zone)
                    break

            if action == "dribble":
                event, success = self._resolve_dribble(player, team, opponent, current_zone, chain_minute)
                events.append(event)
                if success:
                    current_zone = self._advance_zone(current_zone, team)
                    self.state.current_zone = current_zone
                else:
                    self.state.possession_team = opponent.name
                    self.state.current_zone = current_zone
                    break

            chain_minute += int(self.rng.integers(1, 3))

        chain_duration = max(1, chain_minute - self.state.minute)
        return chain_duration, events

    def _choose_action(self, player, team: Team, zone: PitchZone) -> str:
        zone_probs = {
            PitchZone.DEF_LEFT: {"pass": 0.82, "dribble": 0.15, "shoot": 0.03},
            PitchZone.DEF_CENTER: {"pass": 0.88, "dribble": 0.10, "shoot": 0.02},
            PitchZone.DEF_RIGHT: {"pass": 0.82, "dribble": 0.15, "shoot": 0.03},
            PitchZone.MID_LEFT: {"pass": 0.68, "dribble": 0.25, "shoot": 0.07},
            PitchZone.MID_CENTER: {"pass": 0.68, "dribble": 0.22, "shoot": 0.10},
            PitchZone.MID_RIGHT: {"pass": 0.68, "dribble": 0.25, "shoot": 0.07},
            PitchZone.ATT_LEFT: {"pass": 0.46, "dribble": 0.30, "shoot": 0.24},
            PitchZone.ATT_CENTER: {"pass": 0.36, "dribble": 0.22, "shoot": 0.42},
            PitchZone.ATT_RIGHT: {"pass": 0.46, "dribble": 0.30, "shoot": 0.24},
        }
        probs = dict(zone_probs[zone])

        if player.playstyle == PlayStyle.POACHER and zone == PitchZone.ATT_CENTER:
            probs["shoot"] *= 1.35
        elif player.playstyle == PlayStyle.DRIBBLER:
            probs["dribble"] *= 1.25
        elif player.playstyle in {PlayStyle.DEEP_LYING_PM, PlayStyle.CREATOR}:
            probs["pass"] *= 1.18

        if team.tactical_config.attack_style == AttackStyle.DIRECT:
            probs["pass"] *= 0.88
            probs["shoot"] *= 1.18
        elif team.tactical_config.attack_style == AttackStyle.POSSESSION:
            probs["pass"] *= 1.18
            probs["shoot"] *= 0.90
        elif team.tactical_config.attack_style == AttackStyle.COUNTER_ATTACK:
            probs["dribble"] *= 1.10
            probs["shoot"] *= 1.08

        total = sum(probs.values())
        actions = list(probs)
        weights = [probs[action] / total for action in actions]
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
        opponent_pressure = opponent.tactical_config.pressing_intensity * 0.12
        possession_bonus = (home_possession_tendency - 0.5) * 0.08 if team.name == self.home.name else (0.5 - home_possession_tendency) * 0.08
        probability = player.pass_success_rate - opponent_pressure + possession_bonus
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
        pressure = opponent.tactical_config.pressing_intensity * 0.10
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
        zone_modifier = {
            PitchZone.ATT_CENTER: 1.14,
            PitchZone.ATT_LEFT: 0.92,
            PitchZone.ATT_RIGHT: 0.92,
            PitchZone.MID_CENTER: 0.62,
        }.get(zone, 0.55)
        shot_quality = float(np.clip(player.shot_accuracy_mean * zone_modifier, 0.05, 0.86))
        on_target = bool(self.rng.random() < shot_quality)
        save_rate = self._goalkeeper_save_rate(opponent)
        scored = bool(on_target and self.rng.random() > save_rate)

        if team.name == self.home.name:
            self.state.home_shots += 1
            if on_target:
                self.state.home_shots_on_target += 1
        else:
            self.state.away_shots += 1
            if on_target:
                self.state.away_shots_on_target += 1

        event_type = EventType.GOAL if scored else EventType.SAVE if on_target else EventType.MISS
        description = f"{player.name} mencetak gol!" if scored else f"Tembakan {player.name} diselamatkan" if on_target else f"Tembakan {player.name} melenceng"
        return (
            MatchEvent(
                minute=minute,
                event_type=event_type,
                team=team.name,
                player_name=player.name,
                zone=zone,
                success=scored,
                description=description,
                extra={"shot_quality": round(shot_quality, 3), "on_target": on_target},
            ),
            scored,
        )

    def _compute_possession_tendency(self) -> float:
        home_quality = self.home.passing_quality + (self.home.tactical_config.tempo * 6)
        away_quality = self.away.passing_quality + (self.away.tactical_config.tempo * 6)

        if self.home.tactical_config.attack_style == AttackStyle.POSSESSION:
            home_quality += 4
        if self.away.tactical_config.attack_style == AttackStyle.POSSESSION:
            away_quality += 4
        if self.home.tactical_config.press_style in {PressStyle.HIGH_PRESS, PressStyle.COUNTER_PRESS}:
            home_quality += 2
        if self.away.tactical_config.press_style in {PressStyle.HIGH_PRESS, PressStyle.COUNTER_PRESS}:
            away_quality += 2

        return float(np.clip(home_quality / (home_quality + away_quality), 0.35, 0.65))

    def _advance_zone(self, zone: PitchZone, team: Team) -> PitchZone:
        direct = team.tactical_config.attack_style in {AttackStyle.DIRECT, AttackStyle.COUNTER_ATTACK}
        transitions = {
            PitchZone.DEF_LEFT: PitchZone.MID_LEFT,
            PitchZone.DEF_CENTER: PitchZone.MID_CENTER,
            PitchZone.DEF_RIGHT: PitchZone.MID_RIGHT,
            PitchZone.MID_LEFT: PitchZone.ATT_LEFT if not direct else PitchZone.ATT_CENTER,
            PitchZone.MID_CENTER: PitchZone.ATT_CENTER,
            PitchZone.MID_RIGHT: PitchZone.ATT_RIGHT if not direct else PitchZone.ATT_CENTER,
            PitchZone.ATT_LEFT: PitchZone.ATT_CENTER,
            PitchZone.ATT_CENTER: PitchZone.ATT_CENTER,
            PitchZone.ATT_RIGHT: PitchZone.ATT_CENTER,
        }
        return transitions[zone]

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
