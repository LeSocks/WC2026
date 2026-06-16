from __future__ import annotations

from dataclasses import dataclass

from src.engine.events import EventType, MatchEvent, PitchZone


@dataclass(frozen=True)
class RewardFunction:
    goal_reward: float = 1.0
    shot_on_target_reward: float = 0.08
    xg_reward_scale: float = 0.40
    progression_reward: float = 0.025
    pressure_reward: float = 0.035
    defensive_turnover_penalty: float = -0.06
    low_quality_shot_penalty: float = -0.03

    def score_step(
        self,
        event: MatchEvent,
        acting_team: str,
        defending_team: str,
        progressed: bool = False,
    ) -> dict[str, float]:
        rewards = {acting_team: 0.0, defending_team: 0.0}
        xg = float(event.extra.get("xg", 0.0))

        if event.event_type == EventType.GOAL:
            rewards[acting_team] += self.goal_reward + (xg * self.xg_reward_scale)
            rewards[defending_team] -= self.goal_reward
        elif event.event_type in {EventType.SAVE, EventType.MISS, EventType.BLOCKED}:
            rewards[acting_team] += xg * self.xg_reward_scale
            if bool(event.extra.get("on_target", False)):
                rewards[acting_team] += self.shot_on_target_reward
            if xg < 0.03:
                rewards[acting_team] += self.low_quality_shot_penalty
        elif event.event_type in {EventType.INTERCEPTION, EventType.TACKLE}:
            rewards[acting_team] += self.defensive_turnover_penalty if event.zone in DEFENSIVE_ZONES else -0.02
            rewards[defending_team] += self.pressure_reward
        elif event.event_type == EventType.PASS and progressed:
            rewards[acting_team] += self.progression_reward
        elif event.event_type == EventType.DRIBBLE and progressed:
            rewards[acting_team] += self.progression_reward * 1.3

        return rewards


DEFENSIVE_ZONES: set[PitchZone] = {
    PitchZone.DEF_LEFT,
    PitchZone.DEF_CENTER,
    PitchZone.DEF_RIGHT,
}
