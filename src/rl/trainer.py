from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.engine.events import EventType
from src.rl.actions import ActionSpace, ActionType, DEFENSIVE_ACTIONS, POSSESSION_ACTIONS
from src.rl.env import FootballEnv
from src.rl.logging import EpisodeMetrics, TrainingLogger, detect_reward_hacking, load_checkpoint
from src.rl.observations import AgentObservation
from src.rl.policies import EpsilonGreedyTabularPolicy, RoleGroupPolicy, canonical_role_group


class EpisodeCallback(Protocol):
    def __call__(
        self,
        metrics: EpisodeMetrics,
        policy: EpsilonGreedyTabularPolicy,
        total_episodes: int,
    ) -> None:
        ...


@dataclass
class TrainingResult:
    episodes: int
    total_rewards: list[float] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    episode_metrics: list[EpisodeMetrics] = field(default_factory=list)
    start_episode: int = 0
    completed_episodes: int = 0

    @property
    def mean_reward(self) -> float:
        if not self.total_rewards:
            return 0.0
        return float(sum(self.total_rewards) / len(self.total_rewards))


@dataclass
class TabularQTrainer:
    learning_rate: float = 0.12
    discount: float = 0.92
    epsilon: float = 0.15

    def train_team_controller(
        self,
        env: FootballEnv,
        episodes: int = 10,
        max_steps: int | None = None,
        logger: TrainingLogger | None = None,
        callback: EpisodeCallback | None = None,
        resume_checkpoint: str | Path | None = None,
        policy: EpsilonGreedyTabularPolicy | None = None,
    ) -> tuple[EpsilonGreedyTabularPolicy, TrainingResult]:
        start_episode = 0
        if resume_checkpoint is not None:
            policy, start_episode = load_checkpoint(resume_checkpoint)
            policy.epsilon = self.epsilon
        elif policy is None:
            policy = EpsilonGreedyTabularPolicy(epsilon=self.epsilon)

        result = TrainingResult(episodes=episodes, start_episode=start_episode)
        rng = env.rng

        for episode in range(start_episode + 1, episodes + 1):
            observations = env.reset()
            done = False
            episode_reward = 0.0
            episode_action_counts: dict[str, int] = {}
            turnovers = 0
            low_quality_shots = 0
            progressions = 0
            steps = 0

            while not done and (max_steps is None or steps < max_steps):
                home_obs = self._representative_observation(observations, "home")
                away_obs = self._representative_observation(observations, "away")
                home_action = policy.select_action(home_obs, env.action_space, rng)
                away_action = policy.select_action(away_obs, env.action_space, rng)
                next_observations, rewards, done, info = env.step({"home": home_action, "away": away_action})

                reward = rewards.get(env.home.name, 0.0) - rewards.get(env.away.name, 0.0)
                next_home_obs = self._representative_observation(next_observations, "home")
                self._update_q_value(policy, home_obs, home_action, reward, next_home_obs, env.action_space)
                episode_reward += reward
                action_value = str(info["action"])
                result.action_counts[action_value] = result.action_counts.get(action_value, 0) + 1
                episode_action_counts[action_value] = episode_action_counts.get(action_value, 0) + 1

                event = info["event"]
                if event.event_type in {EventType.INTERCEPTION, EventType.TACKLE}:
                    turnovers += 1
                if event.event_type in {EventType.SAVE, EventType.MISS, EventType.BLOCKED} and float(event.extra.get("xg", 0.0)) < 0.03:
                    low_quality_shots += 1
                if bool(info.get("progressed", False)):
                    progressions += 1

                observations = next_observations
                steps += 1

            result.total_rewards.append(episode_reward)
            reward_ma_20 = float(sum(result.total_rewards[-20:]) / len(result.total_rewards[-20:]))
            metrics = EpisodeMetrics(
                episode=episode,
                episode_reward=episode_reward,
                reward_ma_20=reward_ma_20,
                home_goals=env.state.home_goals,
                away_goals=env.state.away_goals,
                home_xg=env.state.home_xg,
                away_xg=env.state.away_xg,
                shots=env.state.home_shots + env.state.away_shots,
                shots_on_target=env.state.home_shots_on_target + env.state.away_shots_on_target,
                action_counts=episode_action_counts,
                turnovers=turnovers,
                low_quality_shots=low_quality_shots,
                progressions=progressions,
                epsilon=policy.epsilon,
                q_states=len(policy.q_values),
                elapsed_seconds=0.0,
            )
            metrics = EpisodeMetrics(
                **{**metrics.__dict__, "warnings": detect_reward_hacking(metrics)}
            )
            result.episode_metrics.append(metrics)
            result.completed_episodes = episode

            if logger is not None:
                elapsed_metrics = EpisodeMetrics(
                    **{**metrics.__dict__, "elapsed_seconds": time.monotonic() - logger.started_at}
                )
                logger.log_episode(elapsed_metrics, policy, total_episodes=episodes)
                result.episode_metrics[-1] = elapsed_metrics

            if callback is not None:
                callback(result.episode_metrics[-1], policy, episodes)

        return policy, result

    def train_role_groups(
        self,
        env: FootballEnv,
        episodes: int = 10,
        max_steps: int | None = None,
        logger: TrainingLogger | None = None,
        resume_checkpoint: str | Path | None = None,
        policy: RoleGroupPolicy | None = None,
    ) -> tuple[RoleGroupPolicy, TrainingResult]:
        start_episode = 0
        if resume_checkpoint is not None:
            loaded_policy, start_episode = load_checkpoint(resume_checkpoint)
            if not isinstance(loaded_policy, RoleGroupPolicy):
                raise ValueError("resume_checkpoint must contain a role_group policy")
            policy = loaded_policy
            policy.set_epsilon(self.epsilon)
        elif policy is None:
            policy = RoleGroupPolicy(
                role_policies={
                    role: EpsilonGreedyTabularPolicy(epsilon=self.epsilon)
                    for role in ROLE_TRAINING_GROUPS
                }
            )

        result = TrainingResult(episodes=episodes, start_episode=start_episode)
        rng = env.rng

        for episode in range(start_episode + 1, episodes + 1):
            observations = env.reset()
            done = False
            episode_reward = 0.0
            episode_action_counts: dict[str, int] = {}
            turnovers = 0
            low_quality_shots = 0
            progressions = 0
            steps = 0

            while not done and (max_steps is None or steps < max_steps):
                action_inputs: dict[str, ActionType] = {}
                selected: list[tuple[AgentObservation, ActionType, str]] = []

                for side in ("home", "away"):
                    for role in ROLE_TRAINING_GROUPS:
                        observation = self._representative_role_observation(observations, side, role)
                        action = policy.policy_for_role(role).select_action(observation, env.action_space, rng)
                        action_inputs[f"{side}:{role}"] = action
                        selected.append((observation, action, side))
                    action_inputs[f"{side}:goalkeeper"] = action_inputs[f"{side}:defense"]

                next_observations, rewards, done, info = env.step(action_inputs)
                reward = rewards.get(env.home.name, 0.0) - rewards.get(env.away.name, 0.0)
                episode_reward += reward

                for observation, action, side in selected:
                    role = canonical_role_group(observation.role_group)
                    next_observation = self._representative_role_observation(next_observations, side, role)
                    update_reward = reward if side == "home" else -reward
                    self._update_q_value(
                        policy.policy_for_role(role),
                        observation,
                        action,
                        update_reward,
                        next_observation,
                        env.action_space,
                    )

                action_value = str(info["action"])
                result.action_counts[action_value] = result.action_counts.get(action_value, 0) + 1
                episode_action_counts[action_value] = episode_action_counts.get(action_value, 0) + 1

                event = info["event"]
                if event.event_type in {EventType.INTERCEPTION, EventType.TACKLE}:
                    turnovers += 1
                if event.event_type in {EventType.SAVE, EventType.MISS, EventType.BLOCKED} and float(event.extra.get("xg", 0.0)) < 0.03:
                    low_quality_shots += 1
                if bool(info.get("progressed", False)):
                    progressions += 1

                observations = next_observations
                steps += 1

            result.total_rewards.append(episode_reward)
            reward_ma_20 = float(sum(result.total_rewards[-20:]) / len(result.total_rewards[-20:]))
            metrics = EpisodeMetrics(
                episode=episode,
                episode_reward=episode_reward,
                reward_ma_20=reward_ma_20,
                home_goals=env.state.home_goals,
                away_goals=env.state.away_goals,
                home_xg=env.state.home_xg,
                away_xg=env.state.away_xg,
                shots=env.state.home_shots + env.state.away_shots,
                shots_on_target=env.state.home_shots_on_target + env.state.away_shots_on_target,
                action_counts=episode_action_counts,
                turnovers=turnovers,
                low_quality_shots=low_quality_shots,
                progressions=progressions,
                epsilon=policy.epsilon,
                q_states=policy.q_state_count,
                elapsed_seconds=0.0,
            )
            metrics = EpisodeMetrics(
                **{**metrics.__dict__, "warnings": detect_reward_hacking(metrics)}
            )
            result.episode_metrics.append(metrics)
            result.completed_episodes = episode

            if logger is not None:
                elapsed_metrics = EpisodeMetrics(
                    **{**metrics.__dict__, "elapsed_seconds": time.monotonic() - logger.started_at}
                )
                logger.log_episode(elapsed_metrics, policy, total_episodes=episodes)
                result.episode_metrics[-1] = elapsed_metrics

        return policy, result

    def _update_q_value(
        self,
        policy: EpsilonGreedyTabularPolicy,
        observation: AgentObservation,
        action: ActionType,
        reward: float,
        next_observation: AgentObservation,
        action_space: ActionSpace,
    ) -> None:
        state_key = observation.state_key()
        next_key = next_observation.state_key()
        state_values = policy.q_values.setdefault(state_key, {})
        current = state_values.get(action, 0.0)
        available = POSSESSION_ACTIONS if next_observation.has_possession else DEFENSIVE_ACTIONS
        next_values = policy.q_values.get(next_key, {})
        bootstrap = max((next_values.get(next_action, 0.0) for next_action in available), default=0.0)
        state_values[action] = current + self.learning_rate * (reward + self.discount * bootstrap - current)

    def _representative_observation(self, observations: dict[str, AgentObservation], side: str) -> AgentObservation:
        possession_candidates = [obs for obs in observations.values() if obs.side == side and obs.has_possession]
        if possession_candidates:
            return possession_candidates[0]
        defense_candidates = [obs for obs in observations.values() if obs.side == side and obs.role_group == "defense"]
        return defense_candidates[0]

    def _representative_role_observation(
        self,
        observations: dict[str, AgentObservation],
        side: str,
        role_group: str,
    ) -> AgentObservation:
        canonical = canonical_role_group(role_group)
        candidates = [
            obs
            for obs in observations.values()
            if obs.side == side and canonical_role_group(obs.role_group) == canonical
        ]
        possession_candidates = [obs for obs in candidates if obs.has_possession]
        if possession_candidates:
            return possession_candidates[0]
        if candidates:
            return candidates[0]
        return self._representative_observation(observations, side)


ROLE_TRAINING_GROUPS: tuple[str, ...] = ("defense", "midfield", "attack")
