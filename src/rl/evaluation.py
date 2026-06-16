from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.engine.match import MatchSimulator
from src.models.team import Team
from src.rl.env import FootballEnv
from src.rl.observations import AgentObservation
from src.rl.policies import Policy, RandomPolicy, TacticalPresetPolicy


@dataclass
class EvaluationResult:
    label: str
    episodes: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_goal_diff: float
    avg_xg_diff: float
    avg_goals: float
    avg_shots: float
    avg_shots_on_target: float
    action_distribution: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        return {
            "label": self.label,
            "episodes": self.episodes,
            "win_rate": round(self.win_rate, 6),
            "draw_rate": round(self.draw_rate, 6),
            "loss_rate": round(self.loss_rate, 6),
            "avg_goal_diff": round(self.avg_goal_diff, 6),
            "avg_xg_diff": round(self.avg_xg_diff, 6),
            "avg_goals": round(self.avg_goals, 6),
            "avg_shots": round(self.avg_shots, 6),
            "avg_shots_on_target": round(self.avg_shots_on_target, 6),
            "warnings": ";".join(self.warnings),
            "action_distribution": json.dumps(self.action_distribution, sort_keys=True),
        }


class EvaluationSuite:
    def evaluate_team_policy(
        self,
        env_factory,
        home_policy: Policy,
        away_policy: Policy | None = None,
        episodes: int = 20,
        max_steps: int | None = None,
        label: str = "policy",
    ) -> EvaluationResult:
        away_policy = away_policy or TacticalPresetPolicy()
        wins = 0
        draws = 0
        losses = 0
        goal_diffs: list[int] = []
        xg_diffs: list[float] = []
        goals: list[int] = []
        shots: list[int] = []
        shots_on_target: list[int] = []
        action_counts: dict[str, int] = {}
        total_actions = 0

        for episode_index in range(episodes):
            env = self._make_env(env_factory, episode_index)
            observations = env.reset()
            done = False
            steps = 0

            while not done and (max_steps is None or steps < max_steps):
                home_obs = self._representative_observation(observations, "home")
                away_obs = self._representative_observation(observations, "away")
                actions = {
                    "home": home_policy.select_action(home_obs, env.action_space, env.rng),
                    "away": away_policy.select_action(away_obs, env.action_space, env.rng),
                }
                observations, _, done, info = env.step(actions)
                action_value = str(info["action"])
                action_counts[action_value] = action_counts.get(action_value, 0) + 1
                total_actions += 1
                steps += 1

            home_goals = env.state.home_goals
            away_goals = env.state.away_goals
            wins += int(home_goals > away_goals)
            draws += int(home_goals == away_goals)
            losses += int(home_goals < away_goals)
            goal_diffs.append(home_goals - away_goals)
            xg_diffs.append(env.state.home_xg - env.state.away_xg)
            goals.append(home_goals + away_goals)
            shots.append(env.state.home_shots + env.state.away_shots)
            shots_on_target.append(env.state.home_shots_on_target + env.state.away_shots_on_target)

        distribution = {
            action: count / total_actions for action, count in sorted(action_counts.items()) if total_actions > 0
        }
        result = EvaluationResult(
            label=label,
            episodes=episodes,
            win_rate=wins / episodes,
            draw_rate=draws / episodes,
            loss_rate=losses / episodes,
            avg_goal_diff=float(np.mean(goal_diffs)),
            avg_xg_diff=float(np.mean(xg_diffs)),
            avg_goals=float(np.mean(goals)),
            avg_shots=float(np.mean(shots)),
            avg_shots_on_target=float(np.mean(shots_on_target)),
            action_distribution=distribution,
        )
        result.warnings.extend(detect_policy_warnings(result))
        return result

    def random_baseline(self, env_factory, episodes: int = 20, max_steps: int | None = None) -> EvaluationResult:
        return self.evaluate_team_policy(
            env_factory,
            RandomPolicy(),
            RandomPolicy(),
            episodes=episodes,
            max_steps=max_steps,
            label="random_vs_random",
        )

    def tactical_baseline(self, env_factory, episodes: int = 20, max_steps: int | None = None) -> EvaluationResult:
        policy = TacticalPresetPolicy()
        return self.evaluate_team_policy(
            env_factory,
            policy,
            policy,
            episodes=episodes,
            max_steps=max_steps,
            label="tactical_vs_tactical",
        )

    def rule_based_baseline(
        self,
        home_team: Team,
        away_team: Team,
        episodes: int = 20,
        seed_offset: int = 0,
        label: str = "rule_based_match_simulator",
    ) -> EvaluationResult:
        wins = 0
        draws = 0
        losses = 0
        goal_diffs: list[int] = []
        xg_diffs: list[float] = []
        goals: list[int] = []
        shots: list[int] = []
        shots_on_target: list[int] = []

        for index in range(episodes):
            state = MatchSimulator(home_team, away_team, rng_seed=seed_offset + index).simulate()
            wins += int(state.home_goals > state.away_goals)
            draws += int(state.home_goals == state.away_goals)
            losses += int(state.home_goals < state.away_goals)
            goal_diffs.append(state.home_goals - state.away_goals)
            xg_diffs.append(state.home_xg - state.away_xg)
            goals.append(state.home_goals + state.away_goals)
            shots.append(state.home_shots + state.away_shots)
            shots_on_target.append(state.home_shots_on_target + state.away_shots_on_target)

        return EvaluationResult(
            label=label,
            episodes=episodes,
            win_rate=wins / episodes,
            draw_rate=draws / episodes,
            loss_rate=losses / episodes,
            avg_goal_diff=float(np.mean(goal_diffs)),
            avg_xg_diff=float(np.mean(xg_diffs)),
            avg_goals=float(np.mean(goals)),
            avg_shots=float(np.mean(shots)),
            avg_shots_on_target=float(np.mean(shots_on_target)),
            action_distribution={},
        )

    def _representative_observation(self, observations: dict[str, AgentObservation], side: str) -> AgentObservation:
        possession_candidates = [obs for obs in observations.values() if obs.side == side and obs.has_possession]
        if possession_candidates:
            return possession_candidates[0]
        defense_candidates = [obs for obs in observations.values() if obs.side == side and obs.role_group == "defense"]
        return defense_candidates[0]

    def _make_env(self, env_factory: Callable[..., FootballEnv], episode_index: int) -> FootballEnv:
        try:
            return env_factory(episode_index)
        except TypeError:
            return env_factory()


def write_evaluation_report(
    results: list[EvaluationResult],
    output_dir: str | Path,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "evaluation.json"
    csv_path = output_path / "evaluation.csv"

    payload = {
        "metadata": metadata,
        "results": [
            {
                **result.to_row(),
                "action_distribution": result.action_distribution,
                "warnings": result.warnings,
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = list(results[0].to_row()) if results else ["label"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())

    return json_path, csv_path


def detect_policy_warnings(result: EvaluationResult) -> list[str]:
    warnings: list[str] = []
    shoot_rate = result.action_distribution.get("shoot", 0.0)
    pass_hold_rate = result.action_distribution.get("safe_pass", 0.0) + result.action_distribution.get("hold", 0.0)

    if shoot_rate > 0.38 and result.avg_xg_diff < 0.05:
        warnings.append("possible_shot_spam")
    if pass_hold_rate > 0.82 and result.avg_goals < 1.0:
        warnings.append("possible_stale_possession")
    if result.avg_goals > 5.5:
        warnings.append("high_score_outlier_profile")
    if result.avg_shots_on_target > result.avg_shots:
        warnings.append("invalid_shot_profile")
    return warnings
