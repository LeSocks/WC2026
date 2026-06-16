from __future__ import annotations

from dataclasses import dataclass, field

from src.rl.env import FootballEnv
from src.rl.policies import Policy, TacticalPresetPolicy
from src.rl.trainer import TabularQTrainer, TrainingResult


@dataclass
class SelfPlayResult:
    rounds: int
    training_results: list[TrainingResult] = field(default_factory=list)


@dataclass
class SelfPlayRunner:
    trainer: TabularQTrainer = field(default_factory=TabularQTrainer)

    def run(
        self,
        env_factory,
        rounds: int = 3,
        episodes_per_round: int = 5,
        max_steps: int | None = None,
    ) -> tuple[Policy, SelfPlayResult]:
        current_policy: Policy = TacticalPresetPolicy()
        result = SelfPlayResult(rounds=rounds)

        for _ in range(rounds):
            env: FootballEnv = env_factory()
            env.default_policy = current_policy
            current_policy, training_result = self.trainer.train_team_controller(
                env,
                episodes=episodes_per_round,
                max_steps=max_steps,
            )
            result.training_results.append(training_result)

        return current_policy, result
