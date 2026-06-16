from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.data.loader import load_team
from src.rl.env import FootballEnv
from src.rl.evaluation import EvaluationResult, EvaluationSuite, write_evaluation_report
from src.rl.logging import load_checkpoint, slugify
from src.rl.policies import RandomPolicy, TacticalPresetPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy against football baselines.")
    parser.add_argument("home", nargs="?", default="France")
    parser.add_argument("away", nargs="?", default="Morocco")
    parser.add_argument("--checkpoint", help="Optional checkpoint.json from train_controller.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-minutes", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    home_team = load_team(args.home)
    away_team = load_team(args.away)
    suite = EvaluationSuite()

    def env_factory(index: int) -> FootballEnv:
        return FootballEnv(home_team, away_team, rng_seed=args.seed + index, max_minutes=args.max_minutes)

    results: list[EvaluationResult] = [
        suite.random_baseline(env_factory, episodes=args.episodes, max_steps=args.max_steps),
        suite.tactical_baseline(env_factory, episodes=args.episodes, max_steps=args.max_steps),
        suite.rule_based_baseline(home_team, away_team, episodes=args.episodes, seed_offset=args.seed),
    ]

    checkpoint_episode = None
    if args.checkpoint:
        learned_policy, checkpoint_episode = load_checkpoint(args.checkpoint)
        if hasattr(learned_policy, "set_epsilon"):
            learned_policy.set_epsilon(args.eval_epsilon)
        else:
            learned_policy.epsilon = args.eval_epsilon
        tactical = TacticalPresetPolicy()
        random = RandomPolicy()
        results.extend(
            [
                suite.evaluate_team_policy(
                    env_factory,
                    home_policy=learned_policy,
                    away_policy=tactical,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    label="learned_vs_tactical",
                ),
                suite.evaluate_team_policy(
                    env_factory,
                    home_policy=learned_policy,
                    away_policy=random,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    label="learned_vs_random",
                ),
                suite.evaluate_team_policy(
                    env_factory,
                    home_policy=learned_policy,
                    away_policy=learned_policy,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    label="learned_self_play",
                ),
            ]
        )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.home, args.away)
    metadata = {
        "home": args.home,
        "away": args.away,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_minutes": args.max_minutes,
        "max_steps": args.max_steps,
        "checkpoint": args.checkpoint,
        "checkpoint_episode": checkpoint_episode,
        "eval_epsilon": args.eval_epsilon,
    }
    json_path, csv_path = write_evaluation_report(results, output_dir, metadata)

    if not args.quiet:
        print(format_results_table(results))
    print(f"Evaluation files: {json_path}, {csv_path}")


def default_output_dir(home: str, away: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("evaluation_runs") / f"{timestamp}_{slugify(home)}_vs_{slugify(away)}"


def format_results_table(results: list[EvaluationResult]) -> str:
    lines = [
        "label                  win   draw  loss  gd    xgd   goals shots  warnings",
        "---------------------  ----  ----  ----  ----  ----  ----- -----  --------",
    ]
    for result in results:
        warnings = ",".join(result.warnings) if result.warnings else "-"
        lines.append(
            f"{result.label[:21]:<21}  "
            f"{result.win_rate:>4.0%}  "
            f"{result.draw_rate:>4.0%}  "
            f"{result.loss_rate:>4.0%}  "
            f"{result.avg_goal_diff:>4.2f}  "
            f"{result.avg_xg_diff:>4.2f}  "
            f"{result.avg_goals:>5.2f} "
            f"{result.avg_shots:>5.2f}  "
            f"{warnings}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
