from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_team
from src.rl.env import FootballEnv
from src.rl.logging import TrainingLogger, create_run_dir
from src.rl.trainer import TabularQTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train observable defense/midfield/attack role-group policies.")
    parser.add_argument("home", nargs="?", default="France")
    parser.add_argument("away", nargs="?", default="Morocco")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=int, default=90)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--discount", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--runs-dir", default="training_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default=None, help="Path to a role-group checkpoint.json to resume from.")
    parser.add_argument("--quiet", action="store_true", help="Write files without printing episode summaries.")
    args = parser.parse_args()

    run_dir = Path(args.resume).parent if args.resume else create_run_dir(args.runs_dir, args.home, args.away, args.run_id)
    config = {
        "home": args.home,
        "away": args.away,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_minutes": args.max_minutes,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "epsilon": args.epsilon,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "resume": args.resume,
        "mode": "role_group_tabular_q",
        "roles": ["defense", "midfield", "attack"],
    }
    logger = TrainingLogger(
        run_dir=run_dir,
        config=config,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        echo_terminal=not args.quiet,
    )

    env = FootballEnv(load_team(args.home), load_team(args.away), rng_seed=args.seed, max_minutes=args.max_minutes)
    trainer = TabularQTrainer(
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
    )
    policy, result = trainer.train_role_groups(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        logger=logger,
        resume_checkpoint=args.resume,
    )

    print(
        f"Role-group training complete: episodes={result.completed_episodes}, "
        f"mean_reward={result.mean_reward:.4f}, q_states={policy.q_state_count}"
    )
    print(f"Run files: {run_dir}")


if __name__ == "__main__":
    main()
