from __future__ import annotations

import argparse

from src.data.loader import load_team
from src.rl.env import FootballEnv
from src.rl.observations import AgentObservation
from src.rl.policies import TacticalPresetPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one lightweight RL football environment episode.")
    parser.add_argument("home", nargs="?", default="France")
    parser.add_argument("away", nargs="?", default="Morocco")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    env = FootballEnv(load_team(args.home), load_team(args.away), rng_seed=args.seed)
    policy = TacticalPresetPolicy()
    observations = env.reset()
    done = False

    for _ in range(args.steps):
        if done:
            break
        home_obs = _representative_observation(observations, "home")
        away_obs = _representative_observation(observations, "away")
        observations, rewards, done, info = env.step(
            {
                "home": policy.select_action(home_obs, env.action_space, env.rng),
                "away": policy.select_action(away_obs, env.action_space, env.rng),
            }
        )
        event = info["event"]
        print(f"{event.minute:>2}' {info['acting_team']} {info['action']}: {event.description} | rewards={rewards}")

    print(f"{args.home} {env.state.home_goals} - {env.state.away_goals} {args.away}")
    print(f"xG: {env.state.home_xg:.2f} - {env.state.away_xg:.2f}")


def _representative_observation(observations: dict[str, AgentObservation], side: str) -> AgentObservation:
    possession = [obs for obs in observations.values() if obs.side == side and obs.has_possession]
    if possession:
        return possession[0]
    return next(obs for obs in observations.values() if obs.side == side and obs.role_group == "defense")


if __name__ == "__main__":
    main()
