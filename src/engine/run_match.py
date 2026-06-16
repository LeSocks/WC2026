from __future__ import annotations

import argparse

from src.data.loader import load_team
from src.engine.events import EventType
from src.engine.match import MatchSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one WC2026 tactical simulation match.")
    parser.add_argument("home", nargs="?", default="France")
    parser.add_argument("away", nargs="?", default="Morocco")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    simulator = MatchSimulator(load_team(args.home), load_team(args.away), rng_seed=args.seed)
    state = simulator.simulate()

    print(f"{args.home} {state.home_goals} - {state.away_goals} {args.away}")
    print(
        f"Shots: {state.home_shots} ({state.home_shots_on_target} on target) - "
        f"{state.away_shots} ({state.away_shots_on_target} on target)"
    )
    print(f"Home possession: {state.home_possession_pct:.1f}%")
    print("Key events:")
    for event in state.events:
        if event.event_type in {EventType.KICKOFF, EventType.GOAL, EventType.SAVE, EventType.MISS, EventType.HALF_TIME, EventType.FULL_TIME}:
            print(f"{event.minute:>2}' {event.description}")


if __name__ == "__main__":
    main()
