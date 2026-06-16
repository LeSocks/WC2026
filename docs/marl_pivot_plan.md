# Custom Football MARL Pivot Plan

## Portfolio Framing

This project is now framed as:

> A custom football MARL environment seeded with real national-team metadata, used to learn tactical policies and compare them against rule-based football simulation baselines for WC2026 scenarios.

That framing is stronger and more defensible than claiming a finished world-class 11v11 football AI. The value is in the formulation, baselines, ablations, curriculum, and honest evaluation.

## Kept Components

- 48-team WC2026 data layer.
- Official FIFA rank mapping.
- Player, team, formation, and tactical preset models.
- Rule-based `MatchSimulator` as benchmark and regression guard.
- Diagnostics and event schema.

## New RL Layer

- `FootballEnv`: reset/step environment contract.
- `ActionSpace`: discrete high-level football actions.
- `ObservationBuilder`: compact per-agent state for all 22 starters.
- `RewardFunction`: shaped football rewards with anti-degenerate penalties.
- `Policy`: random, tactical-preset, and tabular epsilon-greedy policies.
- `TabularQTrainer`: local-compute first training loop.
- `SelfPlayRunner`: lightweight iterative self-play scaffold.
- `EvaluationSuite`: policy comparison metrics.
- `TrainingLogger`: terminal, CSV, JSONL, config, and checkpoint observability for long local runs.
- `evaluate_policy`: checkpoint evaluation against random, tactical, self-play, and rule-based baselines.
- `RoleGroupPolicy`: separate shared tabular policies for defense, midfield, and attack.

## Curriculum

- Phase 1: single team-controller policy controls each team.
- Phase 2: role-group agents for defense, midfield, and attack.
- Phase 3: small-sided 5v5 curriculum scenarios.
- Phase 4: 11v11 lightweight MARL with shared policies by role.

## Evaluation Targets

- Random policy.
- Tactical-preset policy.
- Rule-based simulator baseline.
- Mirrored self-play opponent.

Core metrics:

- Win rate.
- xG difference.
- Goals per match.
- Shots and shots on target.
- Turnover rate.
- Possession progression.
- High-score outliers.
- Policy entropy and action distribution.

## Risks

- Local full 11v11 MARL may not converge meaningfully without curriculum.
- Shaped rewards can produce degenerate behavior.
- Learned policies may initially underperform tactical presets.
- Research credibility depends on strict diagnostics and transparent results.
- Training reward alone can be misleading, so readable logs must track xG, action distribution, turnovers, and reward-hacking warnings.
- Long training runs should be treated like experiments: checkpoint first, then compare through evaluation reports before increasing curriculum complexity.

## References

- Google Research Football: `https://arxiv.org/abs/1907.11180`
- GRF MARL benchmark discussion: `https://arxiv.org/abs/2309.12951`
