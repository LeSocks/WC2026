# WC2026 Custom Football MARL Environment

A custom football MARL environment seeded with real national-team metadata, used to learn tactical policies and compare them against rule-based football simulation baselines for WC2026 scenarios.

The original tactical simulator remains in the project as the world model, data layer, tactical prior, and evaluation baseline. The research direction from Day 10 onward is full custom 11v11-compatible MARL, trained locally through a curriculum from lightweight tactical controllers toward shared role policies.

## Day 1 Scope

- Initialize Git repository.
- Create Python virtual environment.
- Scaffold project directories from the TDD.
- Add dependency manifest.
- Verify the environment with a smoke test.

## Day 2 Scope

- Add a polite FBref scraper template with a 3-second request delay.
- Seed the latest top 10 national-team baseline from FIFA's 11 June 2026 ranking update.
- Create manual FIFA-style starter ratings for 10 teams.
- Save raw CSVs under `data/raw/`.

## Day 3 Scope

- Implement `Position`, `PlayStyle`, `PlayerStats`, and `Player`.
- Compute behavioral parameters from FIFA-style stats.
- Add playstyle modifiers for player differentiation.
- Load Day 2 CSV rows into typed `Player` objects.

## Day 4 Scope

- Implement `Team`, `Formation`, and tactical presets for the 10 seeded teams.
- Add lineup ordering from formation slots.
- Expose team quality metrics for the match engine.

## Day 5 Scope

- Implement event types, match state, and a basic possession-chain simulator.
- Resolve pass, dribble, interception, tackle, shot, save, miss, and goal events.
- Add a terminal command for running one simulated match with an event log.

## Day 6 Scope

- Tune shot selection and shot resolution toward realistic match totals.
- Track xG for both teams.
- Add blocked shots and separate on-target probability from goal probability.
- Fix half-time ordering and possession-time attribution.

## Day 7 Scope

- Add repeatable match-distribution diagnostics.
- Run top-10 matchup samples to check goals, shots, SOT, xG, and high-score outliers.
- Add a short development journal under `docs/`.

## Day 8 Scope

- Move tactical formulas into `TacticalEngine`.
- Add pressure modifiers, action probabilities, possession tendency, and transition logic.
- Make tactical style differences directly testable.

## Day 9 Scope

- Add WC2026 group metadata for all 48 teams.
- Generate Tier 1 starter data for all teams into `data/processed/`.
- Keep top-10 manual seed data and use generated team-profile placeholders for the remaining teams.

## Day 10 Scope

- Add `src/rl/` as the custom football MARL layer.
- Implement `FootballEnv.reset()` and `FootballEnv.step(actions)` with 22 agent slots from the start.
- Add compact per-agent observations, high-level discrete actions, and shaped rewards.
- Add tactical, random, tabular Q-learning, self-play, and evaluation scaffolds.
- Keep the existing rule-based `MatchSimulator` as a benchmark and regression guard.

## Day 11 Scope

- Add observable team-controller training for local compute.
- Log readable terminal progress plus CSV, JSONL, config, and checkpoint files.
- Support checkpoint resume for long-running tabular Q-learning runs.
- Add simple reward-hacking warnings for shot spam, stale possession, and reward without chance quality.

## Day 12 Scope

- Add policy evaluation against random, tactical preset, self-play, and rule-based simulator baselines.
- Load `checkpoint.json` from observable training runs.
- Write readable `evaluation.json` and `evaluation.csv` reports.
- Use changing seeds per evaluation episode so reports are not repeated identical matches.

## Day 13 Scope

- Add role-group tabular MARL for `defense`, `midfield`, and `attack` shared policies.
- Keep the same observable training outputs and JSON checkpoint format.
- Allow `evaluate_policy` to evaluate role-group checkpoints with the same baseline harness.
- Keep team-controller training available as the recommended first long-run curriculum.

## MARL Roadmap

- Day 10: RL environment contract: reset, step, observations, actions, rewards.
- Day 11: Single-team tactical controller training loop.
- Day 12: Self-play/evaluation harness and baseline policies.
- Day 13: Role-group MARL for defense, midfield, and attack shared policies.
- Day 14: 5v5 curriculum scenario and reward tuning.
- Day 15: 11v11-compatible training path with shared role policies.
- Day 16: Compare learned policies against rule-based tactical presets.
- Day 17+: Tournament simulation can use either rule-based or learned policies.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

## Project Structure

```text
data/
  raw/
  processed/
  squads/
src/
  models/
  engine/
  rl/
  viz/
  data/
app/
tests/
notebooks/
```

## Data Commands

Regenerate Day 2 raw seed files:

```powershell
python -m src.data.seed_top10_ratings
```

The generator writes:

- `data/raw/fifa_rankings_top10_2026-06-11.csv`
- `data/raw/fifa_style_player_ratings_top10.csv`
- `data/raw/fbref_team_sources.csv`

Load seeded players:

```powershell
python -c "from src.data.loader import load_players; print(len(load_players()))"
```

Print a team lineup:

```powershell
python -c "from src.data.loader import load_team; print(load_team('France').lineup_text())"
```

Run one simulated match:

```powershell
python -m src.engine.run_match France Morocco --seed 42
```

Run one lightweight RL environment episode:

```powershell
python -m src.rl.run_episode France Morocco --seed 42 --steps 12
```

Run observable team-controller training:

```powershell
python -m src.rl.train_controller France Morocco --episodes 500 --seed 42 --log-interval 25 --checkpoint-interval 100
```

Run observable role-group training:

```powershell
python -m src.rl.train_role_groups France Morocco --episodes 500 --seed 42 --log-interval 25 --checkpoint-interval 100
```

Resume a training run:

```powershell
python -m src.rl.train_controller France Morocco --episodes 1000 --resume training_runs\<run_id>\checkpoint.json
```

Evaluate a trained checkpoint:

```powershell
python -m src.rl.evaluate_policy France Morocco --checkpoint training_runs\<run_id>\checkpoint.json --episodes 100 --seed 10000
```

Run a quick distribution check:

```powershell
python -c "from src.data.loader import load_team; from src.engine.match import MatchSimulator; h=load_team('France'); a=load_team('Morocco'); print([MatchSimulator(h,a,rng_seed=i).simulate().home_goals for i in range(5)])"
```

Run Day 7 diagnostics:

```powershell
python -m src.engine.diagnostics --runs-per-pair 20
```

Run diagnostics across all 48 teams:

```powershell
python -m src.engine.diagnostics --runs-per-pair 5 --all-teams
```

Regenerate all 48-team processed data:

```powershell
python -m src.data.seed_48_teams
```

Check default full dataset loading:

```powershell
python -c "from src.data.loader import load_all_teams, load_players; print(len(load_all_teams()), len(load_players()))"
```

Run a small tabular training smoke test:

```powershell
python -c "from src.data.loader import load_team; from src.rl.env import FootballEnv; from src.rl.trainer import TabularQTrainer; env=FootballEnv(load_team('France'), load_team('Morocco'), rng_seed=42, max_minutes=12); policy,result=TabularQTrainer().train_team_controller(env, episodes=3); print(result.mean_reward, len(policy.q_values))"
```

Training logs are written under `training_runs/` by default:

- `metrics.csv`: episode-level metrics for spreadsheet or pandas analysis.
- `events.jsonl`: readable JSON lines for per-episode audit trails.
- `config.json`: teams, seed, hyperparameters, and git commit when available.
- `checkpoint.json`: portable Q-table checkpoint for resume and inspection.

Role-group checkpoints use the same filename but contain separate Q-tables for `defense`, `midfield`, and `attack`.

Evaluation reports are written under `evaluation_runs/` by default:

- `evaluation.json`: metadata plus structured results.
- `evaluation.csv`: compact comparison table for pandas, spreadsheet, or notebook analysis.

## Research Notes

- The environment is 11v11-compatible at the API level, but local-compute training should start with team-controller and role-group curricula before full 11v11 shared policies.
- Discrete high-level actions are deliberate: `hold`, `safe_pass`, `progressive_pass`, `switch_play`, `dribble`, `shoot`, `press`, `drop`, `mark`, and `clear`.
- Evaluation should report learned policy results against random policy, tactical-preset policy, mirrored self-play, and the existing rule-based match simulator.
- Long local training runs should be monitored through `reward_ma_20`, xG difference, action distribution, and reward-hacking warnings instead of reward alone.
- A serious long run should follow this loop: train with `train_controller`, inspect `metrics.csv`, evaluate the checkpoint with `evaluate_policy`, then resume or tune.
- Prefer the long-run order: team-controller first, role-group second, 5v5 curriculum third.
- References: Google Research Football (`https://arxiv.org/abs/1907.11180`) and later GRF MARL benchmark discussion (`https://arxiv.org/abs/2309.12951`).
