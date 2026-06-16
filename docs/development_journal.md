# WC2026 Simulator Development Journal

Short build journal for the first week. The goal is to preserve the main trials, errors, and decisions so later tuning has context.

## Day 1 - Project Setup

- Scope: initialized repo, virtual environment, project folders, dependency manifest, Streamlit placeholder, and scaffold test.
- Trial: installed the full stack from the TDD in `.venv` and verified imports for `numpy`, `pandas`, `scipy`, `streamlit`, `plotly`, and `mplsoccer`.
- Error: the first sandboxed PowerShell commands failed with a `windows sandbox: spawn setup refresh` error. Retried with elevated shell access and continued.
- Result: `pytest` passed and Streamlit responded locally on port `8501`.

## Day 2 - Data Acquisition Seed

- Scope: added a polite FBref scraper template and raw seed CSVs for top-10 team ratings.
- Trial: verified the June 11, 2026 FIFA ranking baseline before seeding teams.
- Error: `pandas.read_html` on raw HTML strings failed under the installed pandas version because it interpreted the string as a file path.
- Fix: wrapped table HTML in `StringIO` inside `src/data/scraper.py`.
- Result: generated 10 ranking rows and 110 starter rating rows.

## Day 3 - Player Model

- Scope: implemented `Position`, `PlayStyle`, `PlayerStats`, and `Player` with behavioral parameters.
- Trial: loaded Day 2 CSV rows into typed `Player` objects and validated all seeded playstyles.
- Error: the seed data had more playstyle values than the original TDD enum.
- Fix: expanded `PlayStyle` to cover seed-specific roles like `creator`, `stopper`, `wing_back`, and `ball_playing_cb`.
- Result: player initialization and behavioral parameter tests passed.

## Day 4 - Team and Formation

- Scope: implemented tactical presets, formations, team construction, lineup printing, and quality metrics.
- Trial: built all 10 seeded teams from CSV using tactical presets.
- Error avoided: aggressive exports from `src/engine/__init__.py` risked circular imports between `Team` and engine modules.
- Fix: kept engine package exports minimal and imported submodules directly.
- Result: all teams could print a complete 11-player lineup.

## Day 5 - Basic Match Engine

- Scope: implemented match events, match state, possession chains, pass/dribble/shot resolution, and a CLI runner.
- Trial: ran `python -m src.engine.run_match France Morocco --seed 42`.
- Error: early engine output had too many shots on target, around 15-17 per match in quick samples.
- Decision: left Day 5 runnable first, then reserved deeper scoring realism work for Day 6.
- Result: one full match could run in terminal with event log and scoreline.

## Day 6 - Shot Resolution Tuning

- Scope: tuned scoring realism, xG tracking, blocked shots, half-time ordering, and possession attribution.
- Trial: compared 100-simulation samples before and after tuning.
- Error: on-target probability originally reused `shot_accuracy_mean` too directly, inflating SOT.
- Fix: separated xG, on-target probability, goalkeeper save rate, and goal probability if on target.
- Result: sample averages moved to roughly 2.1-2.7 goals, 14-18 shots, and 6-8 SOT per match.

## Day 7 - Buffer and Distribution Diagnostics

- Scope: added repeatable diagnostics for matchup distributions and outlier detection.
- Trial: ran 20 simulations across all 45 top-10 pairings, totaling 900 matches.
- Observed distribution: 2.49 goals/match, 14.98 shots/match, 7.08 SOT/match, and 2.38 xG/match.
- Outliers: 17 high-score matches at 7+ total goals out of 900 runs, about 1.9%.
- Decision: no Day 7 scoring patch was needed after the diagnostic run; the current parameters are stable enough for Week 2 tactical-system work.

## Day 8 - Tactical Engine

- Scope: moved tactical formulas out of `MatchSimulator` and into a reusable `TacticalEngine`.
- Trial: compared high press vs low block, possession vs direct attack, and direct transition behavior through focused unit tests.
- Error avoided: keeping tactical formulas inline in `match.py` would make Week 2 tuning hard to isolate.
- Fix: centralized possession tendency, pressure modifier, action probability, pass/dribble modifiers, and zone transition in `src/engine/tactics.py`.
- Result: tactical styles now produce testable differences before running full match simulations.

## Day 9 - Full 48-Team Tier 1 Data

- Scope: added all 12 WC2026 groups and generated a complete 48-team Tier 1 dataset.
- Trial: verified the group list against current tournament schedule pages before encoding the metadata.
- Constraint: only the top-10 teams currently have manual starter ratings; the other 38 teams use generated placeholders from team-level strength profiles.
- Fix: wrote `src/data/seed_48_teams.py` to generate `data/processed/all_teams.json` and `data/processed/all_players_48.csv` reproducibly.
- Result: loader defaults now return 48 teams and 528 players when processed data exists.

## Day 10 - Custom MARL Environment Pivot

- Scope: pivoted the project from a tactical simulator into a custom football MARL research environment while keeping the existing simulator as a benchmark.
- Trial: added `src/rl/` with high-level actions, compact observations, shaped rewards, `FootballEnv.reset()`, `FootballEnv.step(actions)`, baseline policies, tabular Q-learning, self-play, and evaluation scaffolds.
- Decision: expose 22 agent slots from the start, but train locally through curriculum stages instead of pretending full 11v11 from scratch will converge immediately.
- Error avoided: did not replace the stable rule-based match engine; it remains a deterministic baseline and regression guard.
- Risk: reward hacking is likely if shooting, passing, and pressure rewards are not monitored through diagnostics.
- Result: RL environment tests pass, and both `python -m src.engine.run_match France Morocco --seed 42` and `python -m src.rl.run_episode France Morocco --seed 42 --steps 5` run successfully.

## Day 11 - Observable Team-Controller Training

- Scope: added a local-compute training path that is observable through terminal summaries and readable files.
- Trial: created `TrainingLogger`, `train_controller` CLI, episode metrics, JSON checkpoints, and resume support.
- Decision: kept logging dependency-free with CSV, JSONL, and JSON so long runs can be inspected without TensorBoard or W&B.
- Error avoided: did not optimize for GPU training yet; the current bottleneck is formulation and diagnostics, not neural network throughput.
- Guardrail: added warnings for shot spam, stale possession, and reward gains that do not create chance quality.
- Result: training runs now write `metrics.csv`, `events.jsonl`, `config.json`, and `checkpoint.json` under `training_runs/`.

## Day 12 - Evaluation and Baseline Harness

- Scope: added a checkpoint evaluation path so training runs can be compared against baselines instead of judged by reward alone.
- Trial: added `evaluate_policy` CLI and report writing for `evaluation.json` and `evaluation.csv`.
- Decision: compare learned checkpoints against random policy, tactical preset policy, mirrored learned self-play, and the rule-based `MatchSimulator`.
- Fix: evaluation factories can now vary seeds per episode, avoiding repeated identical match samples.
- Result: the project now has a train -> observe -> checkpoint -> evaluate loop suitable for long local runs and notebook analysis.

## Day 13 - Role-Group MARL

- Scope: added lightweight role-group training for shared `defense`, `midfield`, and `attack` tabular policies.
- Trial: added `RoleGroupPolicy`, `train_role_groups` CLI, role-group checkpoint serialization, resume support, and evaluation compatibility.
- Decision: keep role-group training as the second curriculum step after team-controller training, not as the first long run.
- Error avoided: did not start long training automatically; only short smoke runs and tests should be run by automation.
- Result: role-group checkpoints can now be trained, resumed, and evaluated through the same report pipeline.

## Git and Account Notes

- Initial commit was rewritten because the Git author used the old GitHub identity.
- Fix: updated global Git identity to `LeSocks <rafaelcavinet@gmail.com>`, amended with `--reset-author`, cleared old Git Credential Manager auth, and pushed to `LeSocks/WC2026`.
