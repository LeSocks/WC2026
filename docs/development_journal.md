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

## Git and Account Notes

- Initial commit was rewritten because the Git author used the old GitHub identity.
- Fix: updated global Git identity to `LeSocks <rafaelcavinet@gmail.com>`, amended with `--reset-author`, cleared old Git Credential Manager auth, and pushed to `LeSocks/WC2026`.
