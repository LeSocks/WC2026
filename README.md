# WC2026 Tactical Match Simulator

Agent-based football simulation and World Cup 2026 tournament predictor.

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

Run a quick distribution check:

```powershell
python -c "from src.data.loader import load_team; from src.engine.match import MatchSimulator; h=load_team('France'); a=load_team('Morocco'); print([MatchSimulator(h,a,rng_seed=i).simulate().home_goals for i in range(5)])"
```

Run Day 7 diagnostics:

```powershell
python -m src.engine.diagnostics --runs-per-pair 20
```
