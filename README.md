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
