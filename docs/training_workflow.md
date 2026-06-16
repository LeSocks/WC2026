# Training Workflow

Use this workflow for long local training runs. It is designed to feel familiar if you usually train in Colab or Jupyter: the terminal runs the job, while CSV/JSON files become the notebook-readable artifacts.

## 1. Start a Short Smoke Run

```powershell
python -m src.rl.train_controller France Morocco --episodes 20 --seed 42 --log-interval 5 --checkpoint-interval 10
```

Watch the terminal for:

- `reward_ma20`: smoother reward trend.
- `xG`: whether the policy creates real chances.
- `top_actions`: whether the policy collapses into one action.
- `warn`: reward-hacking hints.

## 2. Start a Longer Run

```powershell
python -m src.rl.train_controller France Morocco --episodes 1000 --seed 42 --log-interval 25 --checkpoint-interval 100
```

The run writes to `training_runs/<run_id>/`:

- `metrics.csv`
- `events.jsonl`
- `config.json`
- `checkpoint.json`

## 3. Evaluate the Checkpoint

```powershell
python -m src.rl.evaluate_policy France Morocco --checkpoint training_runs\<run_id>\checkpoint.json --episodes 100 --seed 10000
```

The evaluation compares the checkpoint against:

- random policy
- tactical preset policy
- mirrored learned self-play
- rule-based match simulator

## 4. Inspect in Jupyter

```python
import pandas as pd

metrics = pd.read_csv("training_runs/<run_id>/metrics.csv")
evaluation = pd.read_csv("evaluation_runs/<eval_id>/evaluation.csv")

metrics[["episode", "reward_ma_20", "xg_diff", "q_states"]].tail()
evaluation
```

## 5. Resume if Worth Continuing

```powershell
python -m src.rl.train_controller France Morocco --episodes 2000 --resume training_runs\<run_id>\checkpoint.json
```

Only scale episode count after the evaluation report shows improvement beyond random and tactical baselines.

## 6. Move to Role-Group Training

After team-controller results are readable and not degenerate, train shared role policies:

```powershell
python -m src.rl.train_role_groups France Morocco --episodes 1000 --seed 42 --log-interval 25 --checkpoint-interval 100
```

Evaluate it exactly the same way:

```powershell
python -m src.rl.evaluate_policy France Morocco --checkpoint training_runs\<run_id>\checkpoint.json --episodes 100 --seed 10000
```

Role-group training is more MARL-like, but it should come after the team-controller baseline so the added complexity has a comparison point.
