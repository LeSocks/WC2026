import csv
import json
import subprocess
import sys

from src.data.loader import load_team
from src.engine.events import EventType, MatchEvent, PitchZone
from src.rl.actions import ActionSpace, ActionType
from src.rl.env import FootballEnv
from src.rl.evaluation import EvaluationSuite, write_evaluation_report
from src.rl.logging import EpisodeMetrics, TrainingLogger, load_checkpoint
from src.rl.policies import RandomPolicy, RoleGroupPolicy, TacticalPresetPolicy
from src.rl.rewards import RewardFunction
from src.rl.trainer import TabularQTrainer


def test_action_space_validates_high_level_actions() -> None:
    action_space = ActionSpace()

    assert action_space.validate("safe_pass") == ActionType.SAFE_PASS
    assert action_space.validate(ActionType.PRESS) == ActionType.PRESS


def test_football_env_reset_exposes_11v11_agent_slots() -> None:
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=42, max_minutes=8)

    observations = env.reset()

    assert len(observations) == 22
    assert {key.split(":")[0] for key in observations} == {"home", "away"}
    assert sum(obs.side == "home" for obs in observations.values()) == 11
    assert sum(obs.side == "away" for obs in observations.values()) == 11
    assert all(obs.minute == 1 for obs in observations.values())
    assert all(len(obs.as_vector()) == 16 for obs in observations.values())


def test_football_env_step_accepts_team_controller_actions() -> None:
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=7, max_minutes=6)
    env.reset()

    observations, rewards, done, info = env.step({"home": "progressive_pass", "away": "press"})

    assert len(observations) == 22
    assert set(rewards) == {"France", "Morocco"}
    assert done is False
    assert info["action"] in {action.value for action in ActionType}
    assert env.state.events[-1].event_type in {
        EventType.PASS,
        EventType.INTERCEPTION,
        EventType.DRIBBLE,
        EventType.TACKLE,
        EventType.GOAL,
        EventType.SAVE,
        EventType.MISS,
        EventType.BLOCKED,
    }


def test_football_env_runs_to_terminal_state() -> None:
    env = FootballEnv(load_team("Spain"), load_team("Brazil"), rng_seed=11, max_minutes=3)
    env.reset()
    done = False

    while not done:
        _, _, done, _ = env.step({})

    assert env.state.minute == 3
    assert env.state.events[-1].event_type == EventType.FULL_TIME


def test_reward_function_penalizes_low_quality_shot_spam() -> None:
    event = MatchEvent(
        minute=12,
        event_type=EventType.MISS,
        team="France",
        player_name="Test Player",
        zone=PitchZone.DEF_CENTER,
        success=False,
        extra={"xg": 0.006, "on_target": False},
    )

    rewards = RewardFunction().score_step(event, "France", "Morocco")

    assert rewards["France"] < 0
    assert rewards["Morocco"] == 0


def test_evaluation_suite_reports_policy_action_distribution() -> None:
    def factory() -> FootballEnv:
        return FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=5, max_minutes=6)

    result = EvaluationSuite().evaluate_team_policy(
        factory,
        home_policy=TacticalPresetPolicy(),
        away_policy=RandomPolicy(),
        episodes=2,
    )

    assert result.episodes == 2
    assert result.label == "policy"
    assert 0 <= result.win_rate <= 1
    assert result.win_rate + result.draw_rate + result.loss_rate == 1
    assert result.avg_shots >= 0
    assert result.action_distribution


def test_evaluation_report_writes_json_and_csv(tmp_path) -> None:
    def factory(index: int) -> FootballEnv:
        return FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=30 + index, max_minutes=4)

    suite = EvaluationSuite()
    results = [
        suite.random_baseline(factory, episodes=2),
        suite.rule_based_baseline(load_team("France"), load_team("Morocco"), episodes=2, seed_offset=30),
    ]

    json_path, csv_path = write_evaluation_report(
        results,
        tmp_path / "eval",
        metadata={"home": "France", "away": "Morocco", "episodes": 2},
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["home"] == "France"
    assert {row["label"] for row in payload["results"]} == {"random_vs_random", "rule_based_match_simulator"}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["label"] == "random_vs_random"


def test_tabular_trainer_smoke_updates_policy_values() -> None:
    env = FootballEnv(load_team("Argentina"), load_team("Germany"), rng_seed=9, max_minutes=6)

    policy, result = TabularQTrainer(epsilon=0.2).train_team_controller(env, episodes=2)

    assert result.episodes == 2
    assert len(result.total_rewards) == 2
    assert result.action_counts
    assert policy.q_values


def test_training_logger_writes_readable_run_files(tmp_path) -> None:
    logger = TrainingLogger(
        run_dir=tmp_path / "run",
        config={"home": "France", "away": "Morocco", "episodes": 1},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=12, max_minutes=4)

    policy, result = TabularQTrainer().train_team_controller(env, episodes=1, logger=logger)

    assert result.completed_episodes == 1
    assert (logger.run_dir / "config.json").exists()
    assert (logger.run_dir / "metrics.csv").exists()
    assert (logger.run_dir / "events.jsonl").exists()
    assert (logger.run_dir / "checkpoint.json").exists()

    config = json.loads((logger.run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["home"] == "France"
    assert "git_commit" in config

    with (logger.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["episode"] == "1"
    assert rows[0]["action_counts"]
    assert rows[0]["action_distribution"]

    event_line = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(event_line)["episode"] == 1

    loaded_policy, episode = load_checkpoint(logger.run_dir / "checkpoint.json")
    assert episode == 1
    assert loaded_policy.q_values == policy.q_values


def test_tabular_trainer_invokes_episode_callback() -> None:
    seen: list[EpisodeMetrics] = []
    env = FootballEnv(load_team("Spain"), load_team("Brazil"), rng_seed=14, max_minutes=4)

    TabularQTrainer().train_team_controller(
        env,
        episodes=2,
        callback=lambda metrics, _policy, _total: seen.append(metrics),
    )

    assert [metrics.episode for metrics in seen] == [1, 2]
    assert all(metrics.action_counts for metrics in seen)


def test_training_resume_continues_from_checkpoint(tmp_path) -> None:
    first_logger = TrainingLogger(
        run_dir=tmp_path / "resume_run",
        config={"home": "France", "away": "Morocco", "episodes": 2},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=15, max_minutes=4)
    TabularQTrainer().train_team_controller(env, episodes=2, logger=first_logger)

    second_logger = TrainingLogger(
        run_dir=tmp_path / "resume_run",
        config={"home": "France", "away": "Morocco", "episodes": 4, "resume": "checkpoint.json"},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    resumed_env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=15, max_minutes=4)
    _policy, result = TabularQTrainer().train_team_controller(
        resumed_env,
        episodes=4,
        logger=second_logger,
        resume_checkpoint=tmp_path / "resume_run" / "checkpoint.json",
    )

    assert result.start_episode == 2
    assert result.completed_episodes == 4
    _loaded_policy, episode = load_checkpoint(tmp_path / "resume_run" / "checkpoint.json")
    assert episode == 4


def test_role_group_trainer_updates_three_role_policies(tmp_path) -> None:
    logger = TrainingLogger(
        run_dir=tmp_path / "role_run",
        config={"home": "France", "away": "Morocco", "episodes": 2, "mode": "role_group_tabular_q"},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=41, max_minutes=4)

    policy, result = TabularQTrainer(epsilon=0.2).train_role_groups(env, episodes=2, logger=logger)

    assert isinstance(policy, RoleGroupPolicy)
    assert result.completed_episodes == 2
    assert set(policy.role_policies) == {"defense", "midfield", "attack"}
    assert policy.q_state_count > 0
    assert (logger.run_dir / "checkpoint.json").exists()

    loaded_policy, episode = load_checkpoint(logger.run_dir / "checkpoint.json")
    assert isinstance(loaded_policy, RoleGroupPolicy)
    assert episode == 2
    assert loaded_policy.q_state_count == policy.q_state_count


def test_role_group_training_resume_continues_from_checkpoint(tmp_path) -> None:
    logger = TrainingLogger(
        run_dir=tmp_path / "role_resume",
        config={"home": "France", "away": "Morocco", "episodes": 2, "mode": "role_group_tabular_q"},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=42, max_minutes=4)
    TabularQTrainer().train_role_groups(env, episodes=2, logger=logger)

    resumed_logger = TrainingLogger(
        run_dir=tmp_path / "role_resume",
        config={"home": "France", "away": "Morocco", "episodes": 4, "mode": "role_group_tabular_q"},
        log_interval=1,
        checkpoint_interval=1,
        echo_terminal=False,
    )
    resumed_env = FootballEnv(load_team("France"), load_team("Morocco"), rng_seed=42, max_minutes=4)
    _policy, result = TabularQTrainer().train_role_groups(
        resumed_env,
        episodes=4,
        logger=resumed_logger,
        resume_checkpoint=tmp_path / "role_resume" / "checkpoint.json",
    )

    assert result.start_episode == 2
    assert result.completed_episodes == 4


def test_train_controller_cli_smoke_writes_run_folder(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.train_controller",
            "France",
            "Morocco",
            "--episodes",
            "3",
            "--max-minutes",
            "4",
            "--seed",
            "21",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "cli_smoke",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "cli_smoke"
    assert "Training complete" in completed.stdout
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoint.json").exists()


def test_train_role_groups_cli_smoke_writes_role_checkpoint(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.train_role_groups",
            "France",
            "Morocco",
            "--episodes",
            "3",
            "--max-minutes",
            "4",
            "--seed",
            "43",
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            "role_cli_smoke",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "role_cli_smoke"
    assert "Role-group training complete" in completed.stdout
    payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert payload["policy_type"] == "role_group"
    assert set(payload["role_policies"]) == {"attack", "defense", "midfield"}


def test_evaluate_policy_cli_smoke_reads_checkpoint(tmp_path) -> None:
    train_dir = tmp_path / "train"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.train_controller",
            "France",
            "Morocco",
            "--episodes",
            "2",
            "--max-minutes",
            "4",
            "--seed",
            "31",
            "--runs-dir",
            str(train_dir),
            "--run-id",
            "policy",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    eval_dir = tmp_path / "eval"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.evaluate_policy",
            "France",
            "Morocco",
            "--checkpoint",
            str(train_dir / "policy" / "checkpoint.json"),
            "--episodes",
            "2",
            "--max-minutes",
            "4",
            "--seed",
            "32",
            "--output-dir",
            str(eval_dir),
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluation files" in completed.stdout
    payload = json.loads((eval_dir / "evaluation.json").read_text(encoding="utf-8"))
    labels = {row["label"] for row in payload["results"]}
    assert "learned_vs_tactical" in labels
    assert "learned_vs_random" in labels
    assert "learned_self_play" in labels
    assert "rule_based_match_simulator" in labels
    assert (eval_dir / "evaluation.csv").exists()


def test_evaluate_policy_cli_reads_role_group_checkpoint(tmp_path) -> None:
    train_dir = tmp_path / "role_train"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.train_role_groups",
            "France",
            "Morocco",
            "--episodes",
            "2",
            "--max-minutes",
            "4",
            "--seed",
            "44",
            "--runs-dir",
            str(train_dir),
            "--run-id",
            "role_policy",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    eval_dir = tmp_path / "role_eval"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rl.evaluate_policy",
            "France",
            "Morocco",
            "--checkpoint",
            str(train_dir / "role_policy" / "checkpoint.json"),
            "--episodes",
            "2",
            "--max-minutes",
            "4",
            "--seed",
            "45",
            "--output-dir",
            str(eval_dir),
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((eval_dir / "evaluation.json").read_text(encoding="utf-8"))
    labels = {row["label"] for row in payload["results"]}
    assert "learned_vs_tactical" in labels
    assert "learned_self_play" in labels
