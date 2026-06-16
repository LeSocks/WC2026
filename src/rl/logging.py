from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.rl.actions import ActionType
from src.rl.policies import EpsilonGreedyTabularPolicy, RoleGroupPolicy


METRIC_FIELDS: tuple[str, ...] = (
    "episode",
    "episode_reward",
    "reward_ma_20",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "xg_diff",
    "shots",
    "shots_on_target",
    "turnovers",
    "low_quality_shots",
    "progressions",
    "epsilon",
    "q_states",
    "elapsed_seconds",
    "warnings",
    "action_counts",
    "action_distribution",
)


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    episode_reward: float
    reward_ma_20: float
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    shots: int
    shots_on_target: int
    action_counts: dict[str, int]
    turnovers: int
    low_quality_shots: int
    progressions: int
    epsilon: float
    q_states: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def xg_diff(self) -> float:
        return self.home_xg - self.away_xg

    @property
    def action_distribution(self) -> dict[str, float]:
        total = sum(self.action_counts.values())
        if total == 0:
            return {}
        return {action: count / total for action, count in sorted(self.action_counts.items())}

    def to_row(self) -> dict[str, object]:
        return {
            "episode": self.episode,
            "episode_reward": round(self.episode_reward, 6),
            "reward_ma_20": round(self.reward_ma_20, 6),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_xg": round(self.home_xg, 6),
            "away_xg": round(self.away_xg, 6),
            "xg_diff": round(self.xg_diff, 6),
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "turnovers": self.turnovers,
            "low_quality_shots": self.low_quality_shots,
            "progressions": self.progressions,
            "epsilon": round(self.epsilon, 6),
            "q_states": self.q_states,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "warnings": ";".join(self.warnings),
            "action_counts": json.dumps(self.action_counts, sort_keys=True),
            "action_distribution": json.dumps(
                {key: round(value, 6) for key, value in self.action_distribution.items()},
                sort_keys=True,
            ),
        }


@dataclass
class TrainingLogger:
    run_dir: Path
    config: dict[str, Any]
    log_interval: int = 10
    checkpoint_interval: int = 50
    echo_terminal: bool = True

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.events_path = self.run_dir / "events.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.started_at = time.monotonic()
        self.config.setdefault("git_commit", current_git_commit())
        self._write_config()
        self._ensure_metrics_header()

    def log_episode(
        self,
        metrics: EpisodeMetrics,
        policy: EpsilonGreedyTabularPolicy | RoleGroupPolicy,
        total_episodes: int,
        force_checkpoint: bool = False,
    ) -> None:
        self._append_metrics(metrics)
        self._append_event(metrics)

        if self.echo_terminal and (metrics.episode == 1 or metrics.episode % self.log_interval == 0 or metrics.episode == total_episodes):
            print(format_terminal_summary(metrics, total_episodes), flush=True)

        should_checkpoint = force_checkpoint or metrics.episode % self.checkpoint_interval == 0 or metrics.episode == total_episodes
        if should_checkpoint:
            save_checkpoint(self.checkpoint_path, policy, metrics)

    def _write_config(self) -> None:
        self.config_path.write_text(json.dumps(self.config, indent=2, sort_keys=True), encoding="utf-8")

    def _ensure_metrics_header(self) -> None:
        if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
            return
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writeheader()

    def _append_metrics(self, metrics: EpisodeMetrics) -> None:
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
            writer.writerow(metrics.to_row())

    def _append_event(self, metrics: EpisodeMetrics) -> None:
        payload = {
            **metrics.to_row(),
            "action_counts": metrics.action_counts,
            "action_distribution": metrics.action_distribution,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def create_run_dir(root: str | Path, home: str, away: str, run_id: str | None = None) -> Path:
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{slugify(home)}_vs_{slugify(away)}"
    return Path(root) / run_id


def save_checkpoint(path: str | Path, policy: EpsilonGreedyTabularPolicy | RoleGroupPolicy, metrics: EpisodeMetrics) -> None:
    payload = {
        "episode": metrics.episode,
        "last_metrics": metrics_to_json_payload(metrics),
        **serialize_policy(policy),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: str | Path) -> tuple[EpsilonGreedyTabularPolicy | RoleGroupPolicy, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policy_type = str(payload.get("policy_type", "team_controller"))
    if policy_type == "role_group":
        role_payload = payload.get("role_policies", {})
        if not isinstance(role_payload, dict):
            role_payload = {}
        policy = RoleGroupPolicy(
            role_policies={
                role: EpsilonGreedyTabularPolicy(
                    epsilon=float(role_data.get("epsilon", payload.get("epsilon", 0.10))),
                    q_values=deserialize_q_values(role_data.get("q_values", [])),
                )
                for role, role_data in role_payload.items()
                if isinstance(role_data, dict)
            }
        )
    else:
        policy = EpsilonGreedyTabularPolicy(
            epsilon=float(payload["epsilon"]),
            q_values=deserialize_q_values(payload.get("q_values", [])),
        )
    return policy, int(payload["episode"])


def serialize_policy(policy: EpsilonGreedyTabularPolicy | RoleGroupPolicy) -> dict[str, object]:
    if isinstance(policy, RoleGroupPolicy):
        return {
            "policy_type": "role_group",
            "epsilon": policy.epsilon,
            "role_policies": {
                role: {
                    "epsilon": role_policy.epsilon,
                    "q_values": serialize_q_values(role_policy.q_values),
                }
                for role, role_policy in sorted(policy.role_policies.items())
            },
        }
    return {
        "policy_type": "team_controller",
        "epsilon": policy.epsilon,
        "q_values": serialize_q_values(policy.q_values),
    }


def serialize_q_values(q_values: dict[tuple[object, ...], dict[ActionType, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state_key, action_values in q_values.items():
        rows.append(
            {
                "state_key": list(state_key),
                "actions": {action.value: value for action, value in sorted(action_values.items(), key=lambda item: item[0].value)},
            }
        )
    return rows


def deserialize_q_values(rows: list[dict[str, object]]) -> dict[tuple[object, ...], dict[ActionType, float]]:
    q_values: dict[tuple[object, ...], dict[ActionType, float]] = {}
    for row in rows:
        state_key_payload = row.get("state_key", [])
        action_payload = row.get("actions", {})
        if not isinstance(state_key_payload, list) or not isinstance(action_payload, dict):
            continue
        q_values[tuple(state_key_payload)] = {
            ActionType(str(action)): float(value) for action, value in action_payload.items()
        }
    return q_values


def format_terminal_summary(metrics: EpisodeMetrics, total_episodes: int) -> str:
    actions = ", ".join(
        f"{action} {share:.0%}"
        for action, share in sorted(metrics.action_distribution.items(), key=lambda item: item[1], reverse=True)[:3]
    )
    warning_text = f" | warn={','.join(metrics.warnings)}" if metrics.warnings else ""
    return (
        f"Ep {metrics.episode}/{total_episodes} | "
        f"reward_ma20={metrics.reward_ma_20:.3f} | "
        f"score={metrics.home_goals}-{metrics.away_goals} | "
        f"xG={metrics.home_xg:.2f}-{metrics.away_xg:.2f} | "
        f"shots={metrics.shots}/{metrics.shots_on_target} SOT | "
        f"q_states={metrics.q_states} | "
        f"top_actions={actions or 'none'} | "
        f"{format_duration(metrics.elapsed_seconds)}"
        f"{warning_text}"
    )


def metrics_to_json_payload(metrics: EpisodeMetrics) -> dict[str, object]:
    return {
        **metrics.to_row(),
        "action_counts": metrics.action_counts,
        "action_distribution": metrics.action_distribution,
        "warnings": metrics.warnings,
    }


def detect_reward_hacking(metrics: EpisodeMetrics) -> list[str]:
    distribution = metrics.action_distribution
    warnings: list[str] = []
    shoot_rate = distribution.get("shoot", 0.0)
    pass_hold_rate = distribution.get("safe_pass", 0.0) + distribution.get("hold", 0.0)

    if shoot_rate > 0.38 and metrics.home_xg < 0.8:
        warnings.append("shot_spam")
    if pass_hold_rate > 0.82 and metrics.progressions <= 1:
        warnings.append("stale_possession")
    if metrics.episode_reward > 0.12 and metrics.home_xg < 0.25 and metrics.home_goals == 0:
        warnings.append("reward_without_chance_quality")
    return warnings


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
