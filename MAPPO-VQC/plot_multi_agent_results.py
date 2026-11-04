"""Automatic plotting for multi-agent PredatorPrey VQC-MAPPO experiments.

The module is intentionally not a command-line program. The MAPPO driver
calls :func:`plot_multi_agent_results` automatically after training and
evaluation data have been saved.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIRECTORY = Path("result/plots")
DEFAULT_TRAINING_WINDOW = 50


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Compute a moving average while retaining shorter prefix windows."""
    if window <= 0:
        raise ValueError("window must be greater than zero")

    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)

    for index in range(len(array)):
        start = max(0, index - window + 1)
        result[index] = np.mean(array[start:index + 1])

    return result


def load_training_data(path: str | Path) -> list[dict]:
    """Read episode-level MAPPO training data from CSV."""
    rows: list[dict] = []

    with Path(path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "episode",
            "team_reward",
            "mean_agent_reward",
            "steps",
            "prey_captured",
            "predator_1_reward",
            "predator_2_reward",
            "actor_loss",
            "critic_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Training CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "team_reward": float(row["team_reward"]),
                    "mean_agent_reward": float(row["mean_agent_reward"]),
                    "steps": int(row["steps"]),
                    "prey_captured": float(row["prey_captured"]),
                    "predator_1_reward": float(row["predator_1_reward"]),
                    "predator_2_reward": float(row["predator_2_reward"]),
                    "actor_loss": float(row["actor_loss"]),
                    "critic_loss": float(row["critic_loss"]),
                    "entropy": float(row["entropy"]),
                    "approximate_kl": float(row["approximate_kl"]),
                    "clip_fraction": float(row["clip_fraction"]),
                }
            )

    if not rows:
        raise ValueError("The training CSV is empty.")
    return rows


def load_evaluation_data(path: str | Path) -> list[dict]:
    """Read episode-level deterministic-evaluation data from CSV."""
    rows: list[dict] = []

    with Path(path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "episode",
            "team_reward",
            "mean_agent_reward",
            "steps",
            "prey_captured",
            "predator_1_reward",
            "predator_2_reward",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Evaluation CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "team_reward": float(row["team_reward"]),
                    "mean_agent_reward": float(row["mean_agent_reward"]),
                    "steps": int(row["steps"]),
                    "prey_captured": float(row["prey_captured"]),
                    "predator_1_reward": float(row["predator_1_reward"]),
                    "predator_2_reward": float(row["predator_2_reward"]),
                }
            )

    if not rows:
        raise ValueError("The evaluation CSV is empty.")
    return rows


def _save_series_plot(
    episodes: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    filename: str,
    output_directory: Path,
    window: int | None = None,
) -> Path:
    """Save one generic episode-series plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(episodes, values, linewidth=0.9, alpha=0.35, label="Episode value")

    if window is not None:
        average = moving_average(values, window)
        ax.plot(
            episodes,
            average,
            linewidth=2.0,
            label=f"{window}-episode moving average",
        )
        ax.legend()

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_agent_reward_plot(
    episodes: np.ndarray,
    predator_1_rewards: np.ndarray,
    predator_2_rewards: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot both predators explicitly on the same training reward figure."""
    p1_average = moving_average(predator_1_rewards, window)
    p2_average = moving_average(predator_2_rewards, window)
    marker_interval = max(1, len(episodes) // 20)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        predator_1_rewards,
        linewidth=0.7,
        alpha=0.20,
        label="Predator 1 episode reward",
    )
    ax.plot(
        episodes,
        predator_2_rewards,
        linewidth=0.7,
        alpha=0.20,
        label="Predator 2 episode reward",
    )
    ax.plot(
        episodes,
        p1_average,
        linewidth=2.0,
        marker="o",
        markevery=marker_interval,
        markersize=4,
        label=f"Predator 1 {window}-episode average",
    )
    ax.plot(
        episodes,
        p2_average,
        linewidth=2.0,
        linestyle="--",
        marker="x",
        markevery=marker_interval,
        markersize=5,
        label=f"Predator 2 {window}-episode average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("VQC-MAPPO training rewards of both predators")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "training_predator_rewards.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_individual_reward_plot(
    episodes: np.ndarray,
    rewards: np.ndarray,
    predator_number: int,
    output_directory: Path,
    window: int | None = None,
    evaluation: bool = False,
) -> Path:
    """Save an unambiguous reward figure for one predator."""
    prefix = "evaluation" if evaluation else "training"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(episodes, rewards, linewidth=1.2, label="Episode reward")

    if window is not None:
        ax.plot(
            episodes,
            moving_average(rewards, window),
            linewidth=2.0,
            label=f"{window}-episode moving average",
        )
        ax.legend()

    ax.set_xlabel("Evaluation episode" if evaluation else "Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(
        f"Predator {predator_number} "
        f"{'evaluation' if evaluation else 'training'} reward"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / f"{prefix}_predator_{predator_number}_reward.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_capture_rate_plot(
    episodes: np.ndarray,
    captures: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot the moving binary prey-capture frequency for analysis."""
    capture_rate = 100.0 * moving_average(captures, window)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(episodes, capture_rate, linewidth=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capture rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Training prey-capture rate (window={window})")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / "training_capture_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_ppo_diagnostics(
    episodes: np.ndarray,
    actor_loss: np.ndarray,
    critic_loss: np.ndarray,
    entropy: np.ndarray,
    approximate_kl: np.ndarray,
    clip_fraction: np.ndarray,
    window: int,
    output_directory: Path,
) -> None:
    """Save MAPPO optimization diagnostics."""
    _save_series_plot(
        episodes,
        actor_loss,
        "Actor loss",
        "MAPPO actor loss",
        "training_actor_loss.png",
        output_directory,
        window,
    )
    _save_series_plot(
        episodes,
        critic_loss,
        "Critic loss",
        "Centralized critic loss",
        "training_critic_loss.png",
        output_directory,
        window,
    )
    _save_series_plot(
        episodes,
        entropy,
        "Policy entropy",
        "Shared actor policy entropy",
        "training_entropy.png",
        output_directory,
        window,
    )
    _save_series_plot(
        episodes,
        approximate_kl,
        "Approximate KL",
        "PPO approximate KL divergence",
        "training_approximate_kl.png",
        output_directory,
        window,
    )
    _save_series_plot(
        episodes,
        100.0 * clip_fraction,
        "Clipped samples (%)",
        "PPO clipping fraction",
        "training_clip_fraction.png",
        output_directory,
        window,
    )


def save_evaluation_agent_reward_plot(
    episodes: np.ndarray,
    predator_1_rewards: np.ndarray,
    predator_2_rewards: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot both predators explicitly during deterministic evaluation."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        predator_1_rewards,
        marker="o",
        linewidth=1.4,
        label="Predator 1 reward",
    )
    ax.plot(
        episodes,
        predator_2_rewards,
        marker="x",
        linestyle="--",
        linewidth=1.4,
        label="Predator 2 reward",
    )
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("VQC-MAPPO evaluation rewards of both predators")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "evaluation_predator_rewards.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_capture_plot(
    episodes: np.ndarray,
    captures: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot binary capture outcome for each deterministic evaluation episode."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(episodes, captures, where="mid", linewidth=1.5)
    ax.scatter(episodes, captures, s=24)
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Prey captured")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Deterministic evaluation prey-capture outcome")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / "evaluation_capture.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_multi_agent_results(
    training_csv: str | Path,
    evaluation_csv: str | Path,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    training_window: int = DEFAULT_TRAINING_WINDOW,
) -> Path:
    """Automatically generate all MAPPO training/evaluation figures."""
    training_rows = load_training_data(training_csv)
    evaluation_rows = load_evaluation_data(evaluation_csv)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    training_episodes = np.asarray(
        [row["episode"] for row in training_rows], dtype=np.int64
    )
    p1_training_rewards = np.asarray(
        [row["predator_1_reward"] for row in training_rows], dtype=np.float64
    )
    p2_training_rewards = np.asarray(
        [row["predator_2_reward"] for row in training_rows], dtype=np.float64
    )
    team_training_rewards = np.asarray(
        [row["team_reward"] for row in training_rows], dtype=np.float64
    )
    training_captures = np.asarray(
        [row["prey_captured"] for row in training_rows], dtype=np.float64
    )

    save_training_agent_reward_plot(
        training_episodes,
        p1_training_rewards,
        p2_training_rewards,
        training_window,
        output_path,
    )
    save_individual_reward_plot(
        training_episodes,
        p1_training_rewards,
        1,
        output_path,
        window=training_window,
    )
    save_individual_reward_plot(
        training_episodes,
        p2_training_rewards,
        2,
        output_path,
        window=training_window,
    )
    _save_series_plot(
        training_episodes,
        team_training_rewards,
        "Team reward",
        "VQC-MAPPO training team reward",
        "training_team_reward.png",
        output_path,
        training_window,
    )
    save_training_capture_rate_plot(
        training_episodes,
        training_captures,
        training_window,
        output_path,
    )
    save_training_ppo_diagnostics(
        training_episodes,
        np.asarray([row["actor_loss"] for row in training_rows]),
        np.asarray([row["critic_loss"] for row in training_rows]),
        np.asarray([row["entropy"] for row in training_rows]),
        np.asarray([row["approximate_kl"] for row in training_rows]),
        np.asarray([row["clip_fraction"] for row in training_rows]),
        training_window,
        output_path,
    )

    evaluation_episodes = np.asarray(
        [row["episode"] for row in evaluation_rows], dtype=np.int64
    )
    p1_evaluation_rewards = np.asarray(
        [row["predator_1_reward"] for row in evaluation_rows], dtype=np.float64
    )
    p2_evaluation_rewards = np.asarray(
        [row["predator_2_reward"] for row in evaluation_rows], dtype=np.float64
    )
    team_evaluation_rewards = np.asarray(
        [row["team_reward"] for row in evaluation_rows], dtype=np.float64
    )
    evaluation_captures = np.asarray(
        [row["prey_captured"] for row in evaluation_rows], dtype=np.float64
    )

    save_evaluation_agent_reward_plot(
        evaluation_episodes,
        p1_evaluation_rewards,
        p2_evaluation_rewards,
        output_path,
    )
    save_individual_reward_plot(
        evaluation_episodes,
        p1_evaluation_rewards,
        1,
        output_path,
        evaluation=True,
    )
    save_individual_reward_plot(
        evaluation_episodes,
        p2_evaluation_rewards,
        2,
        output_path,
        evaluation=True,
    )
    _save_series_plot(
        evaluation_episodes,
        team_evaluation_rewards,
        "Team reward",
        "Deterministic evaluation team reward",
        "evaluation_team_reward.png",
        output_path,
    )
    save_evaluation_capture_plot(
        evaluation_episodes,
        evaluation_captures,
        output_path,
    )

    print(f"Training and evaluation plots saved to: {output_path}")
    return output_path
