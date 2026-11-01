"""Automatic plotting for multi-agent PredatorPrey VQC-DQN experiments.

This module is intentionally not a command-line program. The training driver
calls :func:`plot_multi_agent_results` automatically after both training and
evaluation have completed and their CSV data files have been saved.

All figures are written to ``result/plots`` by default.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIRECTORY = Path("result/plots")
DEFAULT_TRAINING_WINDOW = 50


def moving_average(
    values: Sequence[float],
    window: int,
) -> np.ndarray:
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
    """Read episode-level training data from CSV."""
    rows: list[dict] = []

    with Path(path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "episode",
            "team_reward",
            "mean_agent_reward",
            "steps",
            "epsilon",
            "prey_captured",
            "predator_1_reward",
            "predator_1_loss",
            "predator_2_reward",
            "predator_2_loss",
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
                    "epsilon": float(row["epsilon"]),
                    "prey_captured": float(row["prey_captured"]),
                    "predator_1_reward": float(row["predator_1_reward"]),
                    "predator_1_loss": float(row["predator_1_loss"]),
                    "predator_2_reward": float(row["predator_2_reward"]),
                    "predator_2_loss": float(row["predator_2_loss"]),
                }
            )

    if not rows:
        raise ValueError("The training CSV is empty.")

    return rows


def load_evaluation_data(path: str | Path) -> list[dict]:
    """Read episode-level greedy-evaluation data from CSV."""
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


def save_training_agent_reward_plot(
    episodes: np.ndarray,
    predator_1_rewards: np.ndarray,
    predator_2_rewards: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot reward traces for both predators during training."""
    predator_1_average = moving_average(predator_1_rewards, window)
    predator_2_average = moving_average(predator_2_rewards, window)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        predator_1_rewards,
        linewidth=0.8,
        alpha=0.25,
        label="Predator 1 episode reward",
    )
    ax.plot(
        episodes,
        predator_2_rewards,
        linewidth=0.8,
        alpha=0.25,
        label="Predator 2 episode reward",
    )
    ax.plot(
        episodes,
        predator_1_average,
        linewidth=2.0,
        label=f"Predator 1 {window}-episode average",
    )
    ax.plot(
        episodes,
        predator_2_average,
        linewidth=2.0,
        label=f"Predator 2 {window}-episode average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training rewards of both predators")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "training_predator_rewards.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_agent_loss_plot(
    episodes: np.ndarray,
    predator_1_losses: np.ndarray,
    predator_2_losses: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot DQN loss traces for both predators during training."""
    predator_1_average = moving_average(predator_1_losses, window)
    predator_2_average = moving_average(predator_2_losses, window)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        predator_1_losses,
        linewidth=0.8,
        alpha=0.25,
        label="Predator 1 episode loss",
    )
    ax.plot(
        episodes,
        predator_2_losses,
        linewidth=0.8,
        alpha=0.25,
        label="Predator 2 episode loss",
    )
    ax.plot(
        episodes,
        predator_1_average,
        linewidth=2.0,
        label=f"Predator 1 {window}-episode average",
    )
    ax.plot(
        episodes,
        predator_2_average,
        linewidth=2.0,
        label=f"Predator 2 {window}-episode average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean DQN loss")
    ax.set_title("Training losses of both predators")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "training_predator_losses.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_team_reward_plot(
    episodes: np.ndarray,
    team_rewards: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot team reward during training."""
    team_average = moving_average(team_rewards, window)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        team_rewards,
        linewidth=0.8,
        alpha=0.3,
        label="Episode team reward",
    )
    ax.plot(
        episodes,
        team_average,
        linewidth=2.0,
        label=f"{window}-episode moving average",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team reward")
    ax.set_title("Training team reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "training_team_reward.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_capture_rate_plot(
    episodes: np.ndarray,
    captures: np.ndarray,
    window: int,
    output_directory: Path,
) -> Path:
    """Plot moving prey-capture rate during training."""
    capture_rate = 100.0 * moving_average(captures, window)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(episodes, capture_rate, linewidth=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Capture rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Training prey capture rate (window={window})")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / "training_capture_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_epsilon_plot(
    episodes: np.ndarray,
    epsilon: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot the epsilon-greedy exploration schedule."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(episodes, epsilon, linewidth=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon-greedy exploration schedule")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / "training_epsilon.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_agent_reward_plot(
    episodes: np.ndarray,
    predator_1_rewards: np.ndarray,
    predator_2_rewards: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot greedy-evaluation rewards for both predators."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        predator_1_rewards,
        marker="o",
        linewidth=1.5,
        label="Predator 1 reward",
    )
    ax.plot(
        episodes,
        predator_2_rewards,
        marker="o",
        linewidth=1.5,
        label="Predator 2 reward",
    )
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Evaluation rewards of both predators")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "evaluation_predator_rewards.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_team_reward_plot(
    episodes: np.ndarray,
    team_rewards: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot greedy-evaluation team reward."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        episodes,
        team_rewards,
        marker="o",
        linewidth=1.5,
    )
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Team reward")
    ax.set_title("Evaluation team reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = output_directory / "evaluation_team_reward.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation_capture_plot(
    episodes: np.ndarray,
    captures: np.ndarray,
    output_directory: Path,
) -> Path:
    """Plot prey-capture outcome for every evaluation episode."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.step(
        episodes,
        captures,
        where="mid",
        linewidth=1.5,
    )
    ax.scatter(episodes, captures, s=24)
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Prey captured")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Evaluation prey-capture outcome")
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
    """Automatically generate all training and evaluation figures."""
    training_rows = load_training_data(training_csv)
    evaluation_rows = load_evaluation_data(evaluation_csv)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    training_episodes = np.asarray(
        [row["episode"] for row in training_rows],
        dtype=np.int64,
    )
    training_predator_1_rewards = np.asarray(
        [row["predator_1_reward"] for row in training_rows],
        dtype=np.float64,
    )
    training_predator_2_rewards = np.asarray(
        [row["predator_2_reward"] for row in training_rows],
        dtype=np.float64,
    )
    training_predator_1_losses = np.asarray(
        [row["predator_1_loss"] for row in training_rows],
        dtype=np.float64,
    )
    training_predator_2_losses = np.asarray(
        [row["predator_2_loss"] for row in training_rows],
        dtype=np.float64,
    )
    training_team_rewards = np.asarray(
        [row["team_reward"] for row in training_rows],
        dtype=np.float64,
    )
    training_captures = np.asarray(
        [row["prey_captured"] for row in training_rows],
        dtype=np.float64,
    )
    training_epsilon = np.asarray(
        [row["epsilon"] for row in training_rows],
        dtype=np.float64,
    )

    save_training_agent_reward_plot(
        training_episodes,
        training_predator_1_rewards,
        training_predator_2_rewards,
        training_window,
        output_path,
    )
    save_training_agent_loss_plot(
        training_episodes,
        training_predator_1_losses,
        training_predator_2_losses,
        training_window,
        output_path,
    )
    save_training_team_reward_plot(
        training_episodes,
        training_team_rewards,
        training_window,
        output_path,
    )
    save_training_capture_rate_plot(
        training_episodes,
        training_captures,
        training_window,
        output_path,
    )
    save_training_epsilon_plot(
        training_episodes,
        training_epsilon,
        output_path,
    )

    evaluation_episodes = np.asarray(
        [row["episode"] for row in evaluation_rows],
        dtype=np.int64,
    )
    evaluation_predator_1_rewards = np.asarray(
        [row["predator_1_reward"] for row in evaluation_rows],
        dtype=np.float64,
    )
    evaluation_predator_2_rewards = np.asarray(
        [row["predator_2_reward"] for row in evaluation_rows],
        dtype=np.float64,
    )
    evaluation_team_rewards = np.asarray(
        [row["team_reward"] for row in evaluation_rows],
        dtype=np.float64,
    )
    evaluation_captures = np.asarray(
        [row["prey_captured"] for row in evaluation_rows],
        dtype=np.float64,
    )

    save_evaluation_agent_reward_plot(
        evaluation_episodes,
        evaluation_predator_1_rewards,
        evaluation_predator_2_rewards,
        output_path,
    )
    save_evaluation_team_reward_plot(
        evaluation_episodes,
        evaluation_team_rewards,
        output_path,
    )
    save_evaluation_capture_plot(
        evaluation_episodes,
        evaluation_captures,
        output_path,
    )

    print(f"Training and evaluation plots saved to: {output_path}")
    return output_path
