"""Plot training results produced by driver_frozenlake_dqn.py.

The plotting script is deliberately separate from the training driver.

It creates four individual figures:
1. episodic reward + moving average,
2. moving success rate,
3. training loss,
4. epsilon schedule.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot FrozenLake VQC-DQN training results"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="results/frozenlake_vqc_dqn.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=100,
    )

    return parser.parse_args()


def moving_average(values, window):
    """Compute a simple moving average with shorter prefix windows."""
    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty_like(values)

    for index in range(
        len(values)
    ):
        start = max(
            0,
            index - window + 1,
        )

        result[index] = np.mean(
            values[start:index + 1]
        )

    return result


def load_results(path):
    """Read training CSV."""
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "steps": int(row["steps"]),
                    "loss": float(row["loss"]),
                    "epsilon": float(row["epsilon"]),
                    "success": float(row["success"]),
                }
            )

    if not rows:
        raise ValueError(
            "The result CSV is empty."
        )

    return rows


def save_reward_plot(
    episodes,
    rewards,
    window,
    output_dir,
):
    avg_reward = moving_average(
        rewards,
        window,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        episodes,
        rewards,
        linewidth=0.8,
        alpha=0.35,
        label="Episode reward",
    )

    ax.plot(
        episodes,
        avg_reward,
        linewidth=2.0,
        label=f"{window}-episode moving average",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("FrozenLake VQC-DQN Reward")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    path = output_dir / "reward.png"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved: {path}")


def save_success_plot(
    episodes,
    successes,
    window,
    output_dir,
):
    success_rate = (
        100.0
        * moving_average(
            successes,
            window,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        episodes,
        success_rate,
        linewidth=2.0,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"FrozenLake Moving Success Rate "
        f"(window={window})"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = output_dir / "success_rate.png"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved: {path}")


def save_loss_plot(
    episodes,
    losses,
    output_dir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        episodes,
        losses,
        linewidth=1.2,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean DQN loss")
    ax.set_title("FrozenLake VQC-DQN Training Loss")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = output_dir / "loss.png"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved: {path}")


def save_epsilon_plot(
    episodes,
    epsilon,
    output_dir,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        episodes,
        epsilon,
        linewidth=2.0,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon-Greedy Exploration Schedule")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    path = output_dir / "epsilon.png"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Saved: {path}")


def plot_results(
    input_csv,
    output_dir="results/plots",
    window=100,
):
    """Generate and save all FrozenLake training-result figures."""
    rows = load_results(input_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.array([row["episode"] for row in rows])
    rewards = np.array([row["reward"] for row in rows])
    losses = np.array([row["loss"] for row in rows])
    epsilon = np.array([row["epsilon"] for row in rows])
    successes = np.array([row["success"] for row in rows])

    save_reward_plot(episodes, rewards, window, output_dir)
    save_success_plot(episodes, successes, window, output_dir)
    save_loss_plot(episodes, losses, output_dir)
    save_epsilon_plot(episodes, epsilon, output_dir)

    return output_dir


def main():
    args = parse_args()
    plot_results(
        input_csv=args.input,
        output_dir=args.output_dir,
        window=args.window,
    )


if __name__ == "__main__":
    main()
