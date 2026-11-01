"""Quantum ML Project - Multi-agent VQC-DQN on ma-gym PredatorPrey5x5-v0.

This is the executable user/driver program for the first multi-agent
extension of the single-agent QDRL implementation.

Learning design
---------------
The implementation uses independent DQN learners for the two predators.
Each predator receives its own local observation from ma-gym, owns an
independent online VQC-DQN network, target network, replay buffer, and
optimizer, and learns from its individual transition stream. Cooperation is
provided by the environment's shared predator-prey reward structure.

The reusable ``VQC_framework`` is imported without modification.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Sequence

import gym
import numpy as np
import torch

from VQC_framework import VQC
from multi_agent_vqc_dqn import IndependentVQCDQNAgent
from plot_multi_agent_results import plot_multi_agent_results
from util import (
    load_quantum_circuit_config,
    resolve_training_device,
    set_seed,
)


PROJECT_NAME = "Quantum ML Project"
ENVIRONMENT_ID = "ma_gym:PredatorPrey5x5-v0"
DEFAULT_CONFIG = "config_quantum_circuit.yaml"

MODEL_OUTPUT_PATH = Path("model/predator_prey_multi_agent_vqc_dqn.pt")
TRAINING_DATA_PATH = Path("result/predator_prey_training_data.csv")
EVALUATION_DATA_PATH = Path("result/predator_prey_evaluation_data.csv")
CIRCUIT_OUTPUT_DIRECTORY = Path("result/circuits")
PLOT_OUTPUT_DIRECTORY = Path("result/plots")
PLOT_WINDOW = 50


def parse_arguments() -> argparse.Namespace:
    """Parse multi-agent QDRL runtime arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Independent multi-agent VQC-DQN on "
            "ma-gym PredatorPrey5x5-v0"
        )
    )

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--replay-size", type=int, default=2000)
    parser.add_argument("--min-replay-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=20)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=200.0)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--render-evaluation",
        action="store_true",
        help="Render greedy evaluation episodes using ma-gym.",
    )

    return parser.parse_args()


def epsilon_for_episode(
    args: argparse.Namespace,
    episode_index: int,
) -> float:
    """Return exponentially decayed epsilon for one training episode."""
    return (
        args.epsilon_end
        + (
            args.epsilon_start
            - args.epsilon_end
        )
        * math.exp(
            -episode_index
            / args.epsilon_decay
        )
    )


def create_predator_prey_environment(
    seed: int | None = None,
):
    """Create and optionally seed ma-gym PredatorPrey5x5-v0."""
    environment = gym.make(
        ENVIRONMENT_ID
    )

    if seed is not None and hasattr(environment, "seed"):
        environment.seed(seed)

    return environment


def reset_multi_agent_environment(
    environment,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Reset ma-gym while tolerating legacy and newer Gym reset APIs."""
    if seed is not None and hasattr(environment, "seed"):
        environment.seed(seed)

    reset_result = environment.reset()

    if (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    ):
        observations = reset_result[0]
    else:
        observations = reset_result

    return [
        np.asarray(observation, dtype=np.float32)
        for observation in observations
    ]


def step_multi_agent_environment(
    environment,
    actions: Sequence[int],
) -> tuple[
    list[np.ndarray],
    list[float],
    list[bool],
    dict,
]:
    """Execute one joint action with legacy/new Gym API compatibility."""
    step_result = environment.step(
        list(actions)
    )

    if len(step_result) == 4:
        observations, rewards, dones, info = step_result
    elif len(step_result) == 5:
        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = step_result

        terminated_array = np.asarray(
            terminated,
            dtype=bool,
        )
        truncated_array = np.asarray(
            truncated,
            dtype=bool,
        )
        dones = np.logical_or(
            terminated_array,
            truncated_array,
        ).tolist()
    else:
        raise RuntimeError(
            "Unexpected environment.step() return format."
        )

    return (
        [
            np.asarray(observation, dtype=np.float32)
            for observation in observations
        ],
        [float(reward) for reward in rewards],
        [bool(done) for done in dones],
        dict(info),
    )


def environment_specification(environment) -> tuple[int, list[int], list[int]]:
    """Return agent count, local observation dimensions, and action counts."""
    base_environment = getattr(
        environment,
        "unwrapped",
        environment,
    )

    n_agents = int(
        base_environment.n_agents
    )

    observation_dimensions = [
        int(
            np.prod(
                environment.observation_space[agent_id].shape
            )
        )
        for agent_id in range(n_agents)
    ]

    action_dimensions = [
        int(
            environment.action_space[agent_id].n
        )
        for agent_id in range(n_agents)
    ]

    return (
        n_agents,
        observation_dimensions,
        action_dimensions,
    )


def build_independent_agents(
    n_agents: int,
    observation_dimensions: Sequence[int],
    action_dimensions: Sequence[int],
    circuit_config: dict,
    training_device: str,
    args: argparse.Namespace,
) -> list[IndependentVQCDQNAgent]:
    """Create one independent VQC-DQN learner per predator."""
    agents = []

    for agent_id in range(n_agents):
        online_vqc = VQC(
            circuit_config,
            device=training_device,
        )

        target_vqc = VQC(
            circuit_config,
            device=training_device,
        )

        # VQC initializes its own quantum parameters reproducibly. Resetting
        # the Torch seed here gives each agent distinct classical encoder and
        # Q-head initialization while leaving VQC_framework unchanged.
        torch.manual_seed(
            args.seed + 1000 + agent_id
        )

        agent = IndependentVQCDQNAgent(
            agent_id=agent_id,
            online_vqc=online_vqc,
            target_vqc=target_vqc,
            observation_dim=observation_dimensions[agent_id],
            action_dim=action_dimensions[agent_id],
            learning_rate=args.lr,
            discount_factor=args.gamma,
            replay_capacity=args.replay_size,
            batch_size=args.batch_size,
            minimum_replay_size=args.min_replay_size,
        )

        agents.append(agent)

    return agents


def prey_captured_from_info(info: dict) -> int:
    """Return 1 when ma-gym reports that no prey remains alive."""
    prey_alive = info.get(
        "prey_alive"
    )

    if prey_alive is None:
        return 0

    return int(
        not any(bool(status) for status in prey_alive)
    )


def save_training_history(
    history: list[dict],
    output_path: str,
) -> None:
    """Write episode-level multi-agent training records to CSV."""
    if not history:
        raise ValueError(
            "history is empty; no training records are available to save"
        )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    print(
        f"Training history saved to: {path}"
    )


def save_multi_agent_checkpoint(
    agents: Sequence[IndependentVQCDQNAgent],
    circuit_config: dict,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """Save all independent learners in one checkpoint file."""
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "project_name": PROJECT_NAME,
        "environment_id": ENVIRONMENT_ID,
        "quantum_circuit_config": circuit_config,
        "training_config": {
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "replay_size": args.replay_size,
            "minimum_replay_size": args.min_replay_size,
            "target_update": args.target_update,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "agents": [
            agent.checkpoint_state()
            for agent in agents
        ],
    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Multi-agent checkpoint saved to: {path}"
    )


def save_evaluation_history(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Write episode-level greedy-evaluation records to CSV."""
    if not history:
        raise ValueError(
            "history is empty; no evaluation records are available to save"
        )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    print(
        f"Evaluation data saved to: {path}"
    )


def summarize_evaluation(
    history: Sequence[dict],
) -> dict:
    """Return aggregate values computed from evaluation episode records."""
    return {
        "mean_team_reward": float(
            np.mean([row["team_reward"] for row in history])
        ),
        "capture_rate": float(
            np.mean([row["prey_captured"] for row in history])
        ),
    }


def save_quantum_circuit_visualizations(
    agents: Sequence[IndependentVQCDQNAgent],
    sample_observations: Sequence[Sequence[float]],
    output_directory: str,
) -> None:
    """Save the VQC circuit used by every independent predator learner."""
    output_path = Path(
        output_directory
    )
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for agent, observation in zip(
        agents,
        sample_observations,
    ):
        with torch.no_grad():
            quantum_input = (
                agent.online_network.encode_observation(
                    observation
                )
            )

        circuit_path = output_path / (
            f"predator_{agent.agent_id + 1}_vqc_circuit.png"
        )

        agent.online_network.vqc.visualize(
            quantum_input,
            save_path=str(circuit_path),
            show=False,
        )

        print(
            f"Predator {agent.agent_id + 1} circuit saved to: "
            f"{circuit_path}"
        )


def train_multi_agent_policy(
    environment,
    agents: Sequence[IndependentVQCDQNAgent],
    action_rngs: Sequence[random.Random],
    args: argparse.Namespace,
) -> tuple[list[dict], list[np.ndarray]]:
    """Train all independent VQC-DQN predators in the shared environment."""
    history = []

    sample_observations = reset_multi_agent_environment(
        environment,
        seed=args.seed,
    )

    print("\nTraining")
    print("--------")

    for episode in range(
        1,
        args.episodes + 1,
    ):
        observations = reset_multi_agent_environment(
            environment,
            seed=args.seed + episode,
        )

        epsilon = epsilon_for_episode(
            args,
            episode - 1,
        )

        episode_agent_rewards = np.zeros(
            len(agents),
            dtype=np.float64,
        )
        episode_agent_losses = [
            [] for _ in agents
        ]

        steps_taken = 0
        final_info = {}

        for step in range(
            1,
            args.max_steps + 1,
        ):
            joint_actions = [
                agent.select_action(
                    observations[agent_id],
                    epsilon,
                    action_rngs[agent_id],
                )
                for agent_id, agent in enumerate(agents)
            ]

            (
                next_observations,
                rewards,
                dones,
                info,
            ) = step_multi_agent_environment(
                environment,
                joint_actions,
            )

            final_info = info

            for agent_id, agent in enumerate(agents):
                agent.store_transition(
                    observation=observations[agent_id],
                    action=joint_actions[agent_id],
                    reward=rewards[agent_id],
                    next_observation=next_observations[agent_id],
                    done=dones[agent_id],
                )

                loss = agent.optimize_model()
                if loss is not None:
                    episode_agent_losses[agent_id].append(
                        loss
                    )

                episode_agent_rewards[agent_id] += rewards[agent_id]

            observations = next_observations
            steps_taken = step

            if all(dones):
                break

        if episode % args.target_update == 0:
            for agent in agents:
                agent.synchronize_target_network()

        mean_agent_losses = [
            float(np.mean(losses))
            if losses
            else 0.0
            for losses in episode_agent_losses
        ]

        record = {
            "episode": episode,
            "team_reward": float(
                episode_agent_rewards.sum()
            ),
            "mean_agent_reward": float(
                episode_agent_rewards.mean()
            ),
            "steps": steps_taken,
            "epsilon": epsilon,
            "prey_captured": prey_captured_from_info(
                final_info
            ),
        }

        for agent_id in range(len(agents)):
            record[
                f"predator_{agent_id + 1}_reward"
            ] = float(
                episode_agent_rewards[agent_id]
            )
            record[
                f"predator_{agent_id + 1}_loss"
            ] = mean_agent_losses[agent_id]

        history.append(record)

        if (
            episode == 1
            or episode % 25 == 0
            or episode == args.episodes
        ):
            loss_text = ", ".join(
                f"P{agent_id + 1} loss={mean_agent_losses[agent_id]:.5f}"
                for agent_id in range(len(agents))
            )

            print(
                f"Episode {episode:04d}/{args.episodes} | "
                f"team_reward={record['team_reward']:.3f} | "
                f"steps={steps_taken:03d} | "
                f"epsilon={epsilon:.3f} | "
                f"captured={record['prey_captured']} | "
                f"{loss_text}"
            )

    return history, sample_observations


def evaluate_multi_agent_policy(
    agents: Sequence[IndependentVQCDQNAgent],
    args: argparse.Namespace,
) -> list[dict]:
    """Run greedy joint-policy evaluation and return episode-level records."""
    environment = create_predator_prey_environment(
        seed=args.seed + 100_000,
    )

    history: list[dict] = []

    print("\nEvaluation")
    print("----------")

    for episode in range(1, args.eval_episodes + 1):
        observations = reset_multi_agent_environment(
            environment,
            seed=args.seed + 100_000 + episode,
        )

        episode_agent_rewards = np.zeros(
            len(agents),
            dtype=np.float64,
        )
        final_info = {}
        steps_taken = 0

        for step in range(1, args.max_steps + 1):
            joint_actions = []

            for agent_id, agent in enumerate(agents):
                with torch.no_grad():
                    q_values = agent.online_network(
                        observations[agent_id]
                    )

                joint_actions.append(
                    int(q_values.argmax(dim=-1).item())
                )

            (
                observations,
                rewards,
                dones,
                info,
            ) = step_multi_agent_environment(
                environment,
                joint_actions,
            )

            final_info = info
            episode_agent_rewards += np.asarray(
                rewards,
                dtype=np.float64,
            )
            steps_taken = step

            if args.render_evaluation:
                environment.render()

            if all(dones):
                break

        captured = prey_captured_from_info(final_info)

        record = {
            "episode": episode,
            "team_reward": float(episode_agent_rewards.sum()),
            "mean_agent_reward": float(episode_agent_rewards.mean()),
            "steps": steps_taken,
            "prey_captured": captured,
        }

        for agent_id in range(len(agents)):
            record[
                f"predator_{agent_id + 1}_reward"
            ] = float(episode_agent_rewards[agent_id])

        history.append(record)

        print(
            f"Evaluation episode {episode:02d}/"
            f"{args.eval_episodes} | "
            f"team_reward={record['team_reward']:.3f} | "
            f"captured={captured}"
        )

    environment.close()
    return history


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    training_device = resolve_training_device(
        args.device
    )

    circuit_config = load_quantum_circuit_config(
        args.config
    )

    environment = create_predator_prey_environment(
        seed=args.seed
    )

    (
        n_agents,
        observation_dimensions,
        action_dimensions,
    ) = environment_specification(
        environment
    )

    agents = build_independent_agents(
        n_agents=n_agents,
        observation_dimensions=observation_dimensions,
        action_dimensions=action_dimensions,
        circuit_config=circuit_config,
        training_device=training_device,
        args=args,
    )

    action_rngs = [
        random.Random(
            args.seed + 10_000 + agent_id
        )
        for agent_id in range(n_agents)
    ]

    print(f"\n{PROJECT_NAME} - Multi-Agent VQC-DQN")
    print("========================================")
    print(f"Environment         : {ENVIRONMENT_ID}")
    print(f"Agents              : {n_agents}")
    print(f"Observation dims    : {observation_dimensions}")
    print(f"Action dims         : {action_dimensions}")
    print(f"Episodes            : {args.episodes}")
    print(f"Learning rate       : {args.lr}")
    print(f"Gamma               : {args.gamma}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Replay capacity     : {args.replay_size}")
    print(f"Target update       : {args.target_update} episodes")
    print(f"Requested device    : {args.device}")
    print(f"Resolved device     : {training_device}")

    print("\nQuantum circuit")
    print("---------------")
    print(agents[0].online_network.vqc.summary())

    print("\nIndependent learners")
    print("--------------------")
    for agent in agents:
        print(
            f"Predator {agent.agent_id + 1}: "
            f"trainable parameters="
            f"{agent.trainable_parameter_count()}"
        )

    (
        history,
        sample_observations,
    ) = train_multi_agent_policy(
        environment,
        agents,
        action_rngs,
        args,
    )

    environment.close()

    evaluation_history = evaluate_multi_agent_policy(
        agents,
        args,
    )

    evaluation_summary = summarize_evaluation(
        evaluation_history
    )

    print("\nSaving experiment data")
    print("----------------------")
    save_training_history(
        history,
        str(TRAINING_DATA_PATH),
    )
    save_evaluation_history(
        evaluation_history,
        EVALUATION_DATA_PATH,
    )

    save_multi_agent_checkpoint(
        agents,
        circuit_config,
        args,
        str(MODEL_OUTPUT_PATH),
    )

    save_quantum_circuit_visualizations(
        agents,
        sample_observations,
        str(CIRCUIT_OUTPUT_DIRECTORY),
    )

    print("\nGenerating training and evaluation plots")
    print("----------------------------------------")
    plot_multi_agent_results(
        training_csv=TRAINING_DATA_PATH,
        evaluation_csv=EVALUATION_DATA_PATH,
        output_directory=PLOT_OUTPUT_DIRECTORY,
        training_window=PLOT_WINDOW,
    )

    print("\nEvaluation summary")
    print("------------------")
    print(
        f"Mean team reward : "
        f"{evaluation_summary['mean_team_reward']:.3f}"
    )
    print(
        f"Capture rate     : "
        f"{evaluation_summary['capture_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
