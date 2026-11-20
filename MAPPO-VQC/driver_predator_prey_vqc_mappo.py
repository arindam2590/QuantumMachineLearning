"""Quantum ML Project - VQC-MAPPO on ma-gym PredatorPrey5x5-v0.

The executable follows the same organization as the earlier single-agent
FrozenLake and multi-agent DQN projects:

* ``util.py`` owns argument parsing and runtime helpers;
* ``multi_agent_vqc_mappo.py`` owns all MAPPO learning components;
* this driver owns environment interaction, training, evaluation, saving,
  and orchestration;
* ``plot_multi_agent_results.py`` is called automatically after training;
* ``VQC_framework`` is imported without modification.

Learning design
---------------
The two homogeneous predators use one parameter-shared VQC actor for
decentralized action selection. During training, a centralized classical
critic receives the concatenation of both local observations. Generalized
advantage estimation is computed separately per trajectory, while PPO updates
are performed from a combined multi-episode rollout rather than after every
single episode.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import gym
import numpy as np
import torch

from multi_agent_vqc_mappo import (
    CentralizedValueCritic,
    MAPPOEpisodeBuffer,
    VQCMAPPOLearner,
    VQCSharedActor,
)
from plot_multi_agent_results import plot_multi_agent_results
from util import (
    configure_runtime_warnings,
    entropy_coefficient_for_episode,
    load_quantum_circuit_config,
    parse_args,
    resolve_training_device,
    set_seed,
)

# Configure the two known compatibility warnings before PennyLane import.
configure_runtime_warnings()

from VQC_framework import VQC


PROJECT_NAME = "Quantum ML Project"
ALGORITHM_NAME = "VQC-MAPPO"
ENVIRONMENT_ID = "ma_gym:PredatorPrey5x5-v0"
MODEL_OUTPUT_PATH = Path("model/predator_prey_multi_agent_vqc_mappo.pt")
TRAINING_DATA_PATH = Path("result/predator_prey_mappo_training_data.csv")
EVALUATION_DATA_PATH = Path("result/predator_prey_mappo_evaluation_data.csv")
CIRCUIT_OUTPUT_DIRECTORY = Path("result/circuits")
PLOT_OUTPUT_DIRECTORY = Path("result/plots")
PLOT_WINDOW = 25


def create_predator_prey_environment(seed: int | None = None):
    """Create and optionally seed ``PredatorPrey5x5-v0``."""
    environment = gym.make(ENVIRONMENT_ID)

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
) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
    """Execute one joint action with legacy/new Gym API compatibility."""
    step_result = environment.step(list(actions))

    if len(step_result) == 4:
        observations, rewards, dones, info = step_result
    elif len(step_result) == 5:
        observations, rewards, terminated, truncated, info = step_result
        dones = np.logical_or(
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
        ).tolist()
    else:
        raise RuntimeError("Unexpected environment.step() return format.")

    return (
        [
            np.asarray(observation, dtype=np.float32)
            for observation in observations
        ],
        [float(reward) for reward in rewards],
        [bool(done) for done in dones],
        dict(info),
    )


def environment_specification(
    environment,
) -> tuple[int, list[int], list[int]]:
    """Return agent count, local observation dimensions, and action counts."""
    base_environment = getattr(environment, "unwrapped", environment)
    n_agents = int(base_environment.n_agents)

    observation_dimensions = [
        int(np.prod(environment.observation_space[agent_id].shape))
        for agent_id in range(n_agents)
    ]
    action_dimensions = [
        int(environment.action_space[agent_id].n)
        for agent_id in range(n_agents)
    ]

    return n_agents, observation_dimensions, action_dimensions


def validate_parameter_sharing_dimensions(
    observation_dimensions: Sequence[int],
    action_dimensions: Sequence[int],
) -> None:
    """Ensure homogeneous dimensions required by the shared MAPPO actor."""
    if len(set(int(value) for value in observation_dimensions)) != 1:
        raise ValueError(
            "Parameter sharing requires identical observation dimensions "
            "for all predators."
        )
    if len(set(int(value) for value in action_dimensions)) != 1:
        raise ValueError(
            "Parameter sharing requires identical action dimensions for all "
            "predators."
        )


def build_mappo_learner(
    n_agents: int,
    observation_dimensions: Sequence[int],
    action_dimensions: Sequence[int],
    circuit_config: dict,
    training_device: str,
    args,
) -> VQCMAPPOLearner:
    """Construct the shared VQC actor and centralized MAPPO critic."""
    validate_parameter_sharing_dimensions(
        observation_dimensions,
        action_dimensions,
    )

    shared_vqc = VQC(
        circuit_config,
        device=training_device,
    )

    # The VQC class initializes its quantum parameters reproducibly. Reset the
    # seed before constructing the classical actor layers so their parameters
    # follow the experiment seed while VQC_framework remains unchanged.
    torch.manual_seed(args.seed + 1000)
    actor = VQCSharedActor(
        vqc=shared_vqc,
        observation_dim=observation_dimensions[0],
        action_dim=action_dimensions[0],
        n_agents=n_agents,
        hidden_dim=args.actor_hidden_dim,
    ).to(torch.device(training_device))

    torch.manual_seed(args.seed + 2000)
    critic = CentralizedValueCritic(
        centralized_observation_dim=sum(observation_dimensions),
        hidden_dim=args.critic_hidden_dim,
    ).to(torch.device(training_device))

    return VQCMAPPOLearner(
        actor=actor,
        critic=critic,
        actor_learning_rate=args.lr,
        critic_learning_rate=args.critic_lr,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coefficient=args.entropy_coef,
        value_loss_coefficient=args.value_loss_coef,
        max_gradient_norm=args.max_grad_norm,
    )


def prey_captured_from_info(info: dict) -> int:
    """Return binary prey-capture status from ma-gym's episode information."""
    prey_alive = info.get("prey_alive")

    if prey_alive is None:
        return 0

    if isinstance(prey_alive, dict):
        alive_values = list(prey_alive.values())
    else:
        alive_values = list(prey_alive)

    return int(not any(bool(status) for status in alive_values))


def save_training_history(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Write episode-level MAPPO training records to CSV."""
    if not history:
        raise ValueError("history is empty; no training records are available")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"Training history saved to: {path}")


def save_evaluation_history(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Write episode-level deterministic evaluation records to CSV."""
    if not history:
        raise ValueError("history is empty; no evaluation records are available")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"Evaluation data saved to: {path}")


def save_mappo_checkpoint(
    learner: VQCMAPPOLearner,
    circuit_config: dict,
    observation_dimensions: Sequence[int],
    action_dimensions: Sequence[int],
    args,
    output_path: str | Path,
) -> None:
    """Save the shared actor, centralized critic, optimizers, and metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "project_name": PROJECT_NAME,
        "algorithm": ALGORITHM_NAME,
        "environment_id": ENVIRONMENT_ID,
        "quantum_circuit_config": circuit_config,
        "observation_dimensions": list(observation_dimensions),
        "action_dimensions": list(action_dimensions),
        "training_config": {
            "episodes": args.episodes,
            "actor_learning_rate": args.lr,
            "critic_learning_rate": args.critic_lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": args.clip_epsilon,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "entropy_coefficient_initial": args.entropy_coef,
            "entropy_coefficient_final": args.entropy_coef_final,
            "rollout_episodes": args.rollout_episodes,
            "value_loss_coefficient": args.value_loss_coef,
            "max_gradient_norm": args.max_grad_norm,
            "actor_hidden_dim": args.actor_hidden_dim,
            "critic_hidden_dim": args.critic_hidden_dim,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "learner": learner.checkpoint_state(),
    }

    torch.save(checkpoint, path)
    print(f"MAPPO checkpoint saved to: {path}")


def save_quantum_circuit_visualizations(
    learner: VQCMAPPOLearner,
    sample_observations: Sequence[Sequence[float]],
    output_directory: str | Path,
) -> None:
    """Save the shared VQC circuit using each predator's encoded sample."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    for agent_id, observation in enumerate(sample_observations):
        with torch.no_grad():
            quantum_input = learner.actor.encode_observation(
                observation,
                agent_id,
            )

        circuit_path = output_path / (
            f"predator_{agent_id + 1}_vqc_circuit.png"
        )
        learner.actor.vqc.visualize(
            quantum_input,
            save_path=str(circuit_path),
            show=False,
        )
        print(
            f"Predator {agent_id + 1} shared-actor circuit saved to: "
            f"{circuit_path}"
        )


def train_multi_agent_policy(
    environment,
    learner: VQCMAPPOLearner,
    args,
) -> tuple[list[dict], list[np.ndarray]]:
    """Train VQC-MAPPO from combined multi-episode on-policy rollouts."""
    history: list[dict] = []
    sample_observations = reset_multi_agent_environment(
        environment,
        seed=args.seed,
    )

    pending_trajectories: list[MAPPOEpisodeBuffer] = []
    pending_records: list[dict] = []
    last_update_metrics = None
    rollout_update_index = 0

    print("\nTraining")
    print("--------")

    for episode in range(1, args.episodes + 1):
        observations = reset_multi_agent_environment(
            environment,
            seed=args.seed + episode,
        )
        trajectory = MAPPOEpisodeBuffer(
            n_agents=learner.n_agents
        )

        episode_agent_rewards = np.zeros(
            learner.n_agents,
            dtype=np.float64,
        )
        final_info: dict = {}
        steps_taken = 0
        final_done = False

        for step in range(1, args.max_steps + 1):
            centralized_observation = learner.centralized_observation(
                observations
            )
            state_value = learner.estimate_value(
                centralized_observation
            )
            joint_actions, log_probabilities = learner.select_actions(
                observations,
                deterministic=False,
            )

            (
                next_observations,
                rewards,
                dones,
                info,
            ) = step_multi_agent_environment(
                environment,
                joint_actions,
            )

            # PredatorPrey provides a shared cooperative reward to each
            # predator. Mean reward avoids counting that same signal twice.
            team_step_reward = float(np.mean(rewards))
            joint_done = bool(all(dones))

            trajectory.add(
                local_observations=observations,
                centralized_observation=centralized_observation,
                actions=joint_actions,
                log_probabilities=log_probabilities,
                team_reward=team_step_reward,
                value=state_value,
                done=joint_done,
            )

            episode_agent_rewards += np.asarray(
                rewards,
                dtype=np.float64,
            )
            observations = next_observations
            final_info = info
            steps_taken = step
            final_done = joint_done

            if joint_done:
                break

        trajectory.finalize(
            final_centralized_observation=learner.centralized_observation(
                observations
            ),
            final_done=final_done,
        )

        captured = prey_captured_from_info(final_info)
        team_episode_reward = float(episode_agent_rewards.mean())

        # Metrics are filled after the rollout containing this episode has
        # completed its PPO optimization. Repeating rollout-level metrics on
        # its member episodes keeps the CSV episode-aligned for plotting.
        record = {
            "episode": episode,
            "team_reward": team_episode_reward,
            "mean_agent_reward": team_episode_reward,
            "steps": steps_taken,
            "prey_captured": captured,
            "predator_1_reward": float(episode_agent_rewards[0]),
            "predator_2_reward": float(episode_agent_rewards[1]),
            "rollout_update": 0,
            "entropy_coefficient": float("nan"),
            "actor_loss": float("nan"),
            "critic_loss": float("nan"),
            "entropy": float("nan"),
            "approximate_kl": float("nan"),
            "clip_fraction": float("nan"),
        }
        history.append(record)
        pending_records.append(record)
        pending_trajectories.append(trajectory)

        rollout_ready = (
            len(pending_trajectories) >= args.rollout_episodes
            or episode == args.episodes
        )

        if rollout_ready:
            current_entropy_coefficient = entropy_coefficient_for_episode(
                initial_coefficient=args.entropy_coef,
                final_coefficient=args.entropy_coef_final,
                episode=episode,
                total_episodes=args.episodes,
            )
            learner.set_entropy_coefficient(
                current_entropy_coefficient
            )
            last_update_metrics = learner.update(
                pending_trajectories
            )
            rollout_update_index += 1

            for pending_record in pending_records:
                pending_record["rollout_update"] = rollout_update_index
                pending_record["entropy_coefficient"] = (
                    current_entropy_coefficient
                )
                pending_record["actor_loss"] = (
                    last_update_metrics.actor_loss
                )
                pending_record["critic_loss"] = (
                    last_update_metrics.critic_loss
                )
                pending_record["entropy"] = last_update_metrics.entropy
                pending_record["approximate_kl"] = (
                    last_update_metrics.approximate_kl
                )
                pending_record["clip_fraction"] = (
                    last_update_metrics.clip_fraction
                )

            pending_trajectories.clear()
            pending_records.clear()

        if episode == 1 or episode % 25 == 0 or episode == args.episodes:
            recent_window = history[-25:]
            capture_count_25 = sum(
                int(item["prey_captured"]) for item in recent_window
            )

            if last_update_metrics is None:
                update_text = (
                    "update=000 | actor_loss=n/a | critic_loss=n/a | "
                    "entropy=n/a"
                )
            else:
                update_text = (
                    f"update={rollout_update_index:03d} | "
                    f"actor_loss={last_update_metrics.actor_loss:.5f} | "
                    f"critic_loss={last_update_metrics.critic_loss:.5f} | "
                    f"entropy={last_update_metrics.entropy:.4f}"
                )

            print(
                f"Episode {episode:04d}/{args.episodes} | "
                f"team_reward={team_episode_reward:.3f} | "
                f"capture25={capture_count_25:02d}/{len(recent_window):02d} | "
                f"steps={steps_taken:03d} | "
                f"{update_text}"
            )

    return history, sample_observations


def evaluate_multi_agent_policy(
    learner: VQCMAPPOLearner,
    args,
) -> list[dict]:
    """Run deterministic decentralized evaluation after training."""
    environment = create_predator_prey_environment(
        seed=args.seed + 100_000,
    )
    history: list[dict] = []

    learner.actor.eval()
    learner.critic.eval()

    print("\nEvaluation")
    print("----------")

    for episode in range(1, args.eval_episodes + 1):
        observations = reset_multi_agent_environment(
            environment,
            seed=args.seed + 100_000 + episode,
        )
        episode_agent_rewards = np.zeros(
            learner.n_agents,
            dtype=np.float64,
        )
        final_info: dict = {}
        steps_taken = 0

        for step in range(1, args.max_steps + 1):
            joint_actions, _ = learner.select_actions(
                observations,
                deterministic=True,
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

            episode_agent_rewards += np.asarray(
                rewards,
                dtype=np.float64,
            )
            final_info = info
            steps_taken = step

            if args.render_evaluation:
                environment.render()

            if all(dones):
                break

        captured = prey_captured_from_info(final_info)
        team_episode_reward = float(episode_agent_rewards.mean())

        record = {
            "episode": episode,
            "team_reward": team_episode_reward,
            "mean_agent_reward": team_episode_reward,
            "steps": steps_taken,
            "prey_captured": captured,
            "predator_1_reward": float(episode_agent_rewards[0]),
            "predator_2_reward": float(episode_agent_rewards[1]),
        }
        history.append(record)

        if episode == 1 or episode % 25 == 0 or episode == args.eval_episodes:
            recent_window = history[-25:]
            capture_count_25 = sum(
                int(item["prey_captured"]) for item in recent_window
            )
            print(
                f"Evaluation episode {episode:03d}/{args.eval_episodes} | "
                f"team_reward={team_episode_reward:.3f} | "
                f"capture25={capture_count_25:02d}/{len(recent_window):02d}"
            )

    environment.close()
    return history


def summarize_evaluation(history: Sequence[dict]) -> dict:
    """Return aggregate deterministic-evaluation statistics."""
    capture_count = int(
        sum(int(row["prey_captured"]) for row in history)
    )
    return {
        "mean_team_reward": float(
            np.mean([row["team_reward"] for row in history])
        ),
        "capture_count": capture_count,
        "capture_rate": float(capture_count / len(history)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    training_device = resolve_training_device(args.device)
    circuit_config = load_quantum_circuit_config(args.config)

    environment = create_predator_prey_environment(seed=args.seed)
    (
        n_agents,
        observation_dimensions,
        action_dimensions,
    ) = environment_specification(environment)

    learner = build_mappo_learner(
        n_agents=n_agents,
        observation_dimensions=observation_dimensions,
        action_dimensions=action_dimensions,
        circuit_config=circuit_config,
        training_device=training_device,
        args=args,
    )

    print(f"\n{PROJECT_NAME} - Multi-Agent {ALGORITHM_NAME}")
    print("========================================")
    print(f"Environment         : {ENVIRONMENT_ID}")
    print(f"Agents              : {n_agents}")
    print(f"Observation dims    : {observation_dimensions}")
    print(f"Action dims         : {action_dimensions}")
    print(f"Episodes            : {args.episodes}")
    print(f"Actor learning rate : {args.lr}")
    print(f"Critic learning rate: {args.critic_lr}")
    print(f"Gamma               : {args.gamma}")
    print(f"GAE lambda           : {args.gae_lambda}")
    print(f"PPO clip             : {args.clip_epsilon}")
    print(f"PPO epochs           : {args.ppo_epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Rollout episodes     : {args.rollout_episodes}")
    print(
        f"Entropy coefficient  : {args.entropy_coef} -> "
        f"{args.entropy_coef_final}"
    )
    print(f"Actor hidden dim     : {args.actor_hidden_dim}")
    print(f"Critic hidden dim    : {args.critic_hidden_dim}")
    print(f"Requested device     : {args.device}")
    print(f"Resolved device      : {training_device}")

    print("\nQuantum circuit")
    print("---------------")
    print(learner.actor.vqc.summary())

    print("\nMAPPO networks")
    print("--------------")
    print(
        f"Shared VQC actor parameters : {learner.actor_parameter_count()}"
    )
    print(
        f"Centralized critic parameters: {learner.critic_parameter_count()}"
    )
    print(
        "Execution policy            : decentralized local observation + "
        "agent ID"
    )
    print(
        "Training critic input       : concatenated observations from both "
        "predators"
    )

    history, sample_observations = train_multi_agent_policy(
        environment,
        learner,
        args,
    )
    environment.close()

    evaluation_history = evaluate_multi_agent_policy(
        learner,
        args,
    )
    evaluation_summary = summarize_evaluation(
        evaluation_history
    )

    print("\nSaving experiment data")
    print("----------------------")
    save_training_history(
        history,
        TRAINING_DATA_PATH,
    )
    save_evaluation_history(
        evaluation_history,
        EVALUATION_DATA_PATH,
    )
    save_mappo_checkpoint(
        learner=learner,
        circuit_config=circuit_config,
        observation_dimensions=observation_dimensions,
        action_dimensions=action_dimensions,
        args=args,
        output_path=MODEL_OUTPUT_PATH,
    )
    save_quantum_circuit_visualizations(
        learner,
        sample_observations,
        CIRCUIT_OUTPUT_DIRECTORY,
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
        f"Captures         : {evaluation_summary['capture_count']}/"
        f"{args.eval_episodes}"
    )
    print(
        f"Capture rate     : "
        f"{100.0 * evaluation_summary['capture_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()
