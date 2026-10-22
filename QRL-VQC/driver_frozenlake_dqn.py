"""Quantum ML Project - VQC-DQN on Gymnasium FrozenLake.

This is the user/driver program.

Files used
----------
VQC_framework/
    Reusable Variational Quantum Circuit implementation.

dqn_vqc.py
    VQC-based DQN network, replay memory, epsilon-greedy policy, and
    optimization step.

config_quantum_circuit.yaml
    Contains only the quantum-circuit architecture.

The driver contains:
- command-line arguments,
- FrozenLake environment creation,
- training loop,
- evaluation,
- Results/model saving.

FrozenLake state encoding
-------------------------
FrozenLake-v1 has 16 states. With four qubits, the integer state is encoded
as four binary values and converted to rotation angles 0 or pi.

Example:
    state 11 -> 1011 -> [pi, 0, pi, pi]
"""

import csv
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from VQC_framework import VQC
from dqn_vqc import ReplayBuffer, VQCDQN, optimize_dqn, select_action

from plot_frozenlake_results import plot_results
from util import parse_args, resolve_training_device, load_quantum_circuit_config, set_seed


def epsilon_by_episode(args, episode):
    """Exponential epsilon decay."""
    return (
        args.epsilon_end
        + (
            args.epsilon_start
            - args.epsilon_end
        )
        * math.exp(
            -episode / args.epsilon_decay
        )
    )


def create_environment(slippery, render=False):
    """Create FrozenLake-v1."""
    render_mode = "human" if render else None

    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode=render_mode)


def save_training_history(history, output_path):
    """Write episode-level training statistics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "reward", "steps", "loss", "epsilon", "success"],
        )

        writer.writeheader()
        writer.writerows(history)

    print(
        f"\nTraining history saved to: "
        f"{output_path}"
    )


def save_model(policy_net, args, circuit_config, output_path):
    """Save the trained policy-network checkpoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "quantum_circuit_config": circuit_config,
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "slippery": args.slippery,
        },
        output_path,
    )

    print(
        f"Model saved to: "
        f"{output_path}"
    )


def evaluate_policy(policy_net, args, seed):
    """Run greedy evaluation; optionally render it."""
    env = create_environment(slippery=args.slippery, render=args.render)

    successes = 0
    rewards = []

    print("\nEvaluation")
    print("----------")

    for episode in range(args.eval_episodes):
        state, _ = env.reset(seed=seed + 10000 + episode)

        episode_reward = 0.0

        for _ in range(args.max_steps):
            with torch.no_grad():
                q_values = policy_net(
                    torch.tensor(
                        state,
                        dtype=torch.long,
                        device=policy_net.q_head.weight.device,
                    )
                )

                action = int(q_values.argmax().item())

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(
            episode_reward
        )

        successes += int(
            episode_reward > 0.0
        )

        print(
            f"Eval episode "
            f"{episode + 1:02d}/"
            f"{args.eval_episodes}: "
            f"reward={episode_reward:.1f}"
        )

    env.close()

    success_rate = (100.0 * successes / args.eval_episodes)

    mean_reward = float(np.mean(rewards))

    print(
        f"\nMean evaluation reward : "
        f"{mean_reward:.3f}"
    )

    print(
        f"Success rate           : "
        f"{success_rate:.2f}%"
    )

    return mean_reward, success_rate


def save_circuit_figure(policy_net, output_path, example_state=11):
    """Save the actual VQC circuit used by the DQN."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = torch.tensor(example_state, dtype=torch.long, device=policy_net.q_head.weight.device)

    with torch.no_grad():
        quantum_input = policy_net.state_to_quantum_angles(state)

    policy_net.vqc.visualize(quantum_input, save_path=str(output_path), show=False)

    print(f"VQC circuit figure saved to: {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    training_device = resolve_training_device(args.device)

    circuit_config = (load_quantum_circuit_config(args.config))

    # Training environment never renders.
    env = create_environment(slippery=args.slippery, render=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print("\nQuantum ML Project - FrozenLake VQC-DQN")
    print("=======================================")
    print(f"Episodes          : {args.episodes}")
    print(f"Learning rate     : {args.lr}")
    print(f"Gamma             : {args.gamma}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Replay capacity   : {args.replay_size}")
    print(f"Target update     : {args.target_update} episodes")
    print(f"Requested device  : {args.device}")
    print(f"Resolved device   : {training_device}")
    print(f"Slippery          : {args.slippery}")
    print(f"Render evaluation : {args.render}")
    print(f"States            : {n_states}")
    print(f"Actions           : {n_actions}")

    # Online/policy VQC.
    policy_vqc = VQC(circuit_config, device=training_device)

    # Target network must have an independent VQC instance.
    target_vqc = VQC(circuit_config, device=training_device)

    device = policy_vqc.device

    policy_net = VQCDQN(policy_vqc, n_states=n_states, n_actions=n_actions).to(device)

    target_net = VQCDQN(target_vqc, n_states=n_states, n_actions=n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    print("\n" + policy_vqc.summary())

    print("\nDQN")
    print("---")
    print(policy_net)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(args.replay_size)

    rng = random.Random(args.seed)

    history = []

    print("\nTraining")
    print("--------")

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)

        epsilon = epsilon_by_episode(args, episode - 1)

        total_reward = 0.0
        episode_losses = []
        steps_taken = 0

        for step in range(1, args.max_steps + 1):
            action = select_action(policy_net, state, epsilon, n_actions, rng)

            next_state, reward, terminated, truncated, _= env.step(action)

            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps_taken = step

            # Delay learning until the replay buffer has enough experience.
            if (len(replay_buffer) >= args.min_replay_size):
                loss = optimize_dqn(policy_net, target_net, replay_buffer, optimizer, args.batch_size, args.gamma, device)

                if loss is not None:
                    episode_losses.append(loss)

            if done:
                break

        # Periodically synchronize the target network.
        if (episode % args.target_update == 0):
            target_net.load_state_dict(policy_net.state_dict())

        mean_loss = (float(np.mean(episode_losses)) if episode_losses else 0.0)

        success = int(total_reward > 0.0)

        history.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps_taken,
                "loss": mean_loss,
                "epsilon": epsilon,
                "success": success,
            }
        )

        if (episode == 1 or episode % 25 == 0 or episode == args.episodes):
            recent = history[-100:]

            recent_success = (100.0 * sum(item["success"] for item in recent)/ len(recent))

            recent_reward = float(np.mean([item["reward"] for item in recent]))

            print(
                f"Episode "
                f"{episode:04d}/"
                f"{args.episodes} | "
                f"reward={total_reward:.1f} | "
                f"avg100={recent_reward:.3f} | "
                f"success100={recent_success:6.2f}% | "
                f"loss={mean_loss:.5f} | "
                f"eps={epsilon:.3f}"
            )

    env.close()

    save_training_history(history, args.results_output)

    save_model(policy_net, args, circuit_config, args.model_output)

    print("\nGenerating training plots")
    print("-------------------------")
    plot_results(input_csv=args.results_output, output_dir=args.plots_dir, window=args.plot_window)

    print("\nGenerating VQC circuit figure")
    print("-----------------------------")
    save_circuit_figure(policy_net, args.circuit_output, example_state=11)

    evaluate_policy(policy_net, args, args.seed)


if __name__ == "__main__":
    main()
