"""Shared utility functions for FrozenLake VQC-DQN drivers."""

import argparse
import random

import numpy as np
import torch
import yaml


DEFAULT_CONFIG = "config_quantum_circuit.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VQC-DQN on Gymnasium FrozenLake"
    )

    # Main DQN parameters.
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--replay-size", type=int, default=1000)
    parser.add_argument("--min-replay-size", type=int, default=32)
    parser.add_argument("--target-update", type=int, default=20)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=150.0)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    # Deterministic FrozenLake is easier for this first QRL example.
    # Add --slippery to use the stochastic environment.
    parser.add_argument(
        "--slippery",
        action="store_true",
        help="Use stochastic/slippery FrozenLake.",
    )

    # Rendering is used during final greedy evaluation, not during training.
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes in a window.",
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
        "--results-output",
        type=str,
        default="results/frozenlake_vqc_dqn.csv",
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default="results/frozenlake_vqc_dqn.pt",
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/plots",
        help="Directory used to save training-result figures.",
    )

    parser.add_argument(
        "--plot-window",
        type=int,
        default=100,
        help="Moving-average window for reward/success plots.",
    )

    parser.add_argument(
        "--circuit-output",
        type=str,
        default="results/frozenlake_vqc_circuit.png",
        help="Path used to save the VQC circuit visualization.",
    )

    return parser.parse_args()


def resolve_training_device(requested):
    """Choose a stable device for default.qubit based hybrid training."""
    requested = requested.lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"

        print(
            "Warning: CUDA requested but unavailable. "
            "Falling back to CPU."
        )
        return "cpu"

    if requested == "mps":
        print(
            "Warning: Apple MPS requested. PennyLane default.qubit "
            "is CPU based, so this QRL example uses CPU for stable "
            "forward/backward execution."
        )
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            print(
                "Info: MPS is available, but CPU is selected because "
                "the VQC uses PennyLane default.qubit."
            )

        return "cpu"

    raise ValueError(
        "device must be auto, cpu, cuda, or mps"
    )


def load_quantum_circuit_config(path):
    """Load only the quantum-circuit section from YAML."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if (not isinstance(config, dict) or "quantum_circuit" not in config):
        raise KeyError(
            "The configuration file must contain "
            "'quantum_circuit:'."
        )

    return config["quantum_circuit"]


def set_seed(seed):
    """Set basic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
