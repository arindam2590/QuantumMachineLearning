"""Shared utilities for the multi-agent VQC-MAPPO PredatorPrey project.

The organization intentionally follows the single-agent FrozenLake project:
command-line parsing, runtime warning configuration, device resolution, YAML
loading, and deterministic seeding live in ``util.py`` rather than in the
executable driver.
"""

from __future__ import annotations

import argparse
import random
import warnings

import numpy as np
import torch
import yaml


DEFAULT_CONFIG = "config_quantum_circuit.yaml"


def configure_runtime_warnings() -> None:
    """Suppress only the two known project-specific compatibility warnings."""
    warnings.filterwarnings(
        "ignore",
        message=(
            r"PennyLane v0\.44 has dropped .*support for NumPy < 2\.0\.0.*"
        ),
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"This figure includes Axes that are not compatible with"
            r"\s*tight_layout.*"
        ),
        category=UserWarning,
    )


def parse_args() -> argparse.Namespace:
    """Parse VQC-MAPPO training and runtime arguments.

    Plotting and output paths are intentionally not exposed as CLI options.
    Training/evaluation data, plots, circuit figures, and the checkpoint are
    saved automatically by the driver after a run completes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parameter-shared VQC-MAPPO with a centralized critic on "
            "ma-gym PredatorPrey5x5-v0"
        )
    )

    # Experiment length and optimization.
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate of the shared VQC actor.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-3,
        help="Learning rate of the centralized critic.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    # PPO/MAPPO update parameters.
    parser.add_argument("--clip-epsilon", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-loss-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)

    # Environment/evaluation.
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-evaluation",
        action="store_true",
        help="Render deterministic evaluation episodes using ma-gym.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero")
    if args.critic_lr <= 0:
        parser.error("--critic-lr must be greater than zero")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae-lambda must be in [0, 1]")
    if not 0.0 < args.clip_epsilon < 1.0:
        parser.error("--clip-epsilon must be in (0, 1)")
    if args.ppo_epochs <= 0:
        parser.error("--ppo-epochs must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.entropy_coef < 0:
        parser.error("--entropy-coef must be non-negative")
    if args.value_loss_coef < 0:
        parser.error("--value-loss-coef must be non-negative")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be greater than zero")
    if args.actor_hidden_dim <= 0:
        parser.error("--actor-hidden-dim must be greater than zero")
    if args.critic_hidden_dim <= 0:
        parser.error("--critic-hidden-dim must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be greater than zero")

    return args


def resolve_training_device(requested: str) -> str:
    """Choose a stable training device for the PennyLane default.qubit path."""
    requested = requested.lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if requested == "mps":
        print(
            "Warning: Apple MPS requested. PennyLane default.qubit is CPU "
            "based, so this VQC-MAPPO implementation uses CPU for stable "
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
                "Info: MPS is available, but CPU is selected because the "
                "VQC uses PennyLane default.qubit."
            )
        return "cpu"

    raise ValueError("device must be auto, cpu, cuda, or mps")


def load_quantum_circuit_config(path: str) -> dict:
    """Load only the ``quantum_circuit`` section from the YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "quantum_circuit" not in config:
        raise KeyError(
            "The configuration file must contain a 'quantum_circuit:' section."
        )

    return config["quantum_circuit"]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
