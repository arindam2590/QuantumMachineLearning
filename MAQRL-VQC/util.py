"""Shared utilities for the Quantum ML Project.

This module contains only reusable runtime helpers used by the multi-agent
QDRL driver. Command-line parsing remains in the executable driver so the
utility module is independent of any specific experiment interface.
"""

import random

import numpy as np
import torch
import yaml


DEFAULT_CONFIG = "config_quantum_circuit.yaml"


def resolve_training_device(requested: str) -> str:
    """Choose a stable training device for the default.qubit workflow."""
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
            "is CPU based, so this QDRL implementation uses CPU for "
            "stable forward/backward execution."
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


def load_quantum_circuit_config(path: str) -> dict:
    """Load only the ``quantum_circuit`` section from the YAML file."""
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if (
        not isinstance(config, dict)
        or "quantum_circuit" not in config
    ):
        raise KeyError(
            "The configuration file must contain "
            "'quantum_circuit:'."
        )

    return config["quantum_circuit"]


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
