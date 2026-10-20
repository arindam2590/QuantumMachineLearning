"""Quantum ML Project - user/driver code.

This is the ONLY file that contains main().

Current stage
-------------
1. Read primary runtime/training arguments from the command line.
2. Read only quantum-circuit architecture from config_quantum_circuit.yaml.
3. Import the reusable VQC package.
4. Build and execute one simple variational quantum circuit.
5. Print and visualize the generated circuit.

Later, this same user code can be extended gradually for:
    - single-agent Quantum Machine Learning (QML), then
    - single-agent Quantum Reinforcement Learning (QRL).

The VQC_framework package remains independent of the application/environment.
"""

import argparse
from pathlib import Path
import torch
import yaml

from VQC_framework import VQC


PROJECT_NAME = "Quantum ML Project"
DEFAULT_CONFIG = "config_quantum_circuit.yaml"


def parse_args():
    """Read primary experiment/runtime arguments from the command line."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} - Variational Quantum Circuit Driver"
    )

    # Primary user/experiment arguments.
    # Episodes and learning rate are included now so the same driver can be
    # extended later for QML/QRL without redesigning the interface.
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch compute device.",
    )

    # Utility arguments for this demonstration driver.
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the quantum-circuit YAML configuration.",
    )
    parser.add_argument(
        "--circuit-output",
        type=str,
        default="vqc_circuit.png",
        help="Path used to save the circuit visualization (PNG/PDF/SVG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the circuit figure without opening a Matplotlib window.",
    )

    return parser.parse_args()


def load_quantum_circuit_config(path):
    """Load only the ``quantum_circuit`` section from the YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "quantum_circuit" not in config:
        raise KeyError(
            "The configuration file must contain a top-level "
            "'quantum_circuit:' section."
        )

    return config["quantum_circuit"]


def main():
    """Run the current simple-VQC stage of the Quantum ML Project."""
    args = parse_args()
    circuit_config = load_quantum_circuit_config(args.config)

    print(f"\n{PROJECT_NAME}")
    print("=" * len(PROJECT_NAME))

    print("\nPrimary CLI arguments")
    print("---------------------")
    print(f"Episodes      : {args.episodes}")
    print(f"Learning rate : {args.lr}")
    print(f"Device        : {args.device}")
    print(
        "Note: episodes and learning rate are reserved for the upcoming "
        "QML/QRL training stages."
    )

    # The VQC package receives only the circuit configuration plus the
    # runtime device selected by the user.
    vqc = VQC(circuit_config, device=args.device)

    print("\n" + vqc.summary())

    # Demonstration input belongs to user/application code, NOT the quantum
    # circuit configuration. One value is encoded on each qubit.
    sample_input = torch.linspace(
        0.1,
        0.1 * vqc.n_qubits,
        steps=vqc.n_qubits,
        dtype=torch.float32,
        device=vqc.device,
    )

    output = vqc(sample_input)

    print("\nSample input")
    print("------------")
    print(sample_input)

    print("\nCircuit output")
    print("--------------")
    print(output)

    print("\nText circuit")
    print("------------")
    print(vqc.draw(sample_input))

    output_path = Path(args.circuit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vqc.visualize(
        sample_input,
        save_path=str(output_path),
        show=not args.no_show,
    )

    print(f"\nCircuit visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
