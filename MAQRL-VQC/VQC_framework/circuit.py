"""Quantum-circuit construction helpers for :mod:`VQC_framework`.

This module contains only the reusable circuit-building operations used by
:class:`VQC`.  Application logic remains outside the package in ``driver.py``.
"""

from __future__ import annotations

from typing import Sequence

import pennylane as qml
import torch


def validate_circuit_config(
    n_qubits: int,
    n_layers: int,
    encoding_gate: str,
    trainable_rotations: Sequence[str],
    entanglement: str,
    entangling_gate: str,
    measurement_pauli: str,
) -> None:
    """Validate the quantum-circuit configuration values."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    valid_rotations = {"RX", "RY", "RZ"}
    if encoding_gate not in valid_rotations:
        raise ValueError("encoding_gate must be RX, RY, or RZ")

    if not trainable_rotations:
        raise ValueError("trainable_rotations cannot be empty")
    if any(gate not in valid_rotations for gate in trainable_rotations):
        raise ValueError("trainable_rotations may contain only RX, RY, RZ")

    if entanglement not in {"chain", "ring", "none"}:
        raise ValueError("entanglement must be chain, ring, or none")

    if entangling_gate not in {"CNOT", "CZ"}:
        raise ValueError("entangling_gate must be CNOT or CZ")

    if measurement_pauli not in {"X", "Y", "Z"}:
        raise ValueError("measurement_pauli must be X, Y, or Z")


def apply_rotation(name: str, angle: torch.Tensor, wire: int) -> None:
    """Apply one parameterized single-qubit rotation gate."""
    if name == "RX":
        qml.RX(angle, wires=wire)
    elif name == "RY":
        qml.RY(angle, wires=wire)
    elif name == "RZ":
        qml.RZ(angle, wires=wire)
    else:
        raise ValueError(f"Unsupported rotation gate: {name}")


def encode_inputs(
    inputs: torch.Tensor,
    n_qubits: int,
    encoding_gate: str,
) -> None:
    """Encode one classical feature onto each qubit."""
    for wire in range(n_qubits):
        apply_rotation(encoding_gate, inputs[wire], wire)


def apply_variational_layer(
    layer_weights: torch.Tensor,
    n_qubits: int,
    trainable_rotations: Sequence[str],
) -> None:
    """Apply the trainable single-qubit rotations of one VQC layer."""
    for wire in range(n_qubits):
        for gate_index, gate_name in enumerate(trainable_rotations):
            apply_rotation(
                gate_name,
                layer_weights[wire, gate_index],
                wire,
            )


def apply_entanglement(
    n_qubits: int,
    entanglement: str,
    entangling_gate: str,
) -> None:
    """Entangle qubits using the configured topology and two-qubit gate."""
    if entanglement == "none" or n_qubits == 1:
        return

    def apply_gate(control: int, target: int) -> None:
        if entangling_gate == "CNOT":
            qml.CNOT(wires=[control, target])
        else:
            qml.CZ(wires=[control, target])

    for wire in range(n_qubits - 1):
        apply_gate(wire, wire + 1)

    if entanglement == "ring" and n_qubits > 2:
        apply_gate(n_qubits - 1, 0)


def make_observable(measurement_pauli: str, wire: int):
    """Return the configured single-qubit Pauli observable."""
    if measurement_pauli == "X":
        return qml.PauliX(wire)
    if measurement_pauli == "Y":
        return qml.PauliY(wire)
    return qml.PauliZ(wire)
