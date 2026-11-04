"""Circuit drawing and visualization helpers for :mod:`VQC_framework`."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pennylane as qml
import torch


def prepare_drawing_input(
    inputs: Optional[Sequence[float]],
    n_qubits: int,
) -> torch.Tensor:
    """Prepare a valid CPU tensor for circuit visualization."""
    if inputs is None:
        return torch.zeros(n_qubits, dtype=torch.float32)

    x = torch.as_tensor(inputs, dtype=torch.float32).reshape(-1)

    if x.numel() != n_qubits:
        raise ValueError(
            f"Expected {n_qubits} input values, received {x.numel()}."
        )

    return x.to("cpu")


def draw_circuit(qnode, weights, n_qubits: int, inputs=None) -> str:
    """Return a text drawing of the generated PennyLane circuit."""
    x = prepare_drawing_input(inputs, n_qubits)
    return qml.draw(qnode)(
        x,
        weights.detach().to("cpu"),
    )


def visualize_circuit(
    qnode,
    weights,
    n_qubits: int,
    inputs=None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Create and optionally save a Matplotlib visualization of the VQC."""
    x = prepare_drawing_input(inputs, n_qubits)

    fig, ax = qml.draw_mpl(qnode)(
        x,
        weights.detach().to("cpu"),
    )
    fig.suptitle("Variational Quantum Circuit")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
