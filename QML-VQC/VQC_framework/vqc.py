"""Reusable Variational Quantum Circuit class used by Quantum ML Project."""

from __future__ import annotations

from unittest import result
import warnings
from typing import Dict, Optional, Sequence, Union

import pennylane as qml
import torch
from torch import nn

from .circuit import (
    apply_entanglement,
    apply_variational_layer,
    encode_inputs,
    make_observable,
    validate_circuit_config,
)
from .visualization import draw_circuit, visualize_circuit


class VQC(nn.Module):
    """Reusable Variational Quantum Circuit.

    Parameters
    ----------
    config:
        Dictionary containing ONLY quantum-circuit-related parameters:

        ``n_qubits``
            Number of qubits.

        ``n_layers``
            Number of repeated variational layers.

        ``encoding_gate``
            Classical-data encoding gate: ``RX``, ``RY``, or ``RZ``.

        ``trainable_rotations``
            Sequence of trainable single-qubit gates chosen from
            ``RX``, ``RY``, and ``RZ``.

        ``entanglement``
            Entanglement topology: ``chain``, ``ring``, or ``none``.

        ``entangling_gate``
            Two-qubit entangling gate: ``CNOT`` or ``CZ``.

        ``measurement_pauli``
            Observable measured on every qubit: ``X``, ``Y``, or ``Z``.

    device:
        PyTorch-side compute device: ``auto``, ``cpu``, ``cuda``, or ``mps``.

    Notes
    -----
    The quantum backend is intentionally fixed to PennyLane ``default.qubit``.
    This keeps the framework minimal and avoids exposing simulator/backend
    complexity to the user at this stage.
    """

    _DIFF_METHOD = "backprop"
    _WEIGHT_INIT_SCALE = 0.1
    _RANDOM_SEED = 42

    def __init__(self, config: Dict, device: str = "auto") -> None:
        super().__init__()

        self.config = dict(config)

        self.n_qubits = int(self.config.get("n_qubits", 4))
        self.n_layers = int(self.config.get("n_layers", 2))
        self.encoding_gate = str(self.config.get("encoding_gate", "RY")).upper()
        self.trainable_rotations = tuple(
            str(gate).upper()
            for gate in self.config.get(
                "trainable_rotations", ["RX", "RY", "RZ"]
            )
        )
        self.entanglement = str(
            self.config.get("entanglement", "ring")
        ).lower()
        self.entangling_gate = str(
            self.config.get("entangling_gate", "CNOT")
        ).upper()
        self.measurement_pauli = str(
            self.config.get("measurement_pauli", "Z")
        ).upper()

        validate_circuit_config(
            self.n_qubits,
            self.n_layers,
            self.encoding_gate,
            self.trainable_rotations,
            self.entanglement,
            self.entangling_gate,
            self.measurement_pauli,
        )
        self.device = self._resolve_device(device)

        torch.manual_seed(self._RANDOM_SEED)

        weight_shape = (
            self.n_layers,
            self.n_qubits,
            len(self.trainable_rotations),
        )
        initial_weights = self._WEIGHT_INIT_SCALE * torch.randn(
            weight_shape,
            dtype=torch.float32,
        )
        self.weights = nn.Parameter(initial_weights.to(self.device))

        # The only simulator used by this minimal framework.
        self.quantum_device = qml.device(
            "default.qubit",
            wires=self.n_qubits,
        )

        @qml.qnode(
            self.quantum_device,
            interface="torch",
            diff_method=self._DIFF_METHOD,
        )
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # 1. Classical-to-quantum state encoding.
            encode_inputs(inputs, self.n_qubits, self.encoding_gate)

            # 2. Repeated trainable variational layers.
            for layer in range(self.n_layers):
                apply_variational_layer(
                    weights[layer],
                    self.n_qubits,
                    self.trainable_rotations,
                )
                apply_entanglement(
                    self.n_qubits,
                    self.entanglement,
                    self.entangling_gate,
                )

            # 3. Quantum measurement.
            return [
                qml.expval(make_observable(self.measurement_pauli, wire))
                for wire in range(self.n_qubits)
            ]

        self.qnode = circuit

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        """Resolve user-selected CPU/CUDA/Apple-MPS device."""
        requested = str(requested).lower()

        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")

        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")

        if requested == "mps":
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            warnings.warn("MPS requested but unavailable; falling back to CPU.")
            return torch.device("cpu")

        if requested == "cpu":
            return torch.device("cpu")

        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    def forward(
        self,
        inputs: Union[torch.Tensor, Sequence[float]],
    ) -> torch.Tensor:
        """Execute the circuit for one observation or a batch.

        Accepted shapes
        ---------------
        ``[n_qubits]``
            One input observation.

        ``[batch_size, n_qubits]``
            Batch of observations.
        """
        x = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=self.device,
        )

        if x.ndim == 1:
            return self._forward_single(x)
        if x.ndim == 2:
            return torch.stack([self._forward_single(row) for row in x])

        raise ValueError(
            "inputs must have shape [n_qubits] or [batch, n_qubits]"
        )

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:

        if x.numel() != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} input values, "
                f"but received {x.numel()}."
            )

        # default.qubit executes on CPU
        x_cpu = x.reshape(self.n_qubits).to(
            device="cpu",
            dtype=torch.float32
        )

        weights_cpu = self.weights.to(
            device="cpu",
            dtype=torch.float32
        )

        result = self.qnode(
            x_cpu,
            weights_cpu
        )

        if isinstance(result, (tuple, list)):
            result = torch.stack(list(result))

        # Important for Apple MPS
        return result.to(
            device=self.device,
            dtype=self.weights.dtype
        )

    def parameter_count(self) -> int:
        """Return the number of trainable quantum-circuit parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def summary(self) -> str:
        """Return a concise description of the generated VQC."""
        return (
            "VQC Framework Summary\n"
            "---------------------\n"
                f"Qubits               : {self.n_qubits}\n"
                f"Layers               : {self.n_layers}\n"
                f"Encoding gate        : {self.encoding_gate}\n"
                f"Trainable rotations  : {self.trainable_rotations}\n"
                f"Entanglement         : {self.entanglement}\n"
                f"Entangling gate      : {self.entangling_gate}\n"
                f"Measurement          : Pauli-{self.measurement_pauli}\n"
                "Quantum simulator    : PennyLane default.qubit\n"
                f"PyTorch device       : {self.device}\n"
                f"Trainable parameters : {self.parameter_count()}"
            )

    def draw(
        self,
        inputs: Optional[Sequence[float]] = None,
    ) -> str:
        """Return a text drawing of the actual generated quantum circuit."""
        return draw_circuit(
            self.qnode,
            self.weights,
            self.n_qubits,
            inputs,
        )

    def visualize(
        self,
        inputs: Optional[Sequence[float]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Create a Matplotlib visualization of the actual generated VQC."""
        return visualize_circuit(
            self.qnode,
            self.weights,
            self.n_qubits,
            inputs,
            save_path=save_path,
            show=show,
        )
