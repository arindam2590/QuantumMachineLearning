"""Hybrid MNIST + VQC neural-network model.

This module contains only the neural-network architecture. Training,
evaluation, command-line arguments, data loading, and model saving remain
in the driver program.

Architecture
------------
MNIST image (1 x 28 x 28)
    -> Flatten
    -> Linear(784, 128) + ReLU
    -> Linear(128, 32) + ReLU
    -> Linear(32, n_qubits) + Tanh
    -> scale to quantum angles [-pi, pi]
    -> VQC
    -> Linear(n_qubits, 32) + ReLU
    -> Linear(32, 10)
"""

import torch
from torch import nn


class HybridMNISTVQC(nn.Module):
    """Hybrid classical ANN + Variational Quantum Circuit classifier.

    Parameters
    ----------
    vqc:
        An already constructed VQC object imported from VQC_framework.
        The number of qubits in that object determines the number of
        classical features supplied to the quantum circuit.
    """

    def __init__(self, vqc):
        super().__init__()

        self.vqc = vqc
        n_qubits = vqc.n_qubits

        # Classical image feature extractor:
        # 784 image pixels -> n_qubits quantum input features.
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh(),
        )

        # Classical classification head:
        # VQC expectation values -> ten MNIST digit logits.
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, images):
        """Run MNIST images through ANN -> VQC -> ANN classifier."""
        # Compress the 28x28 image to one value per qubit.
        features = self.image_encoder(images)

        # Tanh bounds features to [-1, 1].
        # Scale them to rotation angles in [-pi, pi].
        quantum_inputs = torch.pi * features

        # VQC returns one expectation value per qubit.
        quantum_features = self.vqc(quantum_inputs)

        # Map quantum features to ten digit logits.
        logits = self.classifier(
            quantum_features.to(dtype=torch.float32)
        )

        return logits
