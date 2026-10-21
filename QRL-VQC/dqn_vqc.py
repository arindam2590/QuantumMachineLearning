"""DQN components for FrozenLake using the reusable VQC framework.

This module contains:
- ReplayBuffer
- VQCDQN neural network
- epsilon-greedy action selection
- one DQN optimization step

The environment loop, CLI, saving, evaluation, and rendering stay in the driver.
"""

from collections import deque
import random

import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer:
    """Fixed-size replay memory for DQN transitions."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size, device):
        """Sample a random minibatch and return PyTorch tensors."""
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            states,
            dtype=torch.long,
            device=device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.long,
            device=device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class VQCDQN(nn.Module):
    """DQN whose nonlinear function approximator contains a VQC.

    FrozenLake has 16 discrete states. A state index is converted into a
    binary bit string and then mapped to quantum rotation angles:

        bit 0 -> angle 0
        bit 1 -> angle pi

    For four qubits:
        state 11 -> binary 1011 -> [pi, 0, pi, pi]

    The VQC returns one expectation value per qubit. A small classical linear
    head maps those quantum features to one Q-value per action.
    """

    def __init__(self, vqc, n_states, n_actions):
        super().__init__()

        self.vqc = vqc
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.n_qubits = int(vqc.n_qubits)

        if self.n_states > 2 ** self.n_qubits:
            raise ValueError(
                f"{self.n_states} states require at least "
                f"{(self.n_states - 1).bit_length()} qubits, "
                f"but the VQC has only {self.n_qubits}."
            )

        # The quantum circuit produces n_qubits expectation values.
        # This trainable head converts them into Q-values for all actions.
        self.q_head = nn.Linear(
            self.n_qubits,
            self.n_actions,
        )

    def state_to_quantum_angles(self, states):
        """Convert integer state IDs to binary angle-encoded VQC inputs."""
        states = torch.as_tensor(
            states,
            dtype=torch.long,
            device=self.q_head.weight.device,
        )

        single_state = states.ndim == 0
        if single_state:
            states = states.unsqueeze(0)

        # Most-significant bit first.
        shifts = torch.arange(
            self.n_qubits - 1,
            -1,
            -1,
            device=states.device,
            dtype=torch.long,
        )

        bits = (
            (states.unsqueeze(1) >> shifts) & 1
        ).to(torch.float32)

        angles = torch.pi * bits

        if single_state:
            return angles.squeeze(0)

        return angles

    def forward(self, states):
        """Return Q-values for one state or a batch of states."""
        states = torch.as_tensor(
            states,
            dtype=torch.long,
            device=self.q_head.weight.device,
        )

        single_state = states.ndim == 0

        quantum_inputs = self.state_to_quantum_angles(states)

        quantum_features = self.vqc(
            quantum_inputs
        ).to(dtype=torch.float32)

        q_values = self.q_head(
            quantum_features
        )

        if single_state and q_values.ndim == 2:
            q_values = q_values.squeeze(0)

        return q_values


def select_action(
    policy_net,
    state,
    epsilon,
    n_actions,
    rng,
):
    """Epsilon-greedy DQN action selection."""
    if rng.random() < epsilon:
        return rng.randrange(n_actions)

    with torch.no_grad():
        q_values = policy_net(
            torch.tensor(
                state,
                dtype=torch.long,
                device=policy_net.q_head.weight.device,
            )
        )

    return int(
        q_values.argmax().item()
    )


def optimize_dqn(
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    batch_size,
    gamma,
    device,
):
    """Perform one DQN minibatch update.

    Returns
    -------
    float or None
        Scalar training loss. None is returned when replay memory does not
        yet contain enough samples.
    """
    if len(replay_buffer) < batch_size:
        return None

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(
        batch_size,
        device,
    )

    # Q(s,a) from the online network.
    current_q = policy_net(states).gather(
        1,
        actions,
    ).squeeze(1)

    # Bellman target using the frozen target network.
    with torch.no_grad():
        next_q = target_net(
            next_states
        ).max(dim=1).values

        target_q = rewards + (
            gamma
            * (1.0 - dones)
            * next_q
        )

    loss = F.smooth_l1_loss(
        current_q,
        target_q,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # Small gradient clipping is helpful for stable DQN updates.
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return float(loss.item())
