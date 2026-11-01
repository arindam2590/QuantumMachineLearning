"""Multi-agent VQC-DQN components for cooperative discrete-action tasks.

This module extends the single-agent VQC-DQN implementation to an
independent multi-agent setting while keeping ``VQC_framework`` unchanged.

Each controlled agent owns:
- an online VQC-DQN network,
- a target VQC-DQN network,
- an experience replay buffer,
- an Adam optimizer.

The environment loop remains in the driver. This module only contains the
learning components required by each independent agent.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ExperienceReplayBuffer:
    """Fixed-capacity replay memory for one independent learning agent."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")

        self.capacity = int(capacity)
        self._memory = deque(maxlen=self.capacity)

    def add(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        self._memory.append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a minibatch and convert it to PyTorch tensors."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if len(self._memory) < batch_size:
            raise ValueError(
                "Replay memory does not contain enough transitions for "
                f"batch_size={batch_size}."
            )

        batch = random.sample(self._memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)

        observation_tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        action_tensor = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        reward_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )
        next_observation_tensor = torch.as_tensor(
            np.asarray(next_observations, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        done_tensor = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        )

        return (
            observation_tensor,
            action_tensor,
            reward_tensor,
            next_observation_tensor,
            done_tensor,
        )

    def __len__(self) -> int:
        return len(self._memory)


class VQCDQNNetwork(nn.Module):
    """Hybrid Q-network that uses the reusable VQC as its nonlinear core.

    ``PredatorPrey5x5-v0`` supplies a local observation vector for every
    predator. The observation dimension is larger than the number of qubits
    used by the current VQC configuration, so this network learns a compact
    classical projection before the quantum circuit:

        local observation
            -> linear projection
            -> tanh
            -> angles in [-pi, pi]
            -> VQC
            -> quantum expectation values
            -> linear Q-head
            -> one Q-value per action

    The VQC object is injected from ``VQC_framework`` and is not modified.
    """

    def __init__(
        self,
        vqc: nn.Module,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be greater than zero")
        if action_dim <= 0:
            raise ValueError("action_dim must be greater than zero")

        self.vqc = vqc
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.n_qubits = int(vqc.n_qubits)

        self.observation_encoder = nn.Linear(
            self.observation_dim,
            self.n_qubits,
        )

        self.q_head = nn.Linear(
            self.n_qubits,
            self.action_dim,
        )

    @property
    def device(self) -> torch.device:
        """Return the device used by the classical layers."""
        return self.q_head.weight.device

    def encode_observation(
        self,
        observations: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        """Map local observations to one rotation angle per VQC qubit."""
        x = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        single_observation = x.ndim == 1
        if single_observation:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(
                "observations must have shape [observation_dim] or "
                "[batch_size, observation_dim]"
            )

        if x.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {x.shape[-1]}."
            )

        encoded = torch.tanh(
            self.observation_encoder(x)
        ) * torch.pi

        if single_observation:
            return encoded.squeeze(0)

        return encoded

    def forward(
        self,
        observations: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        """Return Q-values for one local observation or a batch."""
        x = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        single_observation = x.ndim == 1

        quantum_inputs = self.encode_observation(x)
        quantum_features = self.vqc(
            quantum_inputs
        ).to(dtype=torch.float32)

        q_values = self.q_head(
            quantum_features
        )

        if single_observation and q_values.ndim == 2:
            q_values = q_values.squeeze(0)

        return q_values


class IndependentVQCDQNAgent:
    """One independent VQC-DQN learner in a cooperative multi-agent task.

    Each instance owns an online network, target network, replay buffer, and
    optimizer. Cooperation is induced by the environment reward structure;
    parameters are not shared between different agents in this first
    multi-agent extension.
    """

    def __init__(
        self,
        agent_id: int,
        online_vqc: nn.Module,
        target_vqc: nn.Module,
        observation_dim: int,
        action_dim: int,
        learning_rate: float,
        discount_factor: float,
        replay_capacity: int,
        batch_size: int,
        minimum_replay_size: int,
        gradient_clip_norm: float = 10.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if minimum_replay_size < batch_size:
            raise ValueError(
                "minimum_replay_size must be greater than or equal to "
                "batch_size"
            )
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")

        self.agent_id = int(agent_id)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.discount_factor = float(discount_factor)
        self.batch_size = int(batch_size)
        self.minimum_replay_size = int(minimum_replay_size)
        self.gradient_clip_norm = float(gradient_clip_norm)

        self.online_network = VQCDQNNetwork(
            online_vqc,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        ).to(online_vqc.device)

        self.target_network = VQCDQNNetwork(
            target_vqc,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        ).to(target_vqc.device)

        self.device = self.online_network.device

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=float(learning_rate),
        )

        self.replay_buffer = ExperienceReplayBuffer(
            replay_capacity
        )

    def select_action(
        self,
        observation: Sequence[float],
        epsilon: float,
        rng: random.Random,
    ) -> int:
        """Select one action using an epsilon-greedy policy."""
        if rng.random() < epsilon:
            return rng.randrange(self.action_dim)

        with torch.no_grad():
            q_values = self.online_network(
                observation
            )

        return int(
            q_values.argmax(dim=-1).item()
        )

    def store_transition(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        done: bool,
    ) -> None:
        """Add one agent-local transition to replay memory."""
        self.replay_buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

    def can_optimize(self) -> bool:
        """Return whether replay memory is ready for a learning step."""
        return (
            len(self.replay_buffer)
            >= self.minimum_replay_size
            and len(self.replay_buffer)
            >= self.batch_size
        )

    def optimize_model(self) -> float | None:
        """Perform one independent DQN minibatch update."""
        if not self.can_optimize():
            return None

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size,
            self.device,
        )

        current_q = self.online_network(
            observations
        ).gather(
            1,
            actions,
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(
                next_observations
            ).max(dim=1).values

            target_q = rewards + (
                self.discount_factor
                * (1.0 - dones)
                * next_q
            )

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.gradient_clip_norm,
        )

        self.optimizer.step()

        return float(loss.item())

    def synchronize_target_network(self) -> None:
        """Hard-copy online-network parameters into the target network."""
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters in the online network."""
        return sum(
            parameter.numel()
            for parameter in self.online_network.parameters()
            if parameter.requires_grad
        )

    def checkpoint_state(self) -> dict:
        """Return serializable state needed to resume this learner."""
        return {
            "agent_id": self.agent_id,
            "online_network_state_dict": self.online_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
