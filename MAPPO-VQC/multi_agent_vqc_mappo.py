"""VQC-MAPPO components for cooperative multi-agent discrete-action tasks.

The module implements a compact feed-forward MAPPO variant for homogeneous
agents using centralized training and decentralized execution (CTDE):

* one parameter-shared VQC actor receives each predator's local observation;
* a one-hot agent identifier is appended to the local observation so the
  shared policy can learn asymmetric cooperative roles when useful;
* one classical centralized critic receives the concatenation of all local
  observations during training only;
* generalized advantage estimation (GAE) is computed independently for each
  complete trajectory before trajectories are combined into a rollout;
* PPO clipping, scheduled entropy regularization, clipped value loss, and
  gradient clipping are used during optimization.

``VQC_framework`` is consumed as-is and is not modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class VQCSharedActor(nn.Module):
    """Parameter-shared decentralized policy using the reusable VQC."""

    def __init__(
        self,
        vqc: nn.Module,
        observation_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.vqc = vqc
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.n_agents = int(n_agents)
        self.n_qubits = int(vqc.n_qubits)
        self.policy_input_dim = self.observation_dim + self.n_agents
        self.hidden_dim = int(hidden_dim)

        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be greater than zero")
        if self.action_dim <= 1:
            raise ValueError("action_dim must be greater than one")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be greater than zero")
        if self.hidden_dim < self.n_qubits:
            raise ValueError("hidden_dim must be at least the number of qubits")

        # A deeper classical feature extractor gives the shared policy enough
        # capacity to compress the 28-dimensional local observation plus
        # agent identity before the unchanged VQC interface receives four
        # bounded rotation angles.
        bottleneck_dim = max(self.n_qubits, self.hidden_dim // 2)
        self.observation_encoder = nn.Sequential(
            nn.LayerNorm(self.policy_input_dim),
            nn.Linear(self.policy_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, self.n_qubits),
        )
        self.action_head = nn.Linear(
            self.n_qubits,
            self.action_dim,
        )


    @property
    def device(self) -> torch.device:
        """Return the device hosting the classical actor parameters."""
        return self.action_head.weight.device

    def _prepare_inputs(
        self,
        observations: torch.Tensor | np.ndarray | Sequence[float],
        agent_ids: torch.Tensor | np.ndarray | Sequence[int] | int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Convert observations and agent IDs to aligned batched tensors."""
        observation_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        agent_id_tensor = torch.as_tensor(
            agent_ids,
            dtype=torch.long,
            device=self.device,
        )

        single_observation = observation_tensor.ndim == 1
        if single_observation:
            observation_tensor = observation_tensor.unsqueeze(0)

        if agent_id_tensor.ndim == 0:
            agent_id_tensor = agent_id_tensor.unsqueeze(0)

        if observation_tensor.ndim != 2:
            raise ValueError(
                "observations must have shape [observation_dim] or "
                "[batch, observation_dim]"
            )
        if observation_tensor.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observation_tensor.shape[-1]}."
            )
        if agent_id_tensor.ndim != 1:
            raise ValueError("agent_ids must be a scalar or one-dimensional")
        if agent_id_tensor.shape[0] != observation_tensor.shape[0]:
            raise ValueError(
                "The number of agent IDs must match the number of observations."
            )
        if torch.any(agent_id_tensor < 0) or torch.any(
            agent_id_tensor >= self.n_agents
        ):
            raise ValueError("agent_ids contains an invalid agent index")

        return observation_tensor, agent_id_tensor, single_observation

    def encode_observation(
        self,
        observations: torch.Tensor | np.ndarray | Sequence[float],
        agent_ids: torch.Tensor | np.ndarray | Sequence[int] | int,
    ) -> torch.Tensor:
        """Project local observations plus agent identity to VQC angles."""
        observation_tensor, agent_id_tensor, single_observation = (
            self._prepare_inputs(observations, agent_ids)
        )

        one_hot_ids = torch.nn.functional.one_hot(
            agent_id_tensor,
            num_classes=self.n_agents,
        ).to(dtype=torch.float32)

        policy_input = torch.cat(
            [observation_tensor, one_hot_ids],
            dim=-1,
        )
        angles = torch.tanh(
            self.observation_encoder(policy_input)
        ) * torch.pi

        if single_observation:
            return angles.squeeze(0)
        return angles

    def forward(
        self,
        observations: torch.Tensor | np.ndarray | Sequence[float],
        agent_ids: torch.Tensor | np.ndarray | Sequence[int] | int,
    ) -> torch.Tensor:
        """Return action logits for one or more local observations."""
        observation_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        single_observation = observation_tensor.ndim == 1

        quantum_inputs = self.encode_observation(
            observations,
            agent_ids,
        )
        quantum_features = self.vqc(
            quantum_inputs
        ).to(dtype=torch.float32)
        logits = self.action_head(quantum_features)

        if single_observation and logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class CentralizedValueCritic(nn.Module):
    """Classical centralized state-value critic used only during training."""

    def __init__(
        self,
        centralized_observation_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if centralized_observation_dim <= 0:
            raise ValueError(
                "centralized_observation_dim must be greater than zero"
            )
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")

        self.centralized_observation_dim = int(centralized_observation_dim)
        self.hidden_dim = int(hidden_dim)

        reduced_hidden_dim = max(64, self.hidden_dim // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(self.centralized_observation_dim),
            nn.Linear(self.centralized_observation_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, reduced_hidden_dim),
            nn.Tanh(),
            nn.Linear(reduced_hidden_dim, 1),
        )


    @property
    def device(self) -> torch.device:
        """Return the device hosting the critic parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        centralized_observation: torch.Tensor | np.ndarray | Sequence[float],
    ) -> torch.Tensor:
        """Estimate the centralized state value."""
        state = torch.as_tensor(
            centralized_observation,
            dtype=torch.float32,
            device=self.device,
        )
        single_state = state.ndim == 1
        if single_state:
            state = state.unsqueeze(0)

        if state.ndim != 2:
            raise ValueError(
                "centralized_observation must have shape [state_dim] or "
                "[batch, state_dim]"
            )
        if state.shape[-1] != self.centralized_observation_dim:
            raise ValueError(
                "Unexpected centralized observation dimension: "
                f"expected {self.centralized_observation_dim}, "
                f"received {state.shape[-1]}."
            )

        values = self.network(state).squeeze(-1)
        if single_state:
            values = values.squeeze(0)
        return values


class MAPPOEpisodeBuffer:
    """On-policy storage for one complete cooperative episode."""

    def __init__(self, n_agents: int) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be greater than zero")

        self.n_agents = int(n_agents)
        self.local_observations: list[np.ndarray] = []
        self.centralized_observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probabilities: list[np.ndarray] = []
        self.team_rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.final_centralized_observation: np.ndarray | None = None
        self.final_done: bool = True

    def add(
        self,
        local_observations: Sequence[Sequence[float]],
        centralized_observation: Sequence[float],
        actions: Sequence[int],
        log_probabilities: Sequence[float],
        team_reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store one joint transition from the current on-policy rollout."""
        local_array = np.asarray(local_observations, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.int64)
        log_probability_array = np.asarray(
            log_probabilities,
            dtype=np.float32,
        )

        if local_array.shape[0] != self.n_agents:
            raise ValueError(
                "local_observations does not contain one observation per agent"
            )
        if action_array.shape != (self.n_agents,):
            raise ValueError("actions must contain one action per agent")
        if log_probability_array.shape != (self.n_agents,):
            raise ValueError(
                "log_probabilities must contain one value per agent"
            )

        self.local_observations.append(local_array.copy())
        self.centralized_observations.append(
            np.asarray(
                centralized_observation,
                dtype=np.float32,
            ).copy()
        )
        self.actions.append(action_array.copy())
        self.log_probabilities.append(log_probability_array.copy())
        self.team_rewards.append(float(team_reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def finalize(
        self,
        final_centralized_observation: Sequence[float],
        final_done: bool,
    ) -> None:
        """Record the final state needed for GAE bootstrapping."""
        self.final_centralized_observation = np.asarray(
            final_centralized_observation,
            dtype=np.float32,
        ).copy()
        self.final_done = bool(final_done)

    def __len__(self) -> int:
        return len(self.team_rewards)


@dataclass(frozen=True)
class MAPPOUpdateMetrics:
    """Mean statistics from one PPO/MAPPO update."""

    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


class VQCMAPPOLearner:
    """Parameter-shared VQC-MAPPO learner with a centralized critic."""

    def __init__(
        self,
        actor: VQCSharedActor,
        critic: CentralizedValueCritic,
        actor_learning_rate: float,
        critic_learning_rate: float,
        discount_factor: float,
        gae_lambda: float,
        clip_epsilon: float,
        ppo_epochs: int,
        batch_size: int,
        entropy_coefficient: float,
        value_loss_coefficient: float,
        max_gradient_norm: float,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.discount_factor = float(discount_factor)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.max_gradient_norm = float(max_gradient_norm)

        if actor_learning_rate <= 0 or critic_learning_rate <= 0:
            raise ValueError("learning rates must be greater than zero")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.ppo_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("ppo_epochs and batch_size must be positive")

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
            eps=1e-5,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
            eps=1e-5,
        )

    @property
    def device(self) -> torch.device:
        return self.actor.device

    @property
    def n_agents(self) -> int:
        return self.actor.n_agents

    def centralized_observation(
        self,
        local_observations: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Concatenate all local observations for centralized training."""
        observations = np.asarray(
            local_observations,
            dtype=np.float32,
        )
        if observations.shape[0] != self.n_agents:
            raise ValueError(
                "Expected one local observation for every controlled agent."
            )
        return observations.reshape(-1)

    def select_actions(
        self,
        local_observations: Sequence[Sequence[float]],
        deterministic: bool = False,
    ) -> tuple[list[int], list[float]]:
        """Select one decentralized action per predator using the shared actor."""
        observation_array = np.asarray(
            local_observations,
            dtype=np.float32,
        )
        agent_ids = torch.arange(
            self.n_agents,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.actor(
                observation_array,
                agent_ids,
            )
            distribution = Categorical(logits=logits)

            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = distribution.sample()

            log_probabilities = distribution.log_prob(actions)

        return (
            [int(action) for action in actions.cpu().tolist()],
            [
                float(log_probability)
                for log_probability in log_probabilities.cpu().tolist()
            ],
        )

    def estimate_value(
        self,
        centralized_observation: Sequence[float],
    ) -> float:
        """Return one centralized state-value estimate without gradients."""
        with torch.no_grad():
            value = self.critic(centralized_observation)
        return float(value.item())

    def set_entropy_coefficient(self, coefficient: float) -> None:
        """Update the entropy coefficient used by subsequent PPO updates."""
        coefficient = float(coefficient)
        if coefficient < 0.0:
            raise ValueError("entropy coefficient must be non-negative")
        self.entropy_coefficient = coefficient

    def _compute_gae(
        self,
        trajectory: MAPPOEpisodeBuffer,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute generalized advantages and value targets for one episode."""
        if len(trajectory) == 0:
            raise ValueError("Cannot update from an empty trajectory")
        if trajectory.final_centralized_observation is None:
            raise ValueError("Trajectory must be finalized before MAPPO update")

        rewards = np.asarray(trajectory.team_rewards, dtype=np.float32)
        values = np.asarray(trajectory.values, dtype=np.float32)
        dones = np.asarray(trajectory.dones, dtype=np.float32)

        if trajectory.final_done:
            bootstrap_value = 0.0
        else:
            bootstrap_value = self.estimate_value(
                trajectory.final_centralized_observation
            )

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            next_value = (
                bootstrap_value
                if step == len(rewards) - 1
                else values[step + 1]
            )
            non_terminal = 1.0 - dones[step]
            temporal_difference = (
                rewards[step]
                + self.discount_factor * next_value * non_terminal
                - values[step]
            )
            gae = (
                temporal_difference
                + self.discount_factor
                * self.gae_lambda
                * non_terminal
                * gae
            )
            advantages[step] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        trajectories: Sequence[MAPPOEpisodeBuffer] | MAPPOEpisodeBuffer,
    ) -> MAPPOUpdateMetrics:
        """Perform one PPO/MAPPO update from a multi-episode rollout.

        GAE is computed independently inside every trajectory so episode
        boundaries are never crossed. The resulting on-policy samples are then
        concatenated and shuffled into PPO minibatches. This avoids repeatedly
        fitting a single short episode and gives successful and unsuccessful
        trajectories a shared optimization context.
        """
        if isinstance(trajectories, MAPPOEpisodeBuffer):
            rollout = [trajectories]
        else:
            rollout = list(trajectories)

        if not rollout:
            raise ValueError("Cannot update from an empty rollout")

        local_observation_parts: list[np.ndarray] = []
        centralized_observation_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        old_log_probability_parts: list[np.ndarray] = []
        old_value_parts: list[np.ndarray] = []
        advantage_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []

        for trajectory in rollout:
            if len(trajectory) == 0:
                continue

            advantages, returns = self._compute_gae(trajectory)
            local_observations = np.asarray(
                trajectory.local_observations,
                dtype=np.float32,
            )
            centralized_observations = np.asarray(
                trajectory.centralized_observations,
                dtype=np.float32,
            )
            actions = np.asarray(trajectory.actions, dtype=np.int64)
            old_log_probabilities = np.asarray(
                trajectory.log_probabilities,
                dtype=np.float32,
            )
            old_values = np.asarray(trajectory.values, dtype=np.float32)

            if local_observations.shape[1] != self.n_agents:
                raise ValueError(
                    "Trajectory agent count does not match learner"
                )

            local_observation_parts.append(local_observations)
            centralized_observation_parts.append(centralized_observations)
            action_parts.append(actions)
            old_log_probability_parts.append(old_log_probabilities)
            old_value_parts.append(old_values)
            advantage_parts.append(advantages)
            return_parts.append(returns)

        if not advantage_parts:
            raise ValueError("Rollout contains no transitions")

        local_observations = np.concatenate(
            local_observation_parts, axis=0
        )
        centralized_observations = np.concatenate(
            centralized_observation_parts, axis=0
        )
        actions = np.concatenate(action_parts, axis=0)
        old_log_probabilities = np.concatenate(
            old_log_probability_parts, axis=0
        )
        old_values = np.concatenate(old_value_parts, axis=0)
        advantages = np.concatenate(advantage_parts, axis=0)
        returns = np.concatenate(return_parts, axis=0)

        time_steps, n_agents, observation_dim = local_observations.shape

        # Every agent-action pair shares the cooperative team advantage from
        # its time step. Normalize over the full multi-episode rollout rather
        # than independently inside each short episode.
        actor_observations = local_observations.reshape(
            time_steps * n_agents,
            observation_dim,
        )
        actor_agent_ids = np.tile(
            np.arange(n_agents, dtype=np.int64),
            time_steps,
        )
        actor_actions = actions.reshape(-1)
        actor_old_log_probabilities = old_log_probabilities.reshape(-1)
        actor_advantages = np.repeat(advantages, n_agents)
        actor_advantages = (
            actor_advantages - actor_advantages.mean()
        ) / (actor_advantages.std() + 1e-8)

        actor_observation_tensor = torch.as_tensor(
            actor_observations,
            dtype=torch.float32,
            device=self.device,
        )
        actor_agent_id_tensor = torch.as_tensor(
            actor_agent_ids,
            dtype=torch.long,
            device=self.device,
        )
        actor_action_tensor = torch.as_tensor(
            actor_actions,
            dtype=torch.long,
            device=self.device,
        )
        actor_old_log_probability_tensor = torch.as_tensor(
            actor_old_log_probabilities,
            dtype=torch.float32,
            device=self.device,
        )
        actor_advantage_tensor = torch.as_tensor(
            actor_advantages,
            dtype=torch.float32,
            device=self.device,
        )

        centralized_observation_tensor = torch.as_tensor(
            centralized_observations,
            dtype=torch.float32,
            device=self.critic.device,
        )
        return_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.critic.device,
        )
        old_value_tensor = torch.as_tensor(
            old_values,
            dtype=torch.float32,
            device=self.critic.device,
        )

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []

        actor_sample_count = actor_observation_tensor.shape[0]
        critic_sample_count = centralized_observation_tensor.shape[0]

        self.actor.train()
        self.critic.train()

        for _ in range(self.ppo_epochs):
            actor_permutation = torch.randperm(
                actor_sample_count,
                device=self.device,
            )

            for start in range(0, actor_sample_count, self.batch_size):
                batch_indices = actor_permutation[
                    start:start + self.batch_size
                ]

                logits = self.actor(
                    actor_observation_tensor[batch_indices],
                    actor_agent_id_tensor[batch_indices],
                )
                distribution = Categorical(logits=logits)
                new_log_probabilities = distribution.log_prob(
                    actor_action_tensor[batch_indices]
                )
                entropy = distribution.entropy().mean()

                log_ratio = (
                    new_log_probabilities
                    - actor_old_log_probability_tensor[batch_indices]
                )
                ratio = torch.exp(log_ratio)

                advantages_batch = actor_advantage_tensor[batch_indices]
                surrogate_unclipped = ratio * advantages_batch
                surrogate_clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages_batch

                actor_loss = -torch.min(
                    surrogate_unclipped,
                    surrogate_clipped,
                ).mean()
                actor_objective = (
                    actor_loss
                    - self.entropy_coefficient * entropy
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_gradient_norm,
                )
                self.actor_optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.clip_epsilon
                    ).float().mean()

                actor_losses.append(float(actor_loss.item()))
                entropies.append(float(entropy.item()))
                approximate_kls.append(float(approximate_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

            critic_permutation = torch.randperm(
                critic_sample_count,
                device=self.critic.device,
            )

            for start in range(0, critic_sample_count, self.batch_size):
                batch_indices = critic_permutation[
                    start:start + self.batch_size
                ]

                new_values = self.critic(
                    centralized_observation_tensor[batch_indices]
                )
                old_values_batch = old_value_tensor[batch_indices]
                returns_batch = return_tensor[batch_indices]

                clipped_values = old_values_batch + torch.clamp(
                    new_values - old_values_batch,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )
                value_loss_unclipped = (
                    new_values - returns_batch
                ).pow(2)
                value_loss_clipped = (
                    clipped_values - returns_batch
                ).pow(2)
                critic_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (
                    self.value_loss_coefficient * critic_loss
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_gradient_norm,
                )
                self.critic_optimizer.step()

                critic_losses.append(float(critic_loss.item()))

        return MAPPOUpdateMetrics(
            actor_loss=float(np.mean(actor_losses)),
            critic_loss=float(np.mean(critic_losses)),
            entropy=float(np.mean(entropies)),
            approximate_kl=float(np.mean(approximate_kls)),
            clip_fraction=float(np.mean(clip_fractions)),
        )

    def actor_parameter_count(self) -> int:
        """Return trainable parameter count of the shared VQC actor."""
        return sum(
            parameter.numel()
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        )

    def critic_parameter_count(self) -> int:
        """Return trainable parameter count of the centralized critic."""
        return sum(
            parameter.numel()
            for parameter in self.critic.parameters()
            if parameter.requires_grad
        )

    def checkpoint_state(self) -> dict:
        """Return serializable policy, critic, and optimizer state."""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
