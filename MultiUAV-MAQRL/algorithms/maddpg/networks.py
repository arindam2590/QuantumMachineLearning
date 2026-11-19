"""Pure MADDPG MLPs: independent actors, critics and frozen target copies."""
import copy
import math
import torch
from torch import nn
from config.maddpg_config import MADDPGConfig


def _dimension(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _mlp(input_dim, output_dim, hidden_layers, output_activation=None):
    _dimension('input_dim', input_dim)
    _dimension('output_dim', output_dim)
    if not hidden_layers:
        raise ValueError('hidden_layers must not be empty')
    layers = []
    previous = input_dim
    for width in hidden_layers:
        _dimension('hidden width', width)
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Local observation -> three deterministic normalized velocity commands."""
    def __init__(self, observation_dim=157, hidden_layers=(256,256,64)):
        super().__init__()
        self.observation_dim = observation_dim
        self.model = _mlp(observation_dim, 3, hidden_layers, nn.Tanh())

    def forward(self, observation):
        if observation.ndim < 1 or observation.shape[-1] != self.observation_dim:
            raise ValueError('Actor expects local observations with the configured last dimension')
        return self.model(observation)


class Critic(nn.Module):
    """Numeric centralized state and flat joint actions -> scalar Q per sample.

    Agent order must follow env.possible_agents. This module does not flatten
    dictionaries, expose state to actors, or choose an inactive-agent encoding.
    """
    def __init__(self, global_state_dim, num_agents, hidden_layers=(256,256,64)):
        super().__init__()
        _dimension('global_state_dim', global_state_dim)
        _dimension('num_agents', num_agents)
        self.global_state_dim, self.num_agents = global_state_dim, num_agents
        self.joint_action_dim = 3*num_agents
        self.model = _mlp(global_state_dim+self.joint_action_dim,1,hidden_layers)

    def forward(self, global_state, joint_actions):
        if (global_state.ndim < 1 or joint_actions.ndim < 1 or
                global_state.shape[-1] != self.global_state_dim or
                joint_actions.shape[-1] != self.joint_action_dim or
                global_state.shape[:-1] != joint_actions.shape[:-1]):
            raise ValueError('Critic requires matching batch shapes and configured state/joint-action dimensions')
        return self.model(torch.cat((global_state,joint_actions),dim=-1))


def hard_update(target, source):
    """Copy parameters and buffers exactly, without aliasing tensors."""
    target.load_state_dict(source.state_dict(), strict=True)


def soft_update(target, source, tau):
    """Polyak update: target <- (1-tau)*target + tau*source."""
    if not isinstance(tau,(int,float)) or isinstance(tau,bool) or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('tau must be finite and in [0,1]')
    target_state, source_state = target.state_dict(), source.state_dict()
    if target_state.keys() != source_state.keys() or any(target_state[k].shape != source_state[k].shape for k in target_state):
        raise ValueError('Target and source architectures must match')
    with torch.no_grad():
        for name, value in target_state.items():
            incoming = source_state[name]
            if value.is_floating_point() or value.is_complex():
                value.mul_(1-tau).add_(incoming,alpha=tau)
            elif tau > 0:
                value.copy_(incoming)


class MADDPGNetworks(nn.Module):
    """One independent online/target actor and critic pair per agent.

    This container implements no optimizer, losses, exploration or learning loop.
    global_state_dim is required because the frozen environment exposes a
    dictionary rather than a canonical critic tensor.
    """
    def __init__(self, num_agents, global_state_dim, config=None, observation_dim=157):
        super().__init__()
        config = config or MADDPGConfig()
        config.validate()
        _dimension('num_agents',num_agents)
        self.actors = nn.ModuleList([Actor(observation_dim,config.hidden_layers) for _ in range(num_agents)])
        self.critics = nn.ModuleList([Critic(global_state_dim,num_agents,config.hidden_layers) for _ in range(num_agents)])
        self.target_actors = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)
        for target in (self.target_actors,self.target_critics):
            target.requires_grad_(False)
            target.eval()

    def train(self, mode=True):
        super().train(mode)
        self.target_actors.eval()
        self.target_critics.eval()
        return self
