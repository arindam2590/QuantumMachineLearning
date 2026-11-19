"""Pure MADDPG settings, independent of the frozen simulator configuration.

Learning rates, gamma, tau and hidden widths use the user-supplied paper
initialization. Other defaults are development choices, not tuned results.
Decay is multiplicative per policy exploration step, after warm-up.
"""
from dataclasses import dataclass
import math


@dataclass
class MADDPGConfig:
    gradient_clip_norm: float | None = None
    actor_lr: float = .001
    critic_lr: float = .001
    gamma: float = .95
    tau: float = .01
    hidden_layers: tuple[int, ...] = (256, 256, 64)
    optimizer: str = 'Adam'
    replay_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000
    policy_update_frequency: int = 1  # environment transitions between updates
    exploration_noise_std: float = .2  # normalized action units
    exploration_noise_decay: float = .99999
    minimum_exploration_noise: float = .02
    max_training_steps: int = 1_000_000
    validation_interval: int = 10_000
    checkpoint_interval: int = 10_000
    seed: int = 42
    device: str = 'cpu'  # cpu, cuda, mps, auto; availability resolved at runtime
    evaluation_episodes: int = 10

    def validate(self):
        if self.gradient_clip_norm is not None and (isinstance(self.gradient_clip_norm,bool) or not isinstance(self.gradient_clip_norm,(int,float)) or not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm<=0):
            raise ValueError('gradient_clip_norm must be positive or None')
        for name in ('actor_lr', 'critic_lr'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name, lower, upper, strict_lower in (
            ('gamma',0,1,False), ('tau',0,1,True),
            ('exploration_noise_decay',0,1,True),
            ('exploration_noise_std',0,float('inf'),False),
            ('minimum_exploration_noise',0,float('inf'),False)):
            value=getattr(self,name)
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                    or value>upper or (value<=lower if strict_lower else value<lower)):
                raise ValueError(f'Invalid {name}')
        for name in ('replay_capacity','batch_size','policy_update_frequency','max_training_steps',
                     'validation_interval','checkpoint_interval','evaluation_episodes'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('warmup_steps','seed'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.batch_size>self.replay_capacity:
            raise ValueError('batch_size must not exceed replay_capacity')
        if self.warmup_steps>=self.max_training_steps:
            raise ValueError('warmup_steps must be smaller than max_training_steps')
        if self.minimum_exploration_noise>self.exploration_noise_std:
            raise ValueError('minimum noise must not exceed initial noise')
        if not isinstance(self.hidden_layers,tuple) or not self.hidden_layers or any(
            isinstance(width,bool) or not isinstance(width,int) or width<=0 for width in self.hidden_layers):
            raise ValueError('hidden_layers must be a nonempty tuple of positive integers')
        if self.optimizer!='Adam':
            raise ValueError('Only Adam is supported for this baseline')
        if self.device not in ('cpu','cuda','mps','auto'):
            raise ValueError('device must be cpu, cuda, mps, or auto')
