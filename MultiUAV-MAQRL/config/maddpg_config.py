"""Pure MADDPG algorithm settings, independent of environment configuration."""
from dataclasses import dataclass
import math

@dataclass
class MADDPGConfig:
    actor_lr: float = .001
    critic_lr: float = .001
    gamma: float = .95
    tau: float = .01
    hidden_layers: tuple = (256,256,64)
    batch_size: int = 256
    replay_capacity: int = 100000
    warmup_steps: int = 1000
    update_interval: int = 1
    noise_std: float = .2
    seed: int = 42

    def validate(self):
        for k in ('actor_lr','critic_lr','noise_std'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<0:raise ValueError(k)
        if not 0<=self.gamma<=1 or not 0<self.tau<=1:raise ValueError('gamma/tau')
        for k in ('batch_size','replay_capacity','update_interval'):
            if getattr(self,k)<1:raise ValueError(k)
        if self.batch_size>self.replay_capacity or self.warmup_steps<0:raise ValueError('replay/warmup')
        if not self.hidden_layers or any(x<1 for x in self.hidden_layers):raise ValueError('hidden_layers')
