"""Pure MADDPG package reserved for subsequent implementation.

No networks, replay buffer, exploration or optimization logic exists yet.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MADDPGSpec:
    name: str = "maddpg"
    implemented: bool = False
    observation_dim: int = 157
    action_dim: int = 3
