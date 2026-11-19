"""Opt-in experiment rewards; frozen environment rewards remain untouched.

Only an explicit terminal ``out_of_bounds`` status is eligible. A generic
termination flag is deliberately insufficient. Values are penalty magnitudes.
"""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class RewardAdjustmentConfig:
    out_of_bounds_penalty: float = 0.0
    version: str = 'boundary_penalty_v1'

    def __post_init__(self):
        if (isinstance(self.out_of_bounds_penalty, bool) or
                not math.isfinite(self.out_of_bounds_penalty) or self.out_of_bounds_penalty < 0):
            raise ValueError('out_of_bounds_penalty must be a finite nonnegative magnitude')
        if self.version != 'boundary_penalty_v1':
            raise ValueError('Unsupported reward adjustment version')

    def metadata(self):
        return asdict(self)

    @classmethod
    def from_learner(cls, learner):
        return cls(**(getattr(learner, 'training_state', None) or {}).get('reward_adjustment', {}))


class EpisodeRewardAdjustment:
    def __init__(self, config=None):
        self.config = config or RewardAdjustmentConfig()
        self.reset()

    def reset(self):
        self._finished = set()

    def apply(self, rewards, terminated, truncated, infos):
        """Return new learning rewards and signed adjustments, without mutation."""
        adjusted = dict(rewards)
        penalties = {}
        for agent, reward in rewards.items():
            info = infos[agent]
            terminal = bool(terminated.get(agent, False))
            timeout = bool(truncated.get(agent, False))
            eligible = (agent not in self._finished and terminal and not timeout and
                        info.get('out_of_bounds') is True and
                        info.get('status') == 'out_of_bounds' and
                        not info.get('collision', False) and not info.get('success', False) and
                        not info.get('reached_goal', False))
            penalty = -float(self.config.out_of_bounds_penalty) if eligible else 0.0
            adjusted[agent] = float(reward) + penalty
            if not math.isfinite(adjusted[agent]):
                raise ValueError('Nonfinite adjusted reward')
            penalties[agent] = penalty
            if terminal or timeout:
                self._finished.add(agent)
        return adjusted, penalties
