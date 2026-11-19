"""Public-API-only parallel dictionary adapter; no environment transformations."""
import copy
import numpy as np


class ParallelEnvAdapter:
    """Stable joint ordering and numeric centralized-state projection.

    For this environment global state encodes each UAV's world position (3),
    velocity (3), battery (1), active flag (1), then destination (3): 8N+3.
    Static/dynamic obstacle lists are deliberately excluded, avoiding variable
    dimensions/body IDs. This is privileged critic-only state, never actor input.
    Without global_state(), use zero-padded local observations plus active mask.
    """
    def __init__(self, env):
        self.env=env
        self.agent_ids=tuple(env.possible_agents)
        if not self.agent_ids or len(set(self.agent_ids))!=len(self.agent_ids):
            raise ValueError('Require stable unique possible_agents')
        shapes={env.observation_spaces[a].shape for a in self.agent_ids}
        if len(shapes)!=1 or len(next(iter(shapes)))!=1:
            raise ValueError('Require uniform vector observation spaces')
        self.observation_dim=next(iter(shapes))[0]
        if any(env.action_spaces[a].shape!=(3,) for a in self.agent_ids):
            raise ValueError('Require 3D action spaces')
        self.has_global_state=callable(getattr(env,'global_state',None))
        self.global_state_dim=8*len(self.agent_ids)+3 if self.has_global_state else (self.observation_dim+1)*len(self.agent_ids)
        self.observations=None
        self.last_transition=None

    def active_observations(self):
        if self.observations is None: raise RuntimeError('Call reset first')
        return {a:self.observations[a].copy() for a in self.agent_ids if a in self.env.agents}

    def ordered_observations(self):
        active=self.active_observations()
        return np.stack([active.get(a,np.zeros(self.observation_dim,np.float32)) for a in self.agent_ids]), np.array([a in active for a in self.agent_ids],bool)

    def centralized_state(self, final_observations=False):
        if self.has_global_state:
            state=self.env.global_state()
            values=[]
            for a in self.agent_ids:
                uav=state['uavs'][a]
                values.extend([*uav['position'],*uav['velocity'],uav['battery'],float(uav['active'])])
            values.extend(state['destination'])
            result=np.asarray(values,np.float32)
        else:
            if final_observations:
                obs=np.stack([self.observations.get(a,np.zeros(self.observation_dim,np.float32)) for a in self.agent_ids])
                mask=np.array([a in self.observations for a in self.agent_ids],bool)
            else:
                obs,mask=self.ordered_observations()
            result=np.concatenate([obs.ravel(),mask.astype(np.float32)])
        if result.shape!=(self.global_state_dim,) or not np.isfinite(result).all():
            raise ValueError('Invalid centralized state')
        return result.copy()

    def action_dictionary(self, actions):
        active=set(self.env.agents)
        if isinstance(actions,dict):
            if set(actions)!=active: raise ValueError('Action keys must match current active agents')
            result={a:np.asarray(actions[a],np.float32).copy() for a in self.agent_ids if a in active}
        else:
            values=np.asarray(actions,np.float32)
            if values.shape!=(len(self.agent_ids),3): raise ValueError('Expected canonical [N,3] actions')
            result={a:values[i].copy() for i,a in enumerate(self.agent_ids) if a in active}
        for a,value in result.items():
            if not self.env.action_spaces[a].contains(value): raise ValueError(f'Invalid action for {a}')
        return result

    def reset(self, *, seed=None, options=None):
        observations,infos=self.env.reset(seed=seed,options=options)
        self.observations=copy.deepcopy(observations)
        self.last_transition=None
        return observations,infos

    def step(self, actions):
        before=self.active_observations()
        state=self.centralized_state()
        executed=self.action_dictionary(actions)
        observations,rewards,terminated,truncated,infos=self.env.step(executed)
        self.observations=copy.deepcopy(observations)
        next_state=self.centralized_state(final_observations=True)
        self.last_transition=copy.deepcopy(dict(observations=before,global_state=state,
            actions=executed,rewards=rewards,next_observations=observations,next_global_state=next_state,
            terminated=terminated,truncated=truncated))
        return observations,rewards,terminated,truncated,infos

    def close(self):
        self.env.close()
