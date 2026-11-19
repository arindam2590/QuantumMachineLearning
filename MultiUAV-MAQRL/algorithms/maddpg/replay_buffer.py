"""Bounded uniform joint-transition replay for Pure MADDPG (one-step only)."""
import numpy as np
import torch


class ReplayBuffer:
    """Agent axes follow agent_ids exactly; inactive slots are zero padded.

    Pass only pre-step active local observations/actions/rewards/flags. Pass
    post-step observations INCLUDING final observations for finishing agents.
    Global states must already be finite numeric vectors, not simulator dicts.
    Transitions are independent; callers never join resets into a transition.
    """
    def __init__(self, capacity, agent_ids, observation_dim=157, global_state_dim=None, seed=42):
        for name,value in [('capacity',capacity),('observation_dim',observation_dim),('global_state_dim',global_state_dim)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'{name} must be a positive integer')
        self.agent_ids=tuple(agent_ids)
        if not self.agent_ids or any(not isinstance(a,str) or a=='__all__' for a in self.agent_ids) or len(set(self.agent_ids))!=len(self.agent_ids):
            raise ValueError('Require unique string agent IDs in canonical order')
        self.capacity,self.observation_dim,self.global_state_dim=capacity,observation_dim,global_state_dim
        self.rng=np.random.default_rng(seed)
        self._data=[]
        self._cursor=0

    def __len__(self):
        return len(self._data)

    @property
    def nbytes(self):
        """Stored NumPy payload bytes (excludes Python object overhead)."""
        return sum(v.nbytes for row in self._data for v in row.values())

    @staticmethod
    def _array(value,shape,name):
        result=np.asarray(value,dtype=np.float32)
        if result.shape!=shape or not np.isfinite(result).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
        return result.copy()

    def add(self, observations, global_state, actions, rewards, next_observations,
            next_global_state, terminated, truncated):
        active=set(actions)
        if not active or not active.issubset(self.agent_ids):
            raise ValueError('Actions must identify known pre-step active agents')
        for name,mapping in [('observations',observations),('rewards',rewards),
                             ('next_observations',next_observations),
                             ('terminated',terminated),('truncated',truncated)]:
            keys=set(mapping)-({'__all__'} if name in ('terminated','truncated') else set())
            if keys!=active:
                raise ValueError(f'{name} keys must exactly match pre-step active agents')
        n=len(self.agent_ids)
        row=dict(observations=np.zeros((n,self.observation_dim),np.float32),
                 next_observations=np.zeros((n,self.observation_dim),np.float32),
                 actions=np.zeros((n,3),np.float32),rewards=np.zeros(n,np.float32),
                 terminated=np.zeros(n,bool),truncated=np.zeros(n,bool),
                 valid_agents=np.zeros(n,bool),next_active_agents=np.zeros(n,bool))
        for i,agent in enumerate(self.agent_ids):
            if agent not in active:
                continue
            for flags in (terminated,truncated):
                if not isinstance(flags[agent],(bool,np.bool_)):
                    raise ValueError('Termination/truncation flags must be boolean')
            if terminated[agent] and truncated[agent]:
                raise ValueError('An agent cannot terminate and truncate simultaneously')
            row['observations'][i]=self._array(observations[agent],(self.observation_dim,),'observation')
            row['next_observations'][i]=self._array(next_observations[agent],(self.observation_dim,),'next observation')
            action=self._array(actions[agent],(3,),'executed action')
            if np.any(np.abs(action)>1):
                raise ValueError('Executed actions must be normalized to [-1,1]; replay never clips them')
            row['actions'][i]=action
            row['rewards'][i]=self._array(rewards[agent],(),'reward')
            row['terminated'][i]=terminated[agent]
            row['truncated'][i]=truncated[agent]
            row['valid_agents'][i]=True
            row['next_active_agents'][i]=not (terminated[agent] or truncated[agent])
        row['global_state']=self._array(global_state,(self.global_state_dim,),'global state')
        row['next_global_state']=self._array(next_global_state,(self.global_state_dim,),'next global state')
        # Final observations are valid even for truncation/termination.
        row['next_valid_agents']=row['valid_agents'].copy()
        row['bootstrap_mask']=row['valid_agents'] & ~row['terminated']
        row['episode_done']=np.asarray(not row['next_active_agents'].any(),dtype=bool)
        if len(self._data)<self.capacity:
            self._data.append(row)
        else:
            self._data[self._cursor]=row
        self._cursor=(self._cursor+1)%self.capacity

    def sample(self,batch_size,device='cpu'):
        """Uniform sampling without replacement; returns independent torch tensors.

        actions: [B,N,3], observations: [B,N,O], states: [B,S],
        rewards/masks/flags: [B,N], episode_done: [B]. Flatten joint actions
        to [B,3N] when passing to a centralized critic.
        """
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 0<batch_size<=len(self):
            raise ValueError('batch_size must be positive and not exceed stored transitions')
        indices=self.rng.choice(len(self),batch_size,replace=False)
        return {key:torch.as_tensor(np.stack([self._data[i][key] for i in indices]),device=device)
                for key in self._data[0]}

    def clear(self,seed=None):
        self._data.clear()
        self._cursor=0
        if seed is not None:
            self.rng=np.random.default_rng(seed)
