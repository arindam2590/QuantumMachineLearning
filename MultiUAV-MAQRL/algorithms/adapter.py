"""RLlib MultiAgentEnv facade; simulator remains unchanged."""
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env import MultiDroneEnv,EnvConfig

class UAVAdapter(MultiAgentEnv):
    def __init__(self,config=None):
        super().__init__()
        if isinstance(config,dict):
            config=EnvConfig(gui=False,scenario=config.get('scenario','open_field'),num_drones=config.get('num_drones',2))
        self.env=MultiDroneEnv(config or EnvConfig(gui=False))
        self.possible_agents=list(self.env.possible_agents)
        self.agents=[]
        self._agent_ids=set(self.possible_agents)
        self.observation_spaces=self.env.observation_spaces
        self.action_spaces=self.env.action_spaces
        self.ended={};self.latest={}

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed,options=options)
        obs,infos=self.env.reset(seed=seed,options=options)
        self.agents=list(self.env.agents);self.ended={};self.latest=dict(obs)
        return obs,infos

    def step(self,actions):
        if set(actions)!=set(self.agents):raise ValueError('Actions required for exactly active agents')
        obs,rewards,term,trunc,infos=self.env.step(actions)
        term.pop('__all__',None);trunc.pop('__all__',None)
        for a in obs:
            if term[a] or trunc[a]:self.ended[a]=(term[a],trunc[a])
        self.agents=list(self.env.agents);self.latest=obs
        # Mixed terminal/time-limit endings must finish the RLlib episode too.
        term['__all__']=not self.agents and all(t for t,x in self.ended.values())
        trunc['__all__']=not self.agents and not term['__all__']
        return obs,rewards,term,trunc,infos

    def pack(self,obs,active=None):
        ids=self.possible_agents
        values=np.stack([obs.get(a,np.zeros(157,np.float32)) for a in ids]).astype(np.float32)
        mask=np.array([a in (self.agents if active is None else active) for a in ids],np.float32)
        return values,np.concatenate([(values*mask[:,None]).flatten(),mask])

    def render(self):return self.env.render()
    def close(self):self.env.close()
