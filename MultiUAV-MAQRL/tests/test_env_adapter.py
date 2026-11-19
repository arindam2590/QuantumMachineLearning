import numpy as np
import pytest
from algorithms.env_adapter import ParallelEnvAdapter


def test_real_environment_replay_factory_and_reset():
    pytest.importorskip('pybullet')
    from env import MultiDroneEnv
    from config import EnvConfig
    from config.maddpg_config import MADDPGConfig
    from algorithms.registry import create_algorithm
    with MultiDroneEnv(EnvConfig(gui=False,scenario='open_field',num_drones=2,max_episode_steps=2)) as env:
        adapter=ParallelEnvAdapter(env)
        obs,_=adapter.reset(seed=42)
        first=adapter.centralized_state()
        learner=create_algorithm('maddpg',adapter=adapter,config=MADDPGConfig(hidden_layers=(16,8)))
        assert first.shape==(19,) and learner.agent_ids==tuple(env.possible_agents)
        adapter.step(np.zeros((2,3),np.float32))
        learner.record_transition(**adapter.last_transition)
        adapter.step(learner.select_actions(adapter.active_observations()))
        learner.record_transition(**adapter.last_transition)
        assert not env.agents
        padded,mask=adapter.ordered_observations()
        assert padded.shape==(2,157) and not mask.any()
        assert learner.replay.sample(2)['observations'].shape==(2,2,157)
        adapter.reset(seed=42)
        assert np.array_equal(adapter.centralized_state(),first)
        assert adapter.last_transition is None


class Synthetic:
    possible_agents=['uav_0','uav_1']
    def __init__(self):
        from gymnasium.spaces import Box
        self.agents=[]
        self.observation_spaces={a:Box(-1,1,(2,),dtype=np.float32) for a in self.possible_agents}
        self.action_spaces={a:Box(-1,1,(3,),dtype=np.float32) for a in self.possible_agents}
    def reset(self,**kwargs):
        self.agents=list(self.possible_agents)
        return {a:np.ones(2,np.float32) for a in self.agents},{}
    def step(self,actions):
        before=list(self.agents)
        assert set(actions)==set(before)
        self.agents=[a for a in before if a!='uav_0']
        return ({a:np.full(2,.5,np.float32) for a in before},{a:1. for a in before},
                {a:a=='uav_0' for a in before},{a:False for a in before},{a:{} for a in before})
    def close(self):pass


def test_individual_retirement_and_fallback():
    pytest.importorskip('gymnasium')
    adapter=ParallelEnvAdapter(Synthetic())
    adapter.reset()
    assert adapter.global_state_dim==6
    adapter.step(np.zeros((2,3)))
    assert set(adapter.active_observations())=={'uav_1'}
    assert adapter.last_transition['next_observations']['uav_0'][0]==.5
    assert adapter.last_transition['next_global_state'][0]==.5
    assert adapter.ordered_observations()[1].tolist()==[False,True]
    with pytest.raises(ValueError):adapter.action_dictionary({'uav_0':np.zeros(3),'uav_1':np.zeros(3)})
    adapter.step(np.zeros((2,3)))
    assert set(adapter.last_transition['actions'])=={'uav_1'}
