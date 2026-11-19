import numpy as np
import pytest
import torch
from algorithms.maddpg.replay_buffer import ReplayBuffer


def insert(buffer,active=None,value=1.,done=None):
    active=buffer.agent_ids if active is None else active
    obs={a:np.full(157,value,np.float32) for a in reversed(active)}
    actions={a:np.full(3,.01*i,np.float32) for i,a in enumerate(buffer.agent_ids) if a in active}
    term={a:done=='term' for a in active};trunc={a:done=='trunc' for a in active}
    term['__all__']=done=='term';trunc['__all__']=done=='trunc'
    buffer.add(obs,np.ones(5)*value,actions,{a:value for a in active},obs,np.ones(5)*(value+1),term,trunc)
    return obs


@pytest.mark.parametrize('fleet',[2,4,8,16,20])
def test_shapes_ordering_and_deterministic_sampling(fleet):
    ids=[f'uav_{i}' for i in range(fleet)]
    a,b=[ReplayBuffer(3,ids,global_state_dim=5,seed=42) for _ in range(2)]
    for i in range(5):
        insert(a,value=i);insert(b,value=i)
    assert len(a)==3
    first,second=a.sample(3),b.sample(3)
    assert all(torch.equal(first[k],second[k]) for k in first)
    assert first['observations'].shape==(3,fleet,157)
    assert first['actions'].shape==(3,fleet,3)
    assert first['observations'].dtype==torch.float32
    assert first['terminated'].dtype==torch.bool
    assert set(first['rewards'][:,0].tolist())=={2,3,4}
    for i in range(fleet): assert torch.allclose(first['actions'][:,i,0],torch.full((3,),.01*i))


def test_final_observations_masks_and_no_aliasing():
    buffer=ReplayBuffer(3,['uav_0','uav_1'],global_state_dim=5)
    obs=insert(buffer,active=['uav_1'],done='trunc')
    obs['uav_1'][:]=99
    batch=buffer.sample(1)
    assert torch.all(batch['observations'][0,0]==0)
    assert torch.all(batch['observations'][0,1]==1)
    assert batch['valid_agents'].tolist()==[[False,True]]
    assert batch['bootstrap_mask'].tolist()==[[False,True]]
    assert not batch['next_active_agents'].any() and batch['episode_done'].item()
    buffer.clear(seed=42)
    insert(buffer,done='term')
    assert not buffer.sample(1)['bootstrap_mask'].any()
    assert buffer.nbytes>0
    buffer.clear()
    assert len(buffer)==0 and buffer.nbytes==0
    with pytest.raises(ValueError):buffer.sample(1)


def test_invalid_inputs_and_capacity():
    with pytest.raises(ValueError):ReplayBuffer(0,['uav_0'],global_state_dim=5)
    with pytest.raises(ValueError):ReplayBuffer(1,['a','a'],global_state_dim=5)
    buffer=ReplayBuffer(1,['a'],global_state_dim=5)
    with pytest.raises(ValueError):buffer.add({'a':np.zeros(157)},np.zeros(5),{'a':[2,0,0]},
        {'a':0},{'a':np.zeros(157)},np.zeros(5),{'a':False},{'a':False})
    assert len(buffer)==0
