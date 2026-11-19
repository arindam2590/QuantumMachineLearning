import numpy as np
import pytest
import torch
from config.maddpg_config import MADDPGConfig
from algorithms.maddpg.maddpg import MADDPG,bellman_target


def model():
    cfg=MADDPGConfig(hidden_layers=(16,8),batch_size=2,replay_capacity=8,warmup_steps=0,
                     gradient_clip_norm=1.,tau=.1)
    learner=MADDPG(['uav_0','uav_1'],5,cfg)
    for i in range(3):
        obs={a:np.full(157,.1+i*.01,np.float32) for a in learner.agent_ids}
        learner.record_transition(obs,np.ones(5),{a:np.zeros(3) for a in obs},
            {a:1. for a in obs},obs,np.ones(5),{a:i==0 for a in obs},{a:i==1 for a in obs})
    return learner


def test_bellman_termination_and_truncation():
    result=bellman_target(torch.tensor([1.,1.]),torch.tensor([10.,10.]),torch.tensor([True,False]),.95)
    assert torch.allclose(result,torch.tensor([1.,10.5]))
    learner=model();batch=learner.replay.sample(3)
    target=learner.targets(batch)
    assert not target.requires_grad
    assert torch.all(target[batch['terminated']]==1.)


def test_updates_gradient_isolation_soft_targets():
    learner=model();batch=learner.replay.sample(2)
    learner.actor_loss(batch,0).backward()
    assert any(p.grad is not None for p in learner.networks.actors[0].parameters())
    assert all(p.grad is None for p in learner.networks.actors[1].parameters())
    for p in learner.networks.parameters():p.grad=None
    actor=[p.clone() for p in learner.networks.actors[0].parameters()]
    critic=[p.clone() for p in learner.networks.critics[0].parameters()]
    old_target=[p.clone() for p in learner.networks.target_actors[0].parameters()]
    result=learner.update(batch)
    assert all(np.isfinite(v) for r in result.values() for v in r.values())
    assert any(not torch.equal(a,b) for a,b in zip(actor,learner.networks.actors[0].parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(critic,learner.networks.critics[0].parameters()))
    for old,current,target in zip(old_target,learner.networks.actors[0].parameters(),learner.networks.target_actors[0].parameters()):
        assert torch.allclose(target,.9*old+.1*current)
    assert all(p.grad is None for p in learner.networks.target_actors.parameters())


def test_actions_checkpoint_and_continuation(tmp_path):
    learner=model();learner.update()
    obs={'uav_1':np.zeros(157,np.float32)}
    deterministic=learner.select_actions(obs)
    assert set(deterministic)=={'uav_1'}
    learner.noise_std=100
    assert np.all(np.abs(learner.select_actions(obs,True)['uav_1'])<=1)
    path=tmp_path/'checkpoint.pt';learner.save_checkpoint(path)
    restored=MADDPG.load_checkpoint(path,device='cpu')
    assert np.array_equal(learner.select_actions(obs)['uav_1'],restored.select_actions(obs)['uav_1'])
    assert np.array_equal(learner.select_actions(obs,True)['uav_1'],restored.select_actions(obs,True)['uav_1'])
    assert restored.environment_steps==learner.environment_steps and restored.update_steps==learner.update_steps
    a,b=learner.update(),restored.update()
    assert a==b
    assert all(torch.equal(a,b) for a,b in zip(learner.networks.parameters(),restored.networks.parameters()))
