import pytest
import torch
from config.maddpg_config import MADDPGConfig
from algorithms.maddpg.networks import Actor,Critic,MADDPGNetworks,hard_update,soft_update


@pytest.mark.parametrize('fleet',[2,4,8,16,20])
def test_shapes_targets_and_independence(fleet):
    torch.manual_seed(42)
    models=MADDPGNetworks(fleet,13*fleet,config=MADDPGConfig(hidden_layers=(32,16)))
    action=models.actors[0](torch.randn(5,157))
    assert action.shape==(5,3) and torch.all(action.abs()<=1)
    assert models.critics[0](torch.randn(5,13*fleet),torch.randn(5,3*fleet)).shape==(5,1)
    for online,target in [(models.actors,models.target_actors),(models.critics,models.target_critics)]:
        for network,copy in zip(online,target):
            for a,b in zip(network.parameters(),copy.parameters()):
                assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
                assert not b.requires_grad
        assert next(online[0].parameters()).data_ptr()!=next(online[1].parameters()).data_ptr()
    models.train()
    assert not models.target_actors.training and not models.target_critics.training


def test_gradient_flow_and_updates():
    actor=Actor(hidden_layers=(32,16));critic=Critic(20,2,(32,16))
    observations=torch.randn(4,157,requires_grad=True)
    actions=torch.cat([actor(observations),torch.zeros(4,3)],dim=-1)
    critic(torch.randn(4,20),actions).sum().backward()
    assert observations.grad is not None
    assert all(p.grad is not None for p in actor.parameters())
    assert all(p.grad is not None for p in critic.parameters())
    target=Actor(hidden_layers=(32,16))
    hard_update(target,actor)
    assert all(torch.equal(a,b) for a,b in zip(actor.parameters(),target.parameters()))
    with torch.no_grad():
        for p in target.parameters(): p.zero_()
        for p in actor.parameters(): p.fill_(2)
    soft_update(target,actor,.25)
    assert all(torch.allclose(p,torch.full_like(p,.5)) for p in target.parameters())
    before=[p.clone() for p in target.parameters()]
    soft_update(target,actor,0)
    assert all(torch.equal(a,b) for a,b in zip(before,target.parameters()))
    soft_update(target,actor,1)
    assert all(torch.equal(a,b) for a,b in zip(actor.parameters(),target.parameters()))
    with pytest.raises(ValueError): soft_update(target,actor,1.1)


def test_architecture_and_validation():
    actor=Actor()
    linear=[m for m in actor.modules() if isinstance(m,torch.nn.Linear)]
    assert [(m.in_features,m.out_features) for m in linear]==[(157,256),(256,256),(256,64),(64,3)]
    assert actor(torch.randn(157)).shape==(3,)
    with pytest.raises(ValueError): actor(torch.randn(158))
    with pytest.raises(ValueError): Critic(10,2)(torch.randn(2,10),torch.randn(2,3))
