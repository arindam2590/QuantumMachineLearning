import pytest
from config.maddpg_config import MADDPGConfig
from algorithms.cli import prepare


def test_defaults_and_independent_instances():
    a,b=MADDPGConfig(),MADDPGConfig()
    a.validate()
    assert a.actor_lr==a.critic_lr==.001
    assert a.gamma==.95 and a.tau==.01 and a.hidden_layers==(256,256,64)
    a.batch_size=128
    assert b.batch_size==256


@pytest.mark.parametrize('kwargs',[{'actor_lr':0},{'critic_lr':float('nan')},{'gamma':1.1},
    {'tau':0},{'hidden_layers':(0,64)},{'batch_size':100001},{'warmup_steps':-1},
    {'policy_update_frequency':0},{'exploration_noise_decay':0},
    {'minimum_exploration_noise':.5},{'max_training_steps':1000},
    {'validation_interval':0},{'checkpoint_interval':0},{'seed':True},
    {'device':'tpu'},{'optimizer':'SGD'}])
def test_invalid_values(kwargs):
    with pytest.raises(ValueError): MADDPGConfig(**kwargs).validate()


def test_overrides_and_evaluation_mode():
    result=prepare('validate',['--algo','maddpg','--actor-lr','.0005','--batch-size','64',
        '--hidden-layers','128','64','--device','cpu','--seed','7'])
    assert result['algorithm_config']['actor_lr']==.0005
    assert result['algorithm_config']['batch_size']==64
    assert result['algorithm_config']['hidden_layers']==(128,64)
    assert result['algorithm_config']['seed']==7
    assert not result['exploration_enabled']
    with pytest.raises(SystemExit):
        prepare('train',['--algo','maddpg','--tau','0'])


def test_accelerator_request_configuration_is_valid_without_runtime_resolution():
    MADDPGConfig(device='mps').validate()
