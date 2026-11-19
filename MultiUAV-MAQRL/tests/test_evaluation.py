import copy
import json
import numpy as np
import pytest
import torch
pytest.importorskip('pybullet')
from config import EnvConfig
from config.maddpg_config import MADDPGConfig
from algorithms.maddpg.maddpg import MADDPG
from algorithms.evaluation import evaluate,aggregate,validation_main


def test_aggregation():
    rows=[dict(fleet=2,arrivals=1,team_success=False,collisions=1,reward=4.,energy_use=.2,duration=10.,arrival_times={'a':4.},successful_path_efficiency={'a':.5}),
          dict(fleet=2,arrivals=2,team_success=True,collisions=0,reward=6.,energy_use=.4,duration=8.,arrival_times={'a':2.,'b':6.},successful_path_efficiency={'a':.8,'b':.8})]
    result=aggregate(rows)
    assert result['team_success_rate']==.5 and result['individual_arrival_rate']==.75
    assert result['mean_travel_time']==4 and result['mean_reward']==5
    assert result['collision_count']==1 and result['mean_path_efficiency']==pytest.approx(.7)


def test_deterministic_no_updates_checkpoint_and_metrics(tmp_path):
    learner=MADDPG(['uav_0','uav_1'],19,MADDPGConfig(hidden_layers=(8,)))
    before={k:v.clone() for k,v in learner.networks.state_dict().items()}
    rng=copy.deepcopy(learner.rng.bit_generator.state)
    modes=[m.training for m in learner.networks.modules()]
    cfg=EnvConfig(scenario='open_field',num_drones=2,gui=False,max_episode_steps=3)
    a=evaluate(learner,cfg,seeds=[100,101],output=tmp_path/'metrics')
    b=evaluate(learner,cfg,seeds=[100,101])
    assert a==b and a['episodes']==2
    assert (tmp_path/'metrics.csv').exists() and json.loads((tmp_path/'metrics.json').read_text())['episodes']==2
    assert all(torch.equal(v,before[k]) for k,v in learner.networks.state_dict().items())
    assert all(p.grad is None for p in learner.networks.parameters())
    assert learner.update_steps==0 and learner.environment_steps==0 and len(learner.replay)==0
    assert learner.rng.bit_generator.state==rng
    assert modes==[m.training for m in learner.networks.modules()]
    path=tmp_path/'model.pt';learner.save_checkpoint(path)
    result=validation_main(['--algo','maddpg','--checkpoint',str(path),'--scenario','open_field',
        '--num-drones','2','--seeds','100','101','--episode-steps','3','--output',str(tmp_path/'cli')])
    assert result==a
    with pytest.raises(ValueError,match='overlap'):evaluate(learner,cfg,seeds=[42])
    with pytest.raises(ValueError,match='incompatible'):
        evaluate(learner,EnvConfig(gui=False,num_drones=4),seeds=[100])
