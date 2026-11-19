import copy
import math
import numpy as np
import pytest
from algorithms.reward_adjustments import RewardAdjustmentConfig,EpisodeRewardAdjustment


def apply(adjustment, status='out_of_bounds', terminal=True, timeout=False, **overrides):
    info=dict(status=status,out_of_bounds=True,collision=False,success=False)
    info.update(overrides)
    return adjustment.apply({'uav_0':2.5},{'uav_0':terminal},{'uav_0':timeout},{'uav_0':info})


def test_boundary_once_per_agent_and_episode():
    adjustment=EpisodeRewardAdjustment(RewardAdjustmentConfig(20))
    assert apply(adjustment)==({'uav_0':-17.5},{'uav_0':-20.})
    assert apply(adjustment)==({'uav_0':2.5},{'uav_0':0.})
    adjustment.reset()
    assert apply(adjustment)[0]['uav_0']==-17.5
    rewards={'uav_0':-1.,'uav_1':4.}
    infos={'uav_0':dict(status='out_of_bounds',out_of_bounds=True),
           'uav_1':dict(status='active',out_of_bounds=False)}
    adjustment.reset();before=copy.deepcopy(infos)
    adjusted,penalties=adjustment.apply(rewards,{'uav_0':True,'uav_1':False},{},infos)
    assert adjusted=={'uav_0':-21.,'uav_1':4.}
    assert penalties=={'uav_0':-20.,'uav_1':0.}
    assert rewards=={'uav_0':-1.,'uav_1':4.} and infos==before


@pytest.mark.parametrize('status,terminal,timeout,overrides',[
    ('collision',True,False,{'collision':True}),
    ('arrived',True,False,{'success':True}),
    ('truncated',False,True,{}),
    ('battery_depleted',True,False,{}),
    ('out_of_bounds',False,False,{}),
    ('out_of_bounds',True,False,{'out_of_bounds':False}),
    ('out_of_bounds',True,False,{'collision':True}),
    ('out_of_bounds',True,True,{}),
])
def test_no_penalty_for_other_outcomes(status,terminal,timeout,overrides):
    assert apply(EpisodeRewardAdjustment(RewardAdjustmentConfig(20)),status,terminal,timeout,**overrides)[0]['uav_0']==2.5


def test_disabled_default_and_legacy_checkpoint():
    assert apply(EpisodeRewardAdjustment())[0]['uav_0']==2.5
    class Old:training_state=None
    assert RewardAdjustmentConfig.from_learner(Old()).out_of_bounds_penalty==0
    for v in [-1.,math.nan,math.inf,True]:
        with pytest.raises(ValueError):RewardAdjustmentConfig(v)


def test_real_environment_explicit_boundary_and_truncation():
    import pybullet as p
    from config import EnvConfig
    from env import MultiDroneEnv
    cfg=EnvConfig(gui=False,scenario='open_field',num_drones=2,max_episode_steps=2)
    cfg.sensors.camera_enabled=False
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        # Test setup only; normal dynamics and environment sources are unchanged.
        body=env.drones['uav_0'].body_id
        p.resetBasePositionAndOrientation(body,[env.town.side/2+cfg.world_margin+2,0,5],[0,0,0,1],physicsClientId=env.client)
        _,rewards,term,trunc,infos=env.step({a:np.zeros(3) for a in env.agents})
        assert infos['uav_0']['status']=='out_of_bounds' and term['uav_0']
        adjusted,penalty=EpisodeRewardAdjustment(RewardAdjustmentConfig(20)).apply(rewards,term,trunc,infos)
        assert penalty['uav_0']==-20 and penalty['uav_1']==0
        assert adjusted['uav_0']==rewards['uav_0']-20
        _,rewards,term,trunc,infos=env.step({a:np.zeros(3) for a in env.agents})
        assert trunc['uav_1'] and not term['uav_1']
        assert EpisodeRewardAdjustment(RewardAdjustmentConfig(20)).apply(rewards,term,trunc,infos)[1]['uav_1']==0


def test_enabled_training_checkpoint_resume_and_evaluation(tmp_path):
    import csv,json,torch
    from train import build_parser,run
    from algorithms.registry import load_algorithm
    from algorithms.evaluation import evaluate,checkpoint_environment
    args=build_parser().parse_args(['--algo','maddpg','--total-steps','2','--episode-steps','2',
        '--hidden-layers','8','--warmup-steps','2','--batch-size','2','--out-of-bounds-penalty','20',
        '--validation-interval','100','--checkpoint-interval','100','--output',str(tmp_path/'first')])
    summary=run(args)
    learner=load_algorithm('maddpg',summary['final_checkpoint'],'cpu')
    assert RewardAdjustmentConfig.from_learner(learner).out_of_bounds_penalty==20
    row=next(csv.DictReader(next((tmp_path/'first').rglob('*_training.csv')).open()))
    assert float(row['raw_team_reward'])==float(row['team_reward'])
    assert json.loads(row['out_of_bounds_penalties'])=={'uav_0':0.,'uav_1':0.}
    before={k:v.clone() for k,v in learner.networks.state_dict().items()}
    result=evaluate(learner,checkpoint_environment(learner),seeds=[100042])
    assert result['reward_adjustment']['out_of_bounds_penalty']==20
    assert result['mean_raw_reward']==result['mean_reward']
    assert all(torch.equal(v,before[k]) for k,v in learner.networks.state_dict().items())
    args.resume=summary['final_checkpoint'];args.output=tmp_path/'resumed';args.total_steps=3
    args.out_of_bounds_penalty=None
    resumed=run(args)
    assert RewardAdjustmentConfig.from_learner(load_algorithm('maddpg',resumed['final_checkpoint'],'cpu')).out_of_bounds_penalty==20
    args.out_of_bounds_penalty=0
    with pytest.raises(ValueError,match='adjustment mismatch'):run(args)
