import json
import pytest
import torch
pytest.importorskip('pybullet')
from train import build_parser,run
from algorithms.maddpg.maddpg import MADDPG


def args(output):
    return build_parser().parse_args(['--algo','maddpg','--total-steps','6','--episode-steps','3',
        '--warmup-steps','2','--batch-size','2','--hidden-layers','8','--checkpoint-interval','3',
        '--validation-interval','3','--evaluation-episodes','1','--output',str(output)])


def test_training_updates_checkpoints_validation_and_resume(tmp_path):
    a=run(args(tmp_path/'a'))
    b=run(args(tmp_path/'b'))
    assert a['environment_steps']==6 and a['agent_transitions']==12 and a['updates']==5
    assert a['episodes_completed']==2
    assert len(list((tmp_path/'a').rglob('*_training.csv')))==1
    assert len(list((tmp_path/'a').rglob('*_network_loss_plot.png')))==1
    assert not list((tmp_path/'a').rglob('*.jsonl'))
    assert list((tmp_path/'a').rglob('*_step3_checkpoint.pt'))
    assert len(list((tmp_path/'a').rglob('*_validation.csv')))==1
    first=MADDPG.load_checkpoint(a['final_checkpoint'])
    second=MADDPG.load_checkpoint(b['final_checkpoint'])
    assert all(torch.equal(a,b) for a,b in zip(first.networks.parameters(),second.networks.parameters()))
    resumed=args(tmp_path/'resumed');resumed.resume=a['final_checkpoint'];resumed.total_steps=8
    result=run(resumed)
    assert result['environment_steps']==8 and result['updates']==7 and result['replay_size']==8
    assert result['agent_transitions']==16
    final=MADDPG.load_checkpoint(result['final_checkpoint'])
    assert final.training_state['counters']['episodes_started']==3
    assert final.config.hidden_layers==(8,)


def test_episode_budget_and_density_flags(tmp_path):
    parsed=args(tmp_path/'episode_budget')
    parsed.episodes=1
    parsed.building_density=.65
    result=run(parsed)
    assert result['episodes_completed']==1 and result['environment_steps']==3
    model=MADDPG.load_checkpoint(result['final_checkpoint'])
    assert model.training_state['environment']['town']['building_density']==.65
    parsed=build_parser().parse_args(['--algo','maddpg','--scenario','urban',
        '--building-density','.65','--episodes','1000'])
    assert parsed.episodes==1000 and parsed.building_density==.65 and parsed.total_steps is None
