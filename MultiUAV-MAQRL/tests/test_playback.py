import json
import pytest
import torch
pytest.importorskip('pybullet')
from test import build_parser,run
from config.maddpg_config import MADDPGConfig
from algorithms.maddpg.maddpg import MADDPG


def test_checkpoint_headless_playback(tmp_path):
    learner=MADDPG(['uav_0','uav_1'],19,MADDPGConfig(hidden_layers=(8,)))
    checkpoint=tmp_path/'model.pt';learner.save_checkpoint(checkpoint)
    before=checkpoint.read_bytes()
    args=build_parser().parse_args(['--algo','maddpg','--checkpoint',str(checkpoint),'--headless',
        '--scenario','open_field','--num-drones','2','--seed','42','--episodes','2',
        '--episode-steps','3','--output',str(tmp_path/'test_metrics')])
    result=run(args)
    assert result['episodes']==2 and len(result['episode_metrics'])==2
    assert all(r['steps']==3 for r in result['episode_metrics'])
    assert all(set(r['per_agent_status'])=={'uav_0','uav_1'} for r in result['episode_metrics'])
    assert checkpoint.read_bytes()==before
    assert json.loads(next((tmp_path/'test_metrics').rglob('*_test.json')).read_text())['episodes']==2
    assert list((tmp_path/'test_metrics').rglob('*_test.csv'))
    assert run(args)==result
    gui=build_parser().parse_args(['--algo','maddpg','--checkpoint',str(checkpoint),'--render','--realtime'])
    assert gui.render and gui.realtime
