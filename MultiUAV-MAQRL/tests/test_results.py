import json
import re
import pytest
from utils.results import Experiment,seed_set_id,checkpoint_provenance


def test_names_directories_metadata_and_uniqueness(tmp_path):
    metadata=dict(algorithm_config={'gamma':.95},training_budget=20000)
    a=Experiment.create(tmp_path,'MADDPG','open_field',2,42,metadata)
    b=Experiment.create(tmp_path,'MADDPG','open_field',2,42,metadata)
    assert a.directory!=b.directory
    assert a.directory.name=='maddpg_openfield_n2_seed42'
    assert b.directory.name.startswith('maddpg_openfield_n2_seed42_run')
    assert {p.name for p in a.directory.iterdir() if p.is_dir()}=={'checkpoints','training','validation','testing','plots'}
    assert a.path('checkpoint','pt',3).parent.name=='checkpoints'
    assert a.path('training','csv',3).parent.name=='training'
    assert a.path('validation','json',2).parent.name=='validation'
    assert a.path('test','json',2).parent.name=='testing'
    assert a.path('reward_plot','svg',3).parent.name=='plots'
    path=a.path('training','csv',1000,steps=20000,seeds=[100,101])
    assert path.name.startswith('maddpg_openfield_n2_ep1000_seed42_')
    assert path.name.endswith('_training.csv') and '_step20000_' in path.name
    assert re.fullmatch('[a-z0-9_]+[.]csv',path.name)
    assert a.directory.parent==tmp_path/'experiments'/'maddpg'/'openfield'/'n2'
    data=json.loads(next(a.directory.glob('*_metadata.json')).read_text())
    assert data['reward_version'].startswith('sha256_') and data['training_budget']==20000
    assert seed_set_id([100,101])==seed_set_id([100,101])
    assert seed_set_id([100,101])!=seed_set_id([101,100])
    with pytest.raises(ValueError):a.path('test','json',-1)
    with pytest.raises(ValueError):Experiment.create(tmp_path,'maddpg','urban',0,42,{})


def test_checkpoint_handling(tmp_path):
    checkpoint=tmp_path/'old final.pt';checkpoint.write_bytes(b'unchanged')
    class Learner:
        environment_steps=20
        training_state={'experiment':{'run_id':'origin'}}
    metadata=checkpoint_provenance(checkpoint,Learner())
    assert metadata['origin_run']=='origin' and metadata['checkpoint_step']==20
    assert metadata['checkpoint_path']==str(checkpoint.resolve())
    assert checkpoint.read_bytes()==b'unchanged'


def test_evaluations_stay_with_origin_run_without_overwriting(tmp_path):
    from types import SimpleNamespace
    from config import EnvConfig
    from config.maddpg_config import MADDPGConfig
    from utils.results import export_evaluation
    original=Experiment.create(tmp_path,'maddpg','open_field',2,42,{'training_budget':20})
    checkpoint=original.path('checkpoint','pt',3,steps=20)
    checkpoint.write_bytes(b'preserved checkpoint')
    learner=SimpleNamespace(config=MADDPGConfig(),agent_ids=('uav_0','uav_1'),environment_steps=20,
        training_state={'experiment':{'run_id':original.run_id,'directory':str(original.directory),'training_budget':20}})
    result={'episodes':1,'validation_seeds':[100],
            'episode_metrics':[{'reward':1.,'team_success':False,'collisions':0}]}
    first=export_evaluation(result,tmp_path,'maddpg',learner,EnvConfig(scenario='open_field',num_drones=2),checkpoint)
    existing={p:p.read_bytes() for p in original.directory.rglob('*') if p.is_file()}
    second=export_evaluation(result,tmp_path,'maddpg',learner,EnvConfig(scenario='open_field',num_drones=2),checkpoint)
    assert first.directory==second.directory==original.directory
    assert len(list((original.directory/'validation').glob('*_validation.json')))==2
    assert len(list((original.directory/'plots').glob('*.png')))==6
    assert all(p.read_bytes()==content for p,content in existing.items())
    meta=json.loads(next((original.directory/'validation').glob('*_validation_metadata.json')).read_text())
    assert meta['origin_run']==original.run_id and meta['evaluation_episodes']==1
    assert meta['checkpoint_path']==str(checkpoint.resolve()) and meta['training_budget']==20


def test_high_dpi_png_plot(tmp_path):
    from utils.results import write_plot
    from PIL import Image
    path=tmp_path/'team_reward.png'
    write_plot(path,[1.,2.,3.],'reward_plot')
    with Image.open(path) as image:
        assert image.format=='PNG'
        assert image.width>2000 and image.height>1000
        assert image.info['dpi'][0]>=299


def test_loss_plot_averages_agents_equally():
    from utils.results import mean_agent_loss_series
    episodes,series=mean_agent_loss_series([
        dict(episode=1,agent='a',actor_loss=2,critic_loss=4),
        dict(episode=1,agent='a',actor_loss=4,critic_loss=8),
        dict(episode=1,agent='b',actor_loss=9,critic_loss=10)])
    assert episodes==[1] and len(series)==2
    assert series['Actor loss (agent mean)']==[6.]
    assert series['Critic loss (agent mean)']==[8.]
