"""Paired deterministic horizon-control/boundary-penalty final evaluation."""
from pathlib import Path
from collections import Counter
import copy,csv,gc,json
import numpy as np
from algorithms.registry import load_algorithm
from algorithms.evaluation import evaluate,checkpoint_environment
from algorithms.reward_adjustments import RewardAdjustmentConfig
from utils.results import export_evaluation

OUT=Path('outputs/benchmarks/boundary_penalty_20_gamma_095')


def summarize(result):
    rows=result['episode_metrics'];n=len(rows)*2
    statuses=Counter(s['status'] for r in rows for s in r['per_agent_status'].values())
    # Equal-agent, equal-episode average. Each saturation value uses that UAV's active steps.
    return dict(episodes=len(rows),agent_outcomes=n,status_counts=dict(statuses),
        boundary_failure_rate=statuses['out_of_bounds']/n,individual_arrival_rate=result['individual_arrival_rate'],
        team_success_rate=result['team_success_rate'],collisions=result['collision_count'],
        mean_steps=float(np.mean([r['steps'] for r in rows])),mean_raw_reward=result['mean_raw_reward'],mean_learning_reward=result['mean_reward'],
        mean_destination_progress=float(np.mean([v for r in rows for v in r['destination_progress'].values()])),
        per_agent_destination_progress={a:float(np.mean([r['destination_progress'][a] for r in rows])) for a in ['uav_0','uav_1']},
        action_mean=np.mean([v for r in rows for v in r['action_mean'].values()],axis=0).tolist(),
        action_saturation=np.mean([v for r in rows for v in r['action_saturation'].values()],axis=0).tolist(),
        negative_x_saturation=float(np.mean([v[0] for r in rows for v in r['negative_action_saturation'].values()])))


def main():
    manifests={name:json.loads(Path(path).read_text()) for name,path in [
        ('horizon_only','experiments/maddpg/horizon_1600_gamma_095.json'),
        ('boundary_penalty_20','experiments/maddpg/boundary_penalty_20_gamma_095.json')]}
    results={};training={};configs={}
    for name,manifest in manifests.items():
        root=Path(manifest['run_directory'])
        checkpoint=next((root/'checkpoints').glob('*_final_checkpoint.pt'))
        learner=load_algorithm('maddpg',checkpoint,'cpu');cfg=checkpoint_environment(learner)
        from dataclasses import asdict
        configs[name]=dict(algorithm=asdict(learner.config),environment=asdict(cfg))
        expected=0 if name=='horizon_only' else 20
        assert RewardAdjustmentConfig.from_learner(learner).out_of_bounds_penalty==expected
        cfg.gui=False;cfg.realtime=False;cfg.max_episode_steps=1600
        params={k:v.clone() for k,v in learner.networks.state_dict().items()}
        result=evaluate(learner,cfg,seeds=list(range(100042,100052)))
        import torch
        assert all(torch.equal(params[k],v) for k,v in learner.networks.state_dict().items())
        paths=export_evaluation(result,Path('outputs'),'maddpg',learner,cfg,checkpoint,'validation')
        (OUT/(name+'_evaluation.json')).write_text(json.dumps(result,indent=2)+'\n')
        results[name]=summarize(result)
        with next((root/'training').glob('*_training.csv')).open() as f:rows=list(csv.DictReader(f))
        statuses=Counter(s for r in rows for s in json.loads(r['agent_status']).values())
        training[name]=dict(episode_rows=len(rows),complete_episodes=sum(r['completed']=='True' for r in rows),
            status_counts=dict(statuses),arrivals=sum(int(r['successes']) for r in rows),collisions=sum(int(r['collisions']) for r in rows),
            last100_progress=float(np.mean([v for r in rows[-100:] for v in json.loads(r['destination_progress']).values()])),
            last100_action_mean=np.mean([v for r in rows[-100:] for v in json.loads(r['action_mean']).values()],axis=0).tolist(),
            last100_action_saturation=np.mean([v for r in rows[-100:] for v in json.loads(r['action_saturation']).values()],axis=0).tolist(),
            mean_episode_steps=float(np.mean([int(r['steps']) for r in rows])))
        print(name,json.dumps(results[name]),flush=True)
        del learner,params;gc.collect()
    assert configs['horizon_only']==configs['boundary_penalty_20'], 'Uncontrolled configuration difference'
    (OUT/'matched_configuration.json').write_text(json.dumps(configs,indent=2)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(dict(validation=results,training=training),indent=2)+'\n')

if __name__=='__main__':main()
