"""Final noise-free historical/new policy comparison at a shared 80s limit."""
from pathlib import Path
import copy,gc,json
from collections import Counter
import numpy as np
from algorithms.registry import load_algorithm
from algorithms.evaluation import checkpoint_environment
from algorithms.env_adapter import ParallelEnvAdapter
from env import MultiDroneEnv
from utils.training_telemetry import EpisodeTelemetry


def main():
    parent=Path('outputs/experiments/maddpg/urban/n2');out=Path('outputs/benchmarks/horizon_1600_gamma_095')
    baseline=parent/'maddpg_urban_n2_seed42'
    newer=[]
    for p in parent.glob('*/*_metadata.json'):
        m=json.loads(p.read_text())
        if m.get('kind')=='training' and m['environment']['max_episode_steps']==1600 and m['training_budget']==186374:newer.append(p.parent)
    if len(newer)!=1:raise ValueError('Expected one completed horizon experiment')
    results=[]
    for label,root in [('historical',baseline),('horizon_1600',newer[0])]:
        checkpoint=next((root/'checkpoints').glob('*_final_checkpoint.pt'))
        learner=load_algorithm('maddpg',checkpoint,'cpu');cfg=checkpoint_environment(learner)
        cfg.gui=False;cfg.realtime=False;cfg.max_episode_steps=1600
        with MultiDroneEnv(cfg) as env:
            adapter=ParallelEnvAdapter(env)
            for seed in range(100042,100052):
                adapter.reset(seed=seed);state=env.global_state()
                t=EpisodeTelemetry(adapter.agent_ids,{a:float(np.linalg.norm(np.asarray(state['uavs'][a]['position'])-state['destination'])) for a in adapter.agent_ids})
                reward=0.;steps=0
                while env.agents:
                    actions=learner.select_actions(adapter.active_observations(),training=False)
                    if steps%20==0:t.sample_q(learner,adapter,actions)
                    _,r,_,_,infos=adapter.step(actions);t.observe(actions,infos);steps+=1;reward+=sum(r.values())
                row=dict(policy=label,seed=seed,steps=steps,reward=reward,team_success=env.episode_statistics['team_success'],**t.result())
                results.append(row);print(label,seed,row['agent_status'],flush=True)
        del learner;gc.collect()
    (out/'final_policy_comparison.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
