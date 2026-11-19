"""Read-only model diagnostics and short matched-seed evaluations, no training."""
import csv,json,copy
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from algorithms.registry import load_algorithm
from algorithms.evaluation import checkpoint_environment
from algorithms.env_adapter import ParallelEnvAdapter
from env import MultiDroneEnv
from main import demo_policy


def main():
 root=Path('outputs/experiments/maddpg/urban/n2/maddpg_urban_n2_seed42')
 out=Path('outputs/benchmarks/maddpg_1500_diagnosis');out.mkdir(parents=True,exist_ok=True)
 checkpoint=next((root/'checkpoints').glob('*_final_checkpoint.pt'))
 learner=load_algorithm('maddpg',checkpoint,'cpu')
 cfg=checkpoint_environment(learner);cfg.gui=False;cfg.realtime=False
 training=list(csv.DictReader(next((root/'training').glob('*_training.csv')).open()))
 stats=dict(episodes=len(training),successes=sum(int(r['successes']) for r in training),
  team_successes=sum(r['team_success']=='True' for r in training),collisions=sum(int(r['collisions']) for r in training),
  lengths={str(q):float(np.quantile([int(r['steps']) for r in training],q)) for q in [0,.25,.5,.75,.9,1]},
  agent_collisions=dict(Counter(a for r in training for a in json.loads(r['collision_agents']))))
 loss=defaultdict(list)
 with next((root/'training').glob('*_actor_critic_loss.csv')).open() as stream:
  for r in csv.DictReader(stream):
   for term in ['actor_loss','critic_loss']:loss[(r['agent'],term)].append(float(r[term]))
 stats['loss']={str(k):dict(first_mean=float(np.mean(v[:1000])),last_mean=float(np.mean(v[-1000:])),min=min(v),max=max(v),finite=bool(np.isfinite(v).all())) for k,v in loss.items()}
 batch=learner.replay.sample(512)
 with torch.no_grad():
  targets=learner.targets(batch)
  stats['q']={a:dict(mean=float(c(batch['global_state'],batch['actions'].flatten(1)).mean()),std=float(c(batch['global_state'],batch['actions'].flatten(1)).std()),target_mean=float(targets[:,i].mean())) for i,(a,c) in enumerate(zip(learner.agent_ids,learner.networks.critics))}
 stats['replay']={k:list(v.shape) for k,v in batch.items()}
 stats['terminal_rate']=float(batch['terminated'].float().mean());stats['truncation_rate']=float(batch['truncated'].float().mean())
 traces=[];episodes=[]
 for seed in [100000,100001,100002]:
  for policy in ['trained','random','rule']:
   with MultiDroneEnv(copy.deepcopy(cfg)) as env:
    adapter=ParallelEnvAdapter(env);obs,_=adapter.reset(seed=seed);rng=np.random.default_rng(seed)
    initial={a:d.state()[0] for a,d in env.drones.items()};best={a:float(np.linalg.norm(env.destination.center-p)) for a,p in initial.items()}
    cumulative={a:defaultdict(float) for a in initial};kinds={};status={};final={};steps=0
    for a,p in initial.items():
     inner=env.destination.half_size-cfg.drone.radius
     lower=float(np.linalg.norm(np.maximum(np.abs(p-env.destination.center)-inner,0)))
     traces.append(dict(policy=policy,seed=seed,step=0,agent=a,distance=best[a],position=p.tolist(),min_travel_seconds=lower/cfg.drone.max_speed_xy))
    while env.agents:
     active=list(env.agents)
     if policy=='trained':actions=learner.select_actions(adapter.active_observations(),training=False)
     elif policy=='random':actions={a:rng.uniform(-1,1,3).astype(np.float32) for a in active}
     else:actions=demo_policy(obs,env)
     # Capture body categories before retirement removes/replaces collision bodies.
     ids={d.body_id for d in env.drones.values()};static=set(env.town.static_body_ids);moving={x['body'] for x in env.moving_obstacles.items}
     obs,rewards,term,trunc,infos=adapter.step(actions);steps+=1
     for a in active:
      state=env.drones[a].state();distance=infos[a]['distance_to_goal'];best[a]=min(best[a],distance);final[a]=distance;status[a]=infos[a]['status']
      for key,val in infos[a]['reward_components'].items():cumulative[a][key]+=val
      if infos[a]['collision']:
       hits=env.drones[a].collision_bodies
       kinds[a]=('uav' if hits&ids else 'static' if hits&static else 'moving' if hits&moving else 'unclassified')
      traces.append(dict(policy=policy,seed=seed,step=steps,agent=a,distance=distance,position=state[0].tolist(),velocity=state[1].tolist(),action=actions[a].tolist(),reward=rewards[a],components=infos[a]['reward_components']))
    episodes.append(dict(policy=policy,seed=seed,steps=steps,status=status,collision_kind=kinds,initial_distances={a:float(np.linalg.norm(env.destination.center-p)) for a,p in initial.items()},minimum_distances=best,final_distances=final,reward_components={a:dict(v) for a,v in cumulative.items()},team_success=env.episode_statistics['team_success']))
 stats['configuration']=dict(algorithm=vars(learner.config),environment=__import__('dataclasses').asdict(cfg))
 stats['matched_evaluations']=episodes
 for policy in ['trained','random','rule']:
  rows=[r for r in traces if r['policy']==policy and r['step']>0]
  actions=np.array([r['action'] for r in rows]);velocities=np.array([r['velocity'] for r in rows])
  stats[policy+'_action_velocity']=dict(action_mean=actions.mean(0).tolist(),action_std=actions.std(0).tolist(),saturation_fraction=(np.abs(actions)>.98).mean(0).tolist(),velocity_mean=velocities.mean(0).tolist())
 (out/'diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
 with (out/'traces.jsonl').open('w') as stream:
  for row in traces:stream.write(json.dumps(row)+'\n')
 print(json.dumps({k:v for k,v in stats.items() if k not in ['configuration','replay']},indent=2))

if __name__=='__main__':main()
