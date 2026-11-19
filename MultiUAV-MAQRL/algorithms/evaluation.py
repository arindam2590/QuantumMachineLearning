"""Deterministic complete-episode validation; no learning or replay mutations."""
from contextlib import contextmanager
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from env import MultiDroneEnv
from .env_adapter import ParallelEnvAdapter
from .reward_adjustments import RewardAdjustmentConfig,EpisodeRewardAdjustment
from utils.training_telemetry import EpisodeTelemetry


@contextmanager
def deterministic_policy(learner):
    networks=getattr(learner,'networks',None)
    modes=[] if networks is None else [(module,module.training) for module in networks.modules()]
    try:
        if networks is not None:networks.eval()
        with torch.no_grad():yield
    finally:
        for module,mode in modes:module.training=mode


def aggregate(rows):
    if not rows: raise ValueError('No episodes to aggregate')
    arrivals=sum(r['arrivals'] for r in rows)
    times=[v for r in rows for v in r['arrival_times'].values()]
    efficiencies=[v for r in rows for v in r['successful_path_efficiency'].values()]
    return dict(episodes=len(rows),team_success_rate=float(np.mean([r['team_success'] for r in rows])),
        individual_arrival_rate=arrivals/sum(r['fleet'] for r in rows),
        collision_count=sum(r['collisions'] for r in rows),
        mean_travel_time=float(np.mean(times)) if times else None,
        mean_reward=float(np.mean([r['reward'] for r in rows])),
        mean_energy_use=float(np.mean([r['energy_use'] for r in rows])),
        mean_path_efficiency=float(np.mean(efficiencies)) if efficiencies else None,
        mean_episode_duration=float(np.mean([r['duration'] for r in rows])))


def save_metrics(result,output):
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    rows=result['episode_metrics']
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in row.items()} for row in rows)
    path.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def evaluate(learner,environment_config,episodes=1,seed=100000,*,seeds=None,output=None,training_seeds=None,
             enforce_heldout=True,episode_callback=None,keep_open=False,result_writer=None,reward_adjustment=None):
    if isinstance(episodes,bool) or not isinstance(episodes,int) or episodes<1:
        raise ValueError('episodes must be positive')
    seeds=list(range(seed,seed+episodes)) if seeds is None else list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(isinstance(s,bool) or not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError('Require distinct nonnegative validation seeds')
    forbidden=set(training_seeds or [learner.config.seed])
    state=getattr(learner,'training_state',None)
    if state:
        count=state['counters']['episodes_started']
        forbidden.update(range(learner.config.seed,learner.config.seed+count))
    if enforce_heldout and forbidden.intersection(seeds):raise ValueError('Validation seeds overlap known training seeds')
    adjustment_config=reward_adjustment if reward_adjustment is not None else RewardAdjustmentConfig.from_learner(learner)
    adjustment=EpisodeRewardAdjustment(adjustment_config)
    rows=[]
    with deterministic_policy(learner), MultiDroneEnv(copy.deepcopy(environment_config)) as env:
        adapter=ParallelEnvAdapter(env)
        if adapter.agent_ids!=learner.agent_ids or adapter.global_state_dim!=learner.global_state_dim or adapter.observation_dim!=learner.observation_dim:
            raise ValueError('Checkpoint and environment dimensions/order are incompatible')
        for validation_seed in seeds:
            adapter.reset(seed=validation_seed)
            adjustment.reset()
            initial=env.global_state()['uavs']
            destination=env.global_state()['destination']
            telemetry=EpisodeTelemetry(adapter.agent_ids,{a:float(np.linalg.norm(np.asarray(initial[a]['position'])-destination)) for a in adapter.agent_ids})
            raw_rewards_sum={a:0. for a in adapter.agent_ids};penalties_sum={a:0. for a in adapter.agent_ids}
            previous={a:np.array(initial[a]['position']) for a in adapter.agent_ids}
            distances={a:0. for a in adapter.agent_ids};arrivals={};efficiency={};collisions=set()
            rewards_sum={a:0. for a in adapter.agent_ids};energies={a:0. for a in adapter.agent_ids}
            statuses={};steps=0
            while env.agents:
                active=list(env.agents)
                actions=learner.select_actions(adapter.active_observations(),training=False)
                if steps%20==0:telemetry.sample_q(learner,adapter,actions)
                _,raw_rewards,terminated,truncated,infos=adapter.step(actions)
                rewards,penalties=adjustment.apply(raw_rewards,terminated,truncated,infos)
                telemetry.observe(actions,infos)
                steps+=1;current=env.global_state()['uavs']
                for a in active:
                    position=np.asarray(current[a]['position'])
                    distances[a]+=float(np.linalg.norm(position-previous[a]));previous[a]=position.copy()
                    raw_rewards_sum[a]+=raw_rewards[a];penalties_sum[a]+=penalties[a]
                    rewards_sum[a]+=rewards[a];energies[a]+=infos[a]['energy_consumed']
                    statuses[a]=dict(status=infos[a]['status'],arrival=bool(infos[a]['success']),collision=bool(infos[a]['collision']))
                    if infos[a]['collision']:collisions.add(a)
                    if infos[a]['success'] and a not in arrivals:
                        arrivals[a]=steps*environment_config.control_dt
                        half=np.maximum(env.destination.half_size-environment_config.drone.radius,0)
                        direct=float(np.linalg.norm(np.maximum(np.abs(np.asarray(initial[a]['position'])-env.destination.center)-half,0)))
                        efficiency[a]=min(1.,direct/max(distances[a],1e-12))
            rows.append(dict(seed=validation_seed,scenario=environment_config.scenario,fleet=len(adapter.agent_ids),
                arrivals=len(arrivals),team_success=len(arrivals)==len(adapter.agent_ids),collisions=len(collisions),
                duration=steps*environment_config.control_dt,steps=steps,reward=sum(rewards_sum.values()),
                energy_use=sum(energies.values()),per_agent_rewards=rewards_sum,travel_distances=distances,
                arrival_times=arrivals,successful_path_efficiency=efficiency,per_agent_energy=energies,
                per_agent_status=statuses,raw_reward=sum(raw_rewards_sum.values()),raw_per_agent_rewards=raw_rewards_sum,
                out_of_bounds_penalties=penalties_sum,**telemetry.result()))
            if episode_callback is not None:episode_callback(rows[-1])
        # Persist metrics before optionally keeping the final GUI scene open.
        result=dict(**aggregate(rows),validation_seeds=seeds,episode_metrics=rows,
                    mean_raw_reward=float(np.mean([r['raw_reward'] for r in rows])),reward_adjustment=adjustment_config.metadata())
        if output is not None:save_metrics(result,output)
        if result_writer is not None:result_writer(result)
        if keep_open and environment_config.gui:
            import pybullet as p
            import time
            print('Testing complete. Close the PyBullet window or press Ctrl+C to exit.',flush=True)
            try:
                while p.isConnected(physicsClientId=env.client):
                    p.getKeyboardEvents(physicsClientId=env.client)
                    time.sleep(.05)
            except KeyboardInterrupt:
                pass
            except p.error:
                if p.isConnected(physicsClientId=env.client):raise
    return result


def checkpoint_environment(learner):
    """Rebuild public configuration instance from checkpoint metadata."""
    from config import EnvConfig
    cfg=EnvConfig(gui=False,num_drones=len(learner.agent_ids),scenario='open_field')
    metadata=getattr(learner,'training_state',None)
    if metadata:
        def populate(instance,values):
            for key,value in values.items():
                if isinstance(value,dict):populate(getattr(instance,key),value)
                else:setattr(instance,key,value)
        populate(cfg,metadata['environment'])
    return cfg


def validation_main(argv=None):
    import argparse
    from config import EnvConfig
    from .registry import available_algorithms,load_algorithm
    parser=argparse.ArgumentParser(description='Deterministic checkpoint validation')
    parser.add_argument('--algo',required=True,choices=available_algorithms())
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--scenario',choices=['urban','open_field','cluttered'])
    parser.add_argument('--num-drones',type=int)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda','mps','auto'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[100000,100001,100002,100003,100004])
    parser.add_argument('--episode-steps',type=int)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    args=parser.parse_args(argv)
    learner=load_algorithm(args.algo,args.checkpoint,args.device)
    cfg=checkpoint_environment(learner)
    cfg.gui=False;cfg.realtime=False
    if args.scenario is not None:cfg.scenario=args.scenario
    if args.num_drones is not None:cfg.num_drones=args.num_drones
    if args.episode_steps is not None:cfg.max_episode_steps=args.episode_steps
    cfg.validate()
    from utils.results import export_evaluation
    result=evaluate(learner,cfg,seeds=args.seeds,result_writer=lambda result:
        export_evaluation(result,args.output,args.algo,learner,cfg,args.checkpoint,'validation'))
    print(json.dumps({k:v for k,v in result.items() if k!='episode_metrics'},indent=2))
    return result
