"""General algorithm-selectable training entry point (Pure MADDPG initially)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from config import EnvConfig
from config.maddpg_config import MADDPGConfig
from env import MultiDroneEnv
from algorithms.env_adapter import ParallelEnvAdapter
from algorithms.registry import available_algorithms,create_algorithm,load_algorithm
from algorithms.evaluation import evaluate
from algorithms.reward_adjustments import RewardAdjustmentConfig,EpisodeRewardAdjustment
from utils.training_telemetry import EpisodeTelemetry
from utils.results import Experiment,write_rows,checkpoint_provenance,export_evaluation,write_plot,mean_agent_loss_series


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algo',required=True,choices=available_algorithms())
    parser.add_argument('--scenario',choices=['open_field','urban','cluttered'],default='open_field')
    parser.add_argument('--num-drones',type=int,default=2)
    parser.add_argument('--total-steps',type=int,default=None,help='absolute step cap; default 1000, or episodes*episode-steps when --episodes is given')
    parser.add_argument('--episodes',type=int,default=None,help='absolute completed-episode cap, including resumed episodes')
    parser.add_argument('--building-density',type=float,default=.55,help='urban eligible-plot occupancy in [0,1]')
    parser.add_argument('--episode-steps',type=int,default=600)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--telemetry-q-interval',type=int,default=20,help='Q sample interval in environment steps; 0 disables')
    parser.add_argument('--out-of-bounds-penalty',type=float,default=None,help='optional learning-reward penalty magnitude; fresh default 0, resume restores checkpoint')
    parser.add_argument('--resume',type=Path)
    defaults=MADDPGConfig()
    for name,value in asdict(defaults).items():
        flag='--'+name.replace('_','-')
        if name=='hidden_layers':parser.add_argument(flag,type=int,nargs='+',default=value)
        elif name=='gradient_clip_norm':parser.add_argument(flag,type=float,default=value)
        else:parser.add_argument(flag,type=type(value),default=value)
    return parser


def run(args):
    if args.episode_steps<1 or (args.total_steps is not None and args.total_steps<1):raise ValueError('Step budgets must be positive')
    if args.telemetry_q_interval<0:raise ValueError('--telemetry-q-interval must be nonnegative')
    if args.episodes is not None and args.episodes<1:raise ValueError('--episodes must be positive')
    adjustment_config=RewardAdjustmentConfig(0. if args.out_of_bounds_penalty is None else args.out_of_bounds_penalty)
    values={name:getattr(args,name) for name in asdict(MADDPGConfig())}
    values['hidden_layers']=tuple(values['hidden_layers'])
    config=MADDPGConfig(**values);config.validate()
    if args.total_steps is None:
        args.total_steps=(1000 if args.episodes is None else
                          min(config.max_training_steps,args.episodes*args.episode_steps))
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    env_config=EnvConfig(gui=False,scenario=args.scenario,num_drones=args.num_drones,max_episode_steps=args.episode_steps)
    env_config.town.building_density=args.building_density
    env_config.validate()
    # Explicit initial baseline; no frozen configuration definitions are changed.
    env_config.drone.velocity_mode='pd';env_config.sensors.camera_enabled=False;env_config.obstacles.count=0
    args.output.mkdir(parents=True,exist_ok=True)
    counters=dict(agent_transitions=0,episodes_started=0,episodes_completed=0,wall_seconds=0.)
    started=time.perf_counter()
    with MultiDroneEnv(env_config) as env:
        adapter=ParallelEnvAdapter(env)
        adapter.reset(seed=config.seed)
        if args.resume:
            learner=load_algorithm(args.algo,args.resume,device=config.device)
            if learner.agent_ids!=adapter.agent_ids or learner.global_state_dim!=adapter.global_state_dim:
                raise ValueError('Checkpoint/environment dimensions do not match')
            state=learner.training_state
            if state is None:raise ValueError('Checkpoint lacks trainer resume metadata')
            if state['environment']!=asdict(env_config):raise ValueError('Resume environment configuration mismatch')
            saved_adjustment=RewardAdjustmentConfig.from_learner(learner)
            if args.out_of_bounds_penalty is not None and adjustment_config!=saved_adjustment:
                raise ValueError('Resume reward adjustment mismatch')
            adjustment_config=saved_adjustment
            config=learner.config
            counters.update(state['counters'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
            if state.get('cuda_rng') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:learner=create_algorithm(args.algo,adapter=adapter,config=config)
        if args.total_steps>config.max_training_steps:
            raise ValueError('--total-steps exceeds configured max_training_steps')
        experiment=Experiment.create(args.output,args.algo,env_config.scenario,args.num_drones,config.seed,
            dict(kind='training',algorithm_config=asdict(config),environment=asdict(env_config),reward_adjustment=adjustment_config.metadata(),
                 training_budget=args.total_steps,episode_budget=args.episodes,telemetry_q_interval=args.telemetry_q_interval,training_seed_rule='seed + episode index',
                 parent={} if not args.resume else checkpoint_provenance(args.resume,learner)))
        adjustment=EpisodeRewardAdjustment(adjustment_config)
        episode_rows=[];update_rows=[];validation_rows=[]
        checkpoint_paths=[]
        base_wall=counters['wall_seconds']
        def save(name):
            counters['wall_seconds']=base_wall+time.perf_counter()-started
            state=dict(experiment=dict(run_id=experiment.run_id,directory=str(experiment.directory),training_budget=args.total_steps),counters=dict(counters),reward_adjustment=adjustment_config.metadata(),environment=asdict(env_config),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                resume_semantics='new episode; PyBullet mid-episode state is not restored')
            episodes=counters['episodes_completed']+int(name!='final.pt' and not env.agents)
            path=experiment.path('final_checkpoint' if name=='final.pt' else 'checkpoint','pt',episodes,steps=learner.environment_steps)
            learner.save_checkpoint(path,training_state=state)
            checkpoint_paths.append(str(path))
            return path
        def log(name,row):
            kind={'episodes.jsonl':'training','updates.jsonl':'loss','validation.jsonl':'validation'}[name]
            {'training':episode_rows,'loss':update_rows,'validation':validation_rows}[kind].append(row)
        try:
            while learner.environment_steps<args.total_steps and (args.episodes is None or counters['episodes_completed']<args.episodes):
                episode=counters['episodes_started'];counters['episodes_started']+=1
                adapter.reset(seed=config.seed+episode)
                adjustment.reset()
                states=env.global_state()
                telemetry=EpisodeTelemetry(adapter.agent_ids,{a:float(np.linalg.norm(np.asarray(states['uavs'][a]['position'])-states['destination'])) for a in adapter.agent_ids})
                raw_rewards_sum={a:0. for a in adapter.agent_ids};penalties_sum={a:0. for a in adapter.agent_ids}
                rewards_sum={a:0. for a in adapter.agent_ids};success=set();collisions=set();length=0
                while env.agents and learner.environment_steps<args.total_steps:
                    active=adapter.active_observations()
                    if learner.environment_steps<config.warmup_steps:
                        actions={a:learner.rng.uniform(-1,1,3).astype(np.float32) for a in active}
                    else:actions=learner.select_actions(active,training=True)
                    if args.telemetry_q_interval and learner.environment_steps%args.telemetry_q_interval==0:
                        telemetry.sample_q(learner,adapter,actions)
                    _,raw_rewards,terminated,truncated,infos=adapter.step(actions)
                    rewards,penalties=adjustment.apply(raw_rewards,terminated,truncated,infos)
                    transition=dict(adapter.last_transition,rewards=rewards)
                    learner.record_transition(**transition)
                    counters['agent_transitions']+=len(actions);length+=1
                    for a,r in rewards.items():
                        rewards_sum[a]+=r
                        raw_rewards_sum[a]+=raw_rewards[a];penalties_sum[a]+=penalties[a]
                        if infos[a]['success']:success.add(a)
                        if infos[a]['collision']:collisions.add(a)
                    losses=learner.update()
                    telemetry.observe(actions,infos,losses)
                    if losses is not None:log('updates.jsonl',dict(episode=episode+1,step=learner.environment_steps,update=learner.update_steps,losses=losses))
                    if learner.environment_steps%config.validation_interval==0:
                        result=evaluate(learner,env_config,config.evaluation_episodes,config.seed+100000,
                            training_seeds=range(config.seed,config.seed+counters['episodes_started']),reward_adjustment=adjustment_config)
                        log('validation.jsonl',dict(step=learner.environment_steps,**result))
                    if learner.environment_steps%config.checkpoint_interval==0:save(f'checkpoint_{learner.environment_steps}.pt')
                completed=not env.agents
                counters['episodes_completed']+=int(completed)
                row=dict(episode=episode,seed=config.seed+episode,steps=length,completed=completed,
                    environment_steps=learner.environment_steps,agent_transitions=counters['agent_transitions'],
                    raw_team_reward=sum(raw_rewards_sum.values()),raw_per_agent_reward=raw_rewards_sum,
                    out_of_bounds_penalties=penalties_sum,
                    updates=learner.update_steps,per_agent_reward=rewards_sum,team_reward=sum(rewards_sum.values()),
                    success_agents=sorted(success),collision_agents=sorted(collisions),
                    successes=len(success),team_success=len(success)==len(adapter.agent_ids),collisions=len(collisions),
                    wall_seconds=base_wall+time.perf_counter()-started)
                row.update(telemetry.result())
                log('episodes.jsonl',row);print(json.dumps(row),flush=True)
        finally:save('final.pt')
        summary=dict(**counters,environment_steps=learner.environment_steps,updates=learner.update_steps,replay_size=len(learner.replay))
        summary.update(run_id=experiment.run_id,run_directory=str(experiment.directory),checkpoint_paths=checkpoint_paths,
                       final_checkpoint=checkpoint_paths[-1])
        episodes=counters['episodes_completed']
        write_rows(experiment.path('training','csv',episodes,steps=learner.environment_steps),episode_rows)
        for kind,field in [('reward_plot','team_reward'),('success_rate_plot','team_success'),('collision_plot','collisions')]:
            data=[r[field] for r in episode_rows]
            if field=='team_success':data=np.cumsum(data)/np.arange(1,len(data)+1) if data else []
            write_plot(experiment.path(kind,'png',episodes,steps=learner.environment_steps),data,kind,
                       x=[r['episode']+1 for r in episode_rows])
        loss_rows=[dict(episode=r['episode'],step=r['step'],update=r['update'],agent=a,**loss) for r in update_rows for a,loss in r['losses'].items()]
        write_rows(experiment.path('actor_critic_loss','csv',episodes,steps=learner.environment_steps),loss_rows)
        loss_episodes,loss_series=mean_agent_loss_series(loss_rows)
        if loss_rows:
            write_plot(experiment.path('network_loss_plot','png',episodes,steps=learner.environment_steps),[],
                       'network_loss_plot',series=loss_series,ylabel='Mean loss per episode',x=loss_episodes)
        if validation_rows:
            from algorithms.evaluation import aggregate
            rows=[dict(row,evaluation_step=entry['step']) for entry in validation_rows for row in entry['episode_metrics']]
            combined=dict(**aggregate(rows),episode_metrics=rows,validation_seeds=[r['seed'] for r in rows])
            export_evaluation(combined,args.output,args.algo,learner,env_config,kind='validation',origin_run=experiment.run_id,experiment=experiment)
        experiment.path('summary','json',episodes,steps=learner.environment_steps).write_text(json.dumps(summary,indent=2)+'\n')
        return summary


if __name__=='__main__':
    parser=build_parser()
    try:run(parser.parse_args())
    except ValueError as error:parser.error(str(error))
    except KeyboardInterrupt:pass
