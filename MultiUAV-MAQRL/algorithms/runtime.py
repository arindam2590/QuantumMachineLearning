"""Shared algorithm factory, Ray execution, metrics and deterministic evaluation."""
import argparse,copy,csv,hashlib,json,random,uuid,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import ray
from ray.tune.registry import register_env
from config.maddpg_config import MADDPGConfig
from env import EnvConfig
from .adapter import UAVAdapter
from .maddpg import MADDPG

ALGORITHMS={'maddpg':MADDPG}

def environment_hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('env').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def write_results(directory,prefix,rows,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/(prefix+'.json')).write_text(json.dumps(dict(metadata=metadata,episodes=rows),indent=2))
    if rows:
        with (directory/(prefix+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
            for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def mean_optimizer_loss(updates, name):
    values=[float(v[name]) for update in updates for v in update.values() if name in v]
    if not values:return None
    if not all(np.isfinite(v) for v in values) or (name=='critic' and any(v<0 for v in values)):
        raise FloatingPointError('Invalid optimizer loss')
    return float(np.mean(values))


def collision_incidents(infos, drones, owners):
    """Unique UAV pairs; one terminal obstacle incident per UAV, not contact points."""
    events=set();unknown=set()
    for agent,info in infos.items():
        if not info.get('collision'):continue
        hits=set(drones[agent].collision_bodies)
        if not hits:unknown.add(agent)
        for body in hits:
            other=owners.get(body)
            if other is not None and other!=agent:events.add(('uav',*sorted((agent,other))))
            elif other is None:events.add(('obstacle',agent))
    # Symmetrically flagged partners need not both have retained final-substep hits.
    covered={a for event in events if event[0]=='uav' for a in event[1:]}
    unknown-=covered
    events.update(('unknown',a) for a in unknown)
    return events,unknown


def console_section(title,fields):
    print('\n'+title+'\n'+'-'*78,flush=True)
    for name,value in fields:print(f'{name:<34}: {value}',flush=True)


def episode_console(row, budget):
    number=lambda value:'N/A' if value is None else f'{value:.2f}'
    suffix=' (partial)' if not row['completed'] else ''
    print(f"Episode: {row['episode']}/{budget if budget is not None else 'N/A'}{suffix} | Steps: {row['steps']} | "
          f"Avg. Team Reward: {number(row['average_team_reward'])} | Avg. Actor Loss: {number(row['average_actor_loss'])} | "
          f"Avg. Critic Loss: {number(row['average_critic_loss'])} | Collisions: {row['collision_events'] if row['collision_events'] is not None else 'N/A'} | "
          f"Goal Reach: {row['arrivals']}/{len(row['per_agent_rewards'])} | Boundary Exits: {row['boundary_exits']}",flush=True)
    print('\n'+'-'*78,flush=True)


def train_loop(options):
    started=time.perf_counter()
    args=argparse.Namespace(**options);config=MADDPGConfig(**args.algorithm_config);config.validate()
    if args.scenario!='urban':raise ValueError('New training experiments require scenario=urban')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    episode_budget=getattr(args,'episodes',None)
    step_budget=getattr(args,'total_steps',None)
    if episode_budget is not None and episode_budget<1:raise ValueError('episodes must be positive')
    if episode_budget is None and (step_budget is None or step_budget<1):raise ValueError('Specify --episodes or legacy --total-steps')
    if not 1<=args.episode_steps<=1500:raise ValueError('episode-steps must be between1 and1500')
    cfg=EnvConfig(gui=False,scenario=args.scenario,num_drones=args.num_drones,max_episode_steps=args.episode_steps)
    cfg.drone.velocity_mode='pd';cfg.sensors.camera_enabled=False
    cfg.town.building_density=args.building_density
    register_env('multi_uav',lambda context:UAVAdapter(context))
    env=UAVAdapter(cfg);model=ALGORITHMS[args.algo](env.possible_agents,config)
    directory=Path(args.output)/args.algo/args.scenario/f'n{args.num_drones}'/uuid.uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    console_section('TRAINING EXPERIMENT',[
        ('Algorithm / scenario',f'{args.algo} / {cfg.scenario}'),('Number of UAVs',cfg.num_drones),
        ('Total episodes',episode_budget if episode_budget is not None else 'N/A (legacy step budget)'),
        ('Maximum steps per episode',cfg.max_episode_steps),('Legacy step cap',step_budget if step_budget is not None else 'None'),
        ('Observation / action dimensions',f'{env.env.observation_space.shape} / {env.env.action_space.shape}'),
        ('Controller',cfg.drone.velocity_mode.upper()),('Actor / critic learning rate',f'{config.actor_lr} / {config.critic_lr}'),
        ('Gamma / tau',f'{config.gamma} / {config.tau}'),('Batch size / replay capacity',f'{config.batch_size} / {config.replay_capacity}'),
        ('Exploration noise',f'Gaussian sigma={config.noise_std}, training only'),('Device',str(next(model.actors.parameters()).device)),
        ('Random seed',config.seed),('Checkpoint / result directory',str(directory)),
        ('Reward convention','Avg. Team Reward = current episode UAV-return sum / episode steps; no rolling mean'),
        ('Collision convention','Unique UAV pairs + per-UAV terminal obstacle incidents; unknown attribution yields N/A')])
    diagnostic_enabled=getattr(args,'actor_diagnostics',False)
    diagnostic_rows=[];gradient_rows=[];probe_cohort=None;stop_reason=None
    gradient_squares={a:0. for a in model.ids};gradient_window={a:[] for a in model.ids}
    saturation_streak={a:0 for a in model.ids}
    initial_network_sha256=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for group in [model.actors,model.critics] for v in group.state_dict().values())).hexdigest()
    if diagnostic_enabled:
        for i,agent in enumerate(model.ids):
            def hook(grad,agent=agent):gradient_squares[agent]+=float(grad.detach().square().sum())
            for param in model.actors[i].parameters():param.register_hook(hook)
    def diagnose():
        nonlocal probe_cohort,stop_reason
        if probe_cohort is None:
            probe_cohort={a:np.stack([r[0][i] for r in model.replay if r[8][i]]) for i,a in enumerate(model.ids)}
            probe_cohort={a:v[np.linspace(0,len(v)-1,min(128,len(v)),dtype=int)] for a,v in probe_cohort.items()}
            np.savez(directory/f'{args.algo}_{args.scenario}_n{args.num_drones}_ep{completed}_step{model.steps}_seed{config.seed}_diagnostic_cohort.npz',**probe_cohort)
        measurements={}
        for i,a in enumerate(model.ids):
            x=torch.as_tensor(probe_cohort[a]);flip=x.clone();flip[:,9:12]*=-1
            with torch.no_grad():
                logits=model.actors[i][:-1](x);actions=model.actors[i](x);reverse=model.actors[i](flip)
            saturation=(actions.abs()>=.98).float().mean(0)
            mean_grad=float(np.mean(gradient_window[a])) if gradient_window[a] else None
            severe=bool(torch.all(saturation>=.99)) and mean_grad is not None and mean_grad<=1e-8
            saturation_streak[a]=saturation_streak[a]+1 if severe else 0
            measurements[a]=dict(action_mean=actions.mean(0).tolist(),action_std=actions.std(0).tolist(),saturation=saturation.tolist(),
                logit_mean=logits.mean(0).tolist(),logit_std=logits.std(0).tolist(),logit_min=logits.min(0).values.tolist(),logit_max=logits.max(0).values.tolist(),
                logit_abs_quantiles=torch.quantile(logits.abs().flatten(),torch.tensor([.5,.9,.99])).tolist(),
                actual_update_gradient_mean=mean_grad,actual_update_gradient_max=max(gradient_window[a]) if gradient_window[a] else None,
                goal_flip_mean_abs_action_change=float((actions-reverse).abs().mean()),severe_streak=saturation_streak[a])
            gradient_window[a].clear()
            if saturation_streak[a]>=3:stop_reason=f'Sustained severe saturation and vanishing actor gradient: {a}'
        diagnostic_rows.append(dict(step=model.steps,actors=measurements))
        print('ACTOR DIAGNOSTIC '+json.dumps(diagnostic_rows[-1]),flush=True)
    rows=[];last_loss=None;episode=0;completed=0;agent_transitions=0;periodic_checkpoints=[]
    checkpoint_interval=getattr(args,'checkpoint_interval',5000)
    if checkpoint_interval<1:raise ValueError('checkpoint interval must be positive')
    def checkpoint_metadata(started,finished):
        return dict(algo=args.algo,config=asdict(config),environment=cfg,environment_hashes=environment_hashes(),
                    episodes=started,completed_episodes=finished,requested_episodes=episode_budget,requested_step_budget=step_budget,agent_transitions=agent_transitions,critic_encoding_version=2,
                    execution='custom PyTorch MADDPG in a Ray task; RLlib MultiAgentEnv adapter',ray_version=ray.__version__,
                    training_seeds=list(range(config.seed,config.seed+started)),
                    continuation='Learner state restorable; no mid-episode physics snapshot.')
    active_episode=False
    def report_episode():
        nonlocal completed,episode,active_episode
        if not active_episode or not length:return
        completed+=int(not env.agents);episode+=1
        row=dict(episode=episode,steps=length,environment_steps=model.steps,completed=not env.agents,
            team_reward=sum(rewards_sum.values()),per_agent_rewards=rewards_sum,arrivals=sum(v=='arrived' for v in env.env.status.values()),
            team_success=all(v=='arrived' for v in env.env.status.values()),collisions=sum(v=='collision' for v in env.env.status.values()),
            status=dict(env.env.status),updates=model.updates,
            episode_updates=len(losses),arrival_rate=sum(v=='arrived' for v in env.env.status.values())/len(model.ids),
            boundary_exits=sum(v=='out_of_bounds' for v in env.env.status.values()),goal_progress={a:initial_distances[a]-distances[a] for a in model.ids},seed=config.seed+episode-1,exploration_noise_std=model.config.noise_std,final_goal_distance=distances,
            losses={a:{k:float(np.mean([x[a][k] for x in losses if a in x])) for k in ('actor','critic','q_mean')}
                    for a in model.ids if any(a in x for x in losses)})
        row.update(average_team_reward=row['team_reward']/length if length else None,
            average_actor_loss=mean_optimizer_loss(losses,'actor'),average_critic_loss=mean_optimizer_loss(losses,'critic'),
            collision_events=len(events) if not unknown_collisions else None,known_collision_events=len([v for v in events if v[0]!='unknown']),collision_event_details=[list(v) for v in sorted(events)],unknown_collision_agents=sorted(unknown_collisions),agent_transitions=agent_transitions)
        rows.append(row);episode_console(row,episode_budget)
        active_episode=False
    try:
        while (completed<episode_budget if episode_budget is not None else model.steps<step_budget) and stop_reason is None:
            obs,info=env.reset(seed=config.seed+episode);initial_distances={a:float(np.linalg.norm(env.env.goals[a]-env.env.drones[a].state()[0])) for a in model.ids};rewards_sum={a:0. for a in env.possible_agents};length=0;losses=[];distances={}
            events=set();unknown_collisions=set();active_episode=True
            while env.agents and (step_budget is None or model.steps<step_budget) and stop_reason is None:
                active=list(env.agents);values,state=env.pack(obs,active)
                actions=model.actions({a:obs[a] for a in active},training=True)
                owners={d.body_id:a for a,d in env.env.drones.items() if env.env.status[a]=='active'}
                nxt,rewards,term,trunc,info=env.step(actions)
                new_events,unknown=collision_incidents(info,env.env.drones,owners);events.update(new_events);unknown_collisions.update(unknown)
                nv,ns=env.pack(nxt)
                model.add(values,state,actions,rewards,nv,ns,term,trunc,active)
                agent_transitions+=len(active)
                for a,r in rewards.items():
                    rewards_sum[a]+=r
                    distances[a]=float(info[a]['distance_to_goal'])
                obs={a:nxt[a] for a in env.agents};length+=1
                if diagnostic_enabled:
                    for a in model.ids:gradient_squares[a]=0.
                last=model.update()
                if diagnostic_enabled and last is not None:
                    norms={a:float(np.sqrt(gradient_squares[a])) for a in last}
                    gradient_rows.append(dict(step=model.steps,**norms))
                    for a,v in norms.items():gradient_window[a].append(v)
                if last is not None:last_loss=last;losses.append(last)
                if diagnostic_enabled and model.steps>=config.warmup_steps and model.steps%100==0:diagnose()
                scheduled=(model.steps in args.checkpoint_steps if getattr(args,'checkpoint_steps',None) else model.steps%checkpoint_interval==0)
                if scheduled and (step_budget is None or model.steps<step_budget) and stop_reason is None and not (episode_budget is not None and completed+int(not env.agents)>=episode_budget):
                    finished=completed+int(not env.agents)
                    name=f'{args.algo}_{args.scenario}_n{args.num_drones}_ep{finished}_step{model.steps}_seed{config.seed}_checkpoint.pt'
                    path=directory/name
                    model.save(path,checkpoint_metadata(episode+1,finished));periodic_checkpoints.append(str(path))
                    print(f'Checkpoint saved: {path}',flush=True)
            report_episode()
    except KeyboardInterrupt:
        stop_reason='Interrupted by user (partial physical episode is not restorable)'
        report_episode()
    except Exception:
        env.close()
        raise
    try:
        prefix=f'{args.algo}_{args.scenario}_n{args.num_drones}_ep{completed}_step{model.steps}_seed{config.seed}'
        metadata=checkpoint_metadata(episode,completed)
        checkpoint=directory/(prefix+'_checkpoint.pt');model.save(checkpoint,metadata)
        printable=dict(metadata,stop_reason=stop_reason,requested_budget=step_budget,requested_episodes=episode_budget,agent_transitions=agent_transitions,training_duration_seconds=time.perf_counter()-started,initial_network_sha256=initial_network_sha256,diagnostic_stop_rule="Any actor, every axis saturation >=99% and actual-update mean gradient <=1e-8, three successive 100-step probes",periodic_checkpoints=periodic_checkpoints,environment=asdict(cfg),checkpoint=str(checkpoint),steps=model.steps,updates=model.updates,replay_rows=len(model.replay),final_losses=last_loss)
        if diagnostic_enabled:
            write_results(directory,prefix+'_actor_diagnostics',diagnostic_rows,dict(fixed_cohort=True,stop_reason=stop_reason,gradient_type='actual backward-pass gradients; no optimizer modification'))
            write_results(directory,prefix+'_actor_gradients',gradient_rows,dict(gradient_type='actual training update L2 norm before Adam step'))
        normal=(completed>=episode_budget if episode_budget is not None else model.steps>=step_budget)
        status='completed' if normal and stop_reason is None else 'interrupted' if stop_reason and stop_reason.startswith('Interrupted') else 'early_stopped'
        finished_rows=[r for r in rows if r['completed']]
        printable.update(completion_status=status,completed_episodes=completed,
            mean_team_reward=float(np.mean([r['team_reward'] for r in finished_rows])) if finished_rows else None,
            individual_arrival_rate=sum(r['arrivals'] for r in finished_rows)/(len(finished_rows)*len(model.ids)) if finished_rows else None,
            team_success_rate=sum(r['team_success'] for r in finished_rows)/len(finished_rows) if finished_rows else None,
            collision_events=sum(r['collision_events'] for r in rows) if all(r['collision_events'] is not None for r in rows) else None,boundary_exits=sum(r['boundary_exits'] for r in rows),
            mean_episode_length=float(np.mean([r['steps'] for r in finished_rows])) if finished_rows else None)
        write_results(directory,prefix+'_training',rows,printable)
        console_section('POST-TRAINING SUMMARY',[
            ('Completion status',status),('Stop reason',stop_reason or 'Requested budget reached'),
            ('Requested / completed episodes',f'{episode_budget if episode_budget is not None else "N/A"} / {completed}'),
            ('Partial episode rows',sum(not r['completed'] for r in rows)),('Environment steps',model.steps),
            ('Agent transitions',agent_transitions),('Mean team reward (episode return)',printable['mean_team_reward']),
            ('Individual arrival rate',printable['individual_arrival_rate']),('Team success rate',printable['team_success_rate']),
            ('Collision events',printable['collision_events']),('Boundary exits',printable['boundary_exits']),
            ('Mean episode length',printable['mean_episode_length']),('Training duration (worker seconds)',printable['training_duration_seconds']),
            ('Final checkpoint',str(checkpoint))])
        return printable
    finally:env.close()

def parser(mode):
    p=argparse.ArgumentParser(description=f'General {mode} entry point')
    p.add_argument('--algo',choices=list(ALGORITHMS),default='maddpg')
    p.add_argument('--scenario',choices=['urban'],default='urban')
    p.add_argument('--num-drones',type=int,default=2 if mode=='training' else None)
    p.add_argument('--output',default='outputs')
    p.add_argument('--seed',type=int,default=42 if mode=='training' else 100042 if mode=='validation' else 200042)
    if mode=='training':
        budgets=p.add_mutually_exclusive_group(required=True)
        budgets.add_argument('--episodes',type=int,help='primary stopping budget: complete episodes')
        budgets.add_argument('--total-steps',type=int,help='legacy absolute step cap; may end a partial episode')
        p.add_argument('--checkpoint-interval',type=int,default=5000)
        p.add_argument('--checkpoint-steps',type=int,nargs='+')
        p.add_argument('--actor-diagnostics',action='store_true')
        p.add_argument('--episode-steps',type=int,default=1500,help='public environment time limit,1..1500')
        p.add_argument('--building-density',type=float,default=.65)
        for k,v in asdict(MADDPGConfig()).items():
            if k=='seed':continue
            if k=='hidden_layers':p.add_argument('--hidden-layers',type=int,nargs='+',default=v)
            else:p.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    else:
        p.add_argument('--checkpoint',required=True,type=Path)
        p.add_argument('--episodes',type=int,default=10)
        rendering=p.add_mutually_exclusive_group();rendering.add_argument('--render',action='store_true');rendering.add_argument('--headless',action='store_true')
    return p

def training_main(argv=None):
    args=parser('training').parse_args(argv)
    if ((args.episodes is not None and args.episodes<1) or (args.total_steps is not None and args.total_steps<1) or not 1<=args.episode_steps<=1500 or args.checkpoint_interval<1):raise ValueError('budgets must be positive; episode horizon must be1..1500')
    args.algorithm_config={k:getattr(args,k) for k in asdict(MADDPGConfig())}
    args.algorithm_config['hidden_layers']=tuple(args.algorithm_config['hidden_layers'])
    MADDPGConfig(**args.algorithm_config).validate()
    ray.init(num_cpus=2,include_dashboard=False)
    try:return ray.get(ray.remote(num_cpus=1)(train_loop).remote(vars(args)))
    except KeyboardInterrupt:
        print('TRAINING INTERRUPTED: Ray worker cancellation may prevent a final save; use only successfully written checkpoints.',flush=True)
    finally:ray.shutdown()

def evaluation_main(mode,argv=None):
    args=parser(mode).parse_args(argv)
    if args.episodes<1:raise ValueError('episodes must be positive')
    model=ALGORITHMS[args.algo].load(args.checkpoint)
    if model.metadata['algo']!=args.algo:raise ValueError('algorithm mismatch')
    cfg=copy.deepcopy(model.metadata['environment']);cfg.gui=args.render
    if args.num_drones is not None and cfg.num_drones!=args.num_drones:raise ValueError('fleet mismatch')
    if args.scenario is not None and cfg.scenario!=args.scenario:raise ValueError('scenario mismatch')
    if model.metadata['environment_hashes']!=environment_hashes():raise ValueError('environment source mismatch')
    if set(range(args.seed,args.seed+args.episodes)) & set(model.metadata.get('training_seeds',range(model.config.seed,model.config.seed+model.metadata['episodes']))):
        raise ValueError('Evaluation seeds overlap training seeds')
    before={k:v.detach().clone() for k,v in model.actors.state_dict().items()}
    rows=[];steps=0
    env=UAVAdapter(cfg)
    try:
        for episode in range(args.episodes):
            obs,infos=env.reset(seed=args.seed+episode);reward={a:0. for a in model.ids};length=0;distances={};q_values={a:[] for a in model.ids}
            while env.agents:
                actions=model.actions({a:obs[a] for a in env.agents})
                if length%20==0:
                    _,state=env.pack(obs)
                    joint=np.concatenate([actions.get(a,np.zeros(3,np.float32)) for a in model.ids])
                    with torch.no_grad():
                        features=torch.as_tensor(np.concatenate([state,joint]),dtype=torch.float32)
                        for i,a in enumerate(model.ids):
                            if a in env.agents:q_values[a].append(float(model.critics[i](features).item()))
                obs,r,t,tr,info=env.step(actions);length+=1
                for a,v in info.items():distances[a]=float(v['distance_to_goal'])
                for a,value in r.items():reward[a]+=value
                if cfg.gui:env.render()
            steps+=length
            rows.append(dict(episode=episode+1,seed=args.seed+episode,steps=length,team_reward=sum(reward.values()),per_agent_rewards=reward,
                arrivals=sum(x=='arrived' for x in env.env.status.values()),team_success=all(x=='arrived' for x in env.env.status.values()),
                collisions=sum(x=='collision' for x in env.env.status.values()),status=dict(env.env.status),
                arrival_rate=sum(x=='arrived' for x in env.env.status.values())/len(model.ids),final_goal_distance=distances,
                q_mean={a:float(np.mean(v)) if v else None for a,v in q_values.items()}))
            print(json.dumps(rows[-1]),flush=True)
    finally:env.close()
    assert all(torch.equal(v,model.actors.state_dict()[k]) for k,v in before.items())
    prefix=f'{args.algo}_{cfg.scenario}_n{cfg.num_drones}_ep{args.episodes}_step{steps}_seed{args.seed}_{mode}'
    directory=Path(args.output)/args.algo/cfg.scenario/f'n{cfg.num_drones}'/uuid.uuid4().hex
    write_results(directory,prefix,rows,dict(checkpoint=str(args.checkpoint),policy_steps=model.steps,deterministic=True))
    return rows
