"""Algorithm-independent artifact names and immutable run provenance."""
from dataclasses import dataclass,asdict
from pathlib import Path
import csv
import hashlib
import json
import re
import uuid


def token(value):
    value=str(value).lower().replace('open_field','openfield')
    value=re.sub('[^a-z0-9]+','_',value).strip('_')
    if not value:raise ValueError('Empty result token')
    return value


def integer(name,value,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def seed_set_id(seeds):
    values=[integer('seed',s) for s in seeds]
    if not values:raise ValueError('Seed set cannot be empty')
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:12]


def checkpoint_provenance(path,learner):
    path=Path(path).resolve()
    return dict(checkpoint_path=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_step=learner.environment_steps,
        origin_run=(getattr(learner,'training_state',None) or {}).get('experiment',{}).get('run_id'))


@dataclass(frozen=True)
class Experiment:
    root: Path
    algo: str
    scenario: str
    drones: int
    seed: int
    run_id: str

    def __post_init__(self):
        object.__setattr__(self,'root',Path(self.root))
        for name in ('algo','scenario','run_id'):
            object.__setattr__(self,name,token(getattr(self,name)))
        integer('drones',self.drones,1);integer('seed',self.seed)

    @classmethod
    def create(cls,root,algo,scenario,drones,seed,metadata):
        integer('drones',drones,1);integer('seed',seed)
        # UUID is independent of experiment RNGs, preserving learning trajectories.
        base=f'{token(algo)}_{token(scenario)}_n{drones}_seed{seed}'
        item=cls(Path(root),token(algo),token(scenario),drones,seed,base)
        while True:
            try:
                item.directory.mkdir(parents=True,exist_ok=False)
                break
            except FileExistsError:
                item=cls(Path(root),token(algo),token(scenario),drones,seed,base+'_run'+uuid.uuid4().hex)
        for folder in ('checkpoints','training','validation','testing','plots'):
            (item.directory/folder).mkdir()
        reward=Path(__file__).resolve().parents[1]/'env'/'rewards.py'
        details=dict(metadata,**asdict(item))
        details['root']=str(item.root)
        details['reward_version']='sha256_'+hashlib.sha256(reward.read_bytes()).hexdigest()
        item.path('metadata','json',0).write_text(json.dumps(details,indent=2,allow_nan=False)+'\n')
        return item

    @property
    def directory(self):
        return self.root/'experiments'/self.algo/self.scenario/f'n{self.drones}'/self.run_id

    def path(self,result_type,ext,episodes,*,steps=None,seeds=None):
        integer('episodes',episodes)
        parts=[self.algo,self.scenario,f'n{self.drones}',f'ep{episodes}',f'seed{self.seed}',f'run{self.run_id}']
        if steps is not None:parts.append(f'step{integer("steps",steps)}')
        if seeds is not None:parts.append('seeds'+seed_set_id(seeds))
        kind=token(result_type)
        folder=('checkpoints' if kind in ('checkpoint','final_checkpoint') else
                'plots' if kind.endswith('_plot') else
                'validation' if kind.startswith('validation') else
                'testing' if kind.startswith(('test','testing')) else
                'training' if kind in ('training','loss','actor_critic_loss') else '')
        directory=self.directory/folder
        return directory/('_'.join(parts+[kind])+'.'+token(ext))


def write_rows(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def export_evaluation(result,root,algo,learner,environment,checkpoint=None,kind='validation',origin_run=None,experiment=None):
    from dataclasses import asdict
    seeds=result['validation_seeds']
    provenance=dict(origin_run=origin_run,checkpoint_step=learner.environment_steps) if checkpoint is None else checkpoint_provenance(checkpoint,learner)
    if experiment is None and checkpoint is not None:
        saved=(getattr(learner,'training_state',None) or {}).get('experiment',{})
        if saved.get('run_id') and saved.get('directory'):
            candidate=Experiment(Path(root),token(algo),token(environment.scenario),len(learner.agent_ids),learner.config.seed,saved['run_id'])
            if candidate.directory.resolve()==Path(saved['directory']).resolve() and candidate.directory.is_dir():
                experiment=candidate
    run=experiment or Experiment.create(root,algo,environment.scenario,len(learner.agent_ids),learner.config.seed,
        dict(kind=kind,algorithm_config=asdict(learner.config),environment=asdict(environment),
             evaluation_seeds=seeds,evaluation_episodes=result['episodes'],
             training_budget=((getattr(learner,'training_state',None) or {}).get('experiment',{}).get('training_budget',learner.config.max_training_steps)),**provenance))
    evaluation_id=uuid.uuid4().hex
    def artifact(result_type,extension):
        path=run.path(result_type,extension,result['episodes'],steps=learner.environment_steps,seeds=seeds)
        # Preserve result_type at the end while making repeated evaluations unique.
        return path.with_name(path.name.replace('_'+token(result_type)+'.','_eval'+evaluation_id+'_'+token(result_type)+'.'))
    if experiment is not None:
        details=dict(kind=kind,origin_run=run.run_id,evaluation_id=evaluation_id,
            algorithm_config=asdict(learner.config),environment=asdict(environment),evaluation_seeds=seeds,
            evaluation_episodes=result['episodes'],training_seed=learner.config.seed,
            training_budget=((getattr(learner,'training_state',None) or {}).get('experiment',{}).get('training_budget',learner.config.max_training_steps)),
            reward_version='sha256_'+hashlib.sha256((Path(__file__).resolve().parents[1]/'env'/'rewards.py').read_bytes()).hexdigest(),
            checkpoint_step=learner.environment_steps)
        details.update(provenance)
        artifact(kind+'_metadata','json').write_text(json.dumps(details,indent=2)+'\n')
    for ext in ['csv','json']:
        path=artifact(kind,ext)
        if ext=='csv':write_rows(path,result['episode_metrics'])
        else:path.write_text(json.dumps({k:v for k,v in result.items() if k!='episode_metrics'},indent=2,allow_nan=False)+'\n')
    for kind,field in [('reward_plot','reward'),('success_rate_plot','team_success'),('collision_plot','collisions')]:
        values=[r[field] for r in result['episode_metrics']]
        if field=='team_success':values=[sum(values[:i+1])/(i+1) for i in range(len(values))]
        write_plot(artifact(kind,'png'),values,kind)
    return run


def write_plot(path,values,label,*,series=None,ylabel=None,x=None):
    """One publication-sized PNG per metric, with 300 DPI axes and legends."""
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'multiuav_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    values=list(values)
    curves=series if series is not None else {label.replace('_plot','').replace('_',' ').title():values}
    if not any(len(v) for v in curves.values()):return
    fig,ax=plt.subplots(figsize=(9,5.5))
    try:
        for name,data in curves.items():
            data=list(data)
            if data:ax.plot(list(range(1,len(data)+1)) if x is None else x,data,label=name,linewidth=1.6)
        title=label.replace('_plot','').replace('_',' ').title()
        ax.set(title=title,xlabel='Episode',ylabel=ylabel or {'reward_plot':'Team episode reward','success_rate_plot':'Cumulative team success rate','collision_plot':'Colliding UAV count'}.get(label,title))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True,alpha=.25);ax.legend(loc='best');fig.tight_layout()
        fig.savefig(Path(path).with_suffix('.png'),dpi=300,bbox_inches='tight')
    finally:plt.close(fig)


def mean_agent_loss_series(rows):
    """Average updates within each agent/episode, then equally across agents."""
    from collections import defaultdict
    totals=defaultdict(lambda:[0.,0])
    episodes=set()
    for row in rows:
        episode=int(row['episode']);episodes.add(episode)
        for term in ('actor_loss','critic_loss'):
            key=(episode,row['agent'],term)
            totals[key][0]+=float(row[term]);totals[key][1]+=1
    episodes=sorted(episodes)
    means=defaultdict(list)
    for (episode,agent,term),(total,count) in totals.items():
        means[(episode,term)].append(total/count)
    series={term.replace('_',' ').capitalize()+' (agent mean)':[
        sum(means[(episode,term)])/len(means[(episode,term)]) for episode in episodes]
        for term in ('actor_loss','critic_loss')}
    return episodes,series
