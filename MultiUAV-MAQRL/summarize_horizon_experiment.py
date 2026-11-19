"""Read saved horizon-only artifacts and produce a paired comparison report."""
from pathlib import Path
import csv,json,hashlib
from collections import Counter
import numpy as np


def read_rows(path):
    return list(csv.DictReader(path.open()))


def main():
    parent=Path('outputs/experiments/maddpg/urban/n2')
    baseline=parent/'maddpg_urban_n2_seed42'
    candidates=[]
    for p in parent.glob('*/*_metadata.json'):
        m=json.loads(p.read_text())
        if m.get('kind')=='training' and m['environment']['max_episode_steps']==1600 and m['training_budget']==186374:
            candidates.append(p.parent)
    if len(candidates)!=1:raise ValueError(f'Expected exactly one horizon-only run, found {len(candidates)}')
    run=candidates[0];out=Path('outputs/benchmarks/horizon_1600_gamma_095')
    summary=json.loads(next(run.glob('*_summary.json')).read_text())
    new=read_rows(next((run/'training').glob('*_training.csv')))
    old=read_rows(next((baseline/'training').glob('*_training.csv')))
    validation=read_rows(next((run/'validation').glob('*_validation.csv')))
    old_validation=read_rows(next((baseline/'validation').glob('*_validation.csv')))
    def episodes(rows):
        return dict(episodes=len(rows),completed=sum(r['completed']=='True' for r in rows),arrivals=sum(int(r['successes']) for r in rows),
                    team_successes=sum(r['team_success']=='True' for r in rows),collisions=sum(int(r['collisions']) for r in rows),
                    mean_length=float(np.mean([int(r['steps']) for r in rows])))
    def val(rows):
        reasons=Counter(status['status'] for r in rows for status in json.loads(r['per_agent_status']).values())
        return dict(episodes=len(rows),arrivals=sum(int(r['arrivals']) for r in rows),
                    team_successes=sum(r['team_success']=='True' for r in rows),reasons=dict(reasons),
                    mean_reward=float(np.mean([float(r['reward']) for r in rows])))
    recent=new[-100:]
    telemetry={}
    for agent in ['uav_0','uav_1']:
        telemetry[agent]=dict(action_mean=np.mean([json.loads(r['action_mean'])[agent] for r in recent],axis=0).tolist(),
            saturation=np.mean([json.loads(r['action_saturation'])[agent] for r in recent],axis=0).tolist(),
            mean_progress=float(np.mean([json.loads(r['destination_progress'])[agent] for r in recent])))
    new_reasons=Counter(v for r in new for v in json.loads(r['agent_status']).values())
    protected=json.loads((out/'protected_hashes.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in protected.items())
    result=dict(run=str(run),summary=summary,baseline_training=episodes(old),horizon_training=episodes(new),
        horizon_training_reasons=dict(new_reasons),historical_validation=val(old_validation),horizon_validation=val(validation),
        recent_100_episode_telemetry=telemetry,protected_sources_unchanged=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
