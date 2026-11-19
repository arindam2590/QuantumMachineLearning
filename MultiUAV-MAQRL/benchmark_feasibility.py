"""Fixed-seed urban route feasibility using the EXISTING rule controller."""
import argparse
import json
from pathlib import Path
import math
from config import EnvConfig
from env import MultiDroneEnv
from main import demo_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[100000,100001,100002])
    p.add_argument('--max-steps',type=int,default=2400)
    p.add_argument('--output',type=Path,default=Path('outputs/benchmarks/urban_feasibility.json'))
    args=p.parse_args();rows=[]
    for seed in args.seeds:
        cfg=EnvConfig(scenario='urban',num_drones=2,gui=False,max_episode_steps=args.max_steps)
        cfg.town.building_density=.65
        with MultiDroneEnv(cfg) as env:
            obs,_=env.reset(seed=seed);arrival={};statuses={}
            while env.agents:
                obs,_,_,_,infos=env.step(demo_policy(obs,env))
                for a,info in infos.items():
                    statuses[a]=info['status']
                    if info['success']:arrival[a]=env.step_count*cfg.control_dt
            row=dict(seed=seed,steps=env.step_count,arrival_times=arrival,status=statuses,
                     team_success=env.episode_statistics['team_success'])
            rows.append(row);print(row,flush=True)
    successes=[max(row['arrival_times'].values()) for row in rows if row['team_success']]
    result=dict(controller='existing main.demo_policy',density=.65,fleet=2,policy_hz=20,max_steps=args.max_steps,
        seeds=args.seeds,episodes=rows,observed_max_successful_time=max(successes) if successes else None,
        recommended_horizon_seconds=math.ceil(max(successes)*1.2) if len(successes)==len(rows) else None,
        interpretation='Observed route-specific completion; 20% margin is heuristic, not a minimum-time proof')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
