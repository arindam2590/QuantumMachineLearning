"""Repeatable headless validation and sensor-overhead benchmark (no learning)."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
from config import EnvConfig
from env import MultiDroneEnv
from main import demo_policy


def run(scenario, count, steps, mode, seed, moving):
    config = EnvConfig(scenario=scenario, num_drones=count, gui=False, max_episode_steps=steps)
    config.sensors.lidar_enabled = mode != 'state_only'
    config.sensors.camera_enabled = mode == 'rgbd'
    config.obstacles.count = moving
    with MultiDroneEnv(config) as env:
        obs, _ = env.reset(seed=seed)
        starts = {a:d.state()[0] for a,d in env.drones.items()}
        env.reset(seed=seed)
        assert all(np.array_equal(starts[a], d.state()[0]) for a,d in env.drones.items())
        for a,d in env.drones.items():
            assert env.source.contains(d.state()[0],config.drone.radius)
            for b in env.town.static_body_ids:
                assert not p.getClosestPoints(d.body_id,b,0,physicsClientId=env.client)
        obs = env._observations()
        distance = {a:0. for a in env.possible_agents}
        collisions = dict(uav=0, building=0, moving=0, other=0)
        minimum = float('inf')
        total_reward = 0.
        started = time.perf_counter()
        while env.agents:
            active = list(env.agents)
            before = {a:env.drones[a].state()[0] for a in active}
            bodies = {d.body_id for d in env.drones.values()}
            buildings = set(env.town.static_body_ids)
            moving_ids = {item['body'] for item in env.moving_obstacles.items}
            obs,rewards,_,_,infos = env.step(demo_policy(obs,env))
            total_reward += sum(rewards.values())
            for a in active:
                distance[a] += float(np.linalg.norm(env.drones[a].state()[0]-before[a]))
                if infos[a]['collision']:
                    hits = getattr(env.drones[a], 'collision_bodies', set())
                    collisions['uav'] += bool(hits & bodies)
                    collisions['building'] += bool(hits & buildings)
                    collisions['moving'] += bool(hits & moving_ids)
                    collisions['other'] += not bool(hits & (bodies|buildings|moving_ids))
            positions = np.array([env.drones[a].state()[0] for a in active])
            if len(positions)>1:
                differences = np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
                np.fill_diagonal(differences,np.inf)
                minimum = min(minimum,float(differences.min()))
        elapsed = time.perf_counter()-started
        return dict(scenario=scenario, fleet=count, sensors=mode, seed=seed,
                    **env.episode_statistics, steps_per_second=env.step_count/elapsed,
                    wall_seconds=elapsed, min_inter_uav_distance=minimum,
                    collision_uav=collisions['uav'], collision_building=collisions['building'],
                    collision_moving=collisions['moving'], collision_other=collisions['other'],
                    travel_distances=json.dumps(distance), arrival_times=json.dumps(env.arrival_times),
                    energy_consumption=json.dumps({a:d.battery_model.consumed for a,d in env.drones.items()}),
                    episode_reward=total_reward)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--moving',type=int,default=0)
    parser.add_argument('--sensor-modes',nargs='+',choices=['state_only','lidar','rgbd'],default=['state_only','lidar','rgbd'])
    parser.add_argument('--output',type=Path,default=Path('outputs/validation.csv'))
    args = parser.parse_args()
    rows=[]
    for scenario in ['urban','open_field','cluttered']:
        for count in [2,4,8,16,20]:
            for mode in args.sensor_modes:
                row=run(scenario,count,args.steps,mode,args.seed,args.moving)
                rows.append(row)
                print(scenario,count,mode,round(row['steps_per_second'],2),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    import sys
    if any(argument == '--algo' or argument.startswith('--algo=') for argument in sys.argv[1:]):
        from algorithms.evaluation import validation_main
        validation_main()
    else:
        main()
