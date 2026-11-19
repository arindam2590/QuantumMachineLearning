"""Reproducible cProfile benchmark with output fingerprints and peak RSS."""
import argparse
import csv
import cProfile
import hashlib
import json
from pathlib import Path
import resource
import time
import pstats
import numpy as np
from config import EnvConfig
from env import MultiDroneEnv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',default='baseline')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--camera',action='store_true')
    parser.add_argument('--moving',type=int,default=0)
    args=parser.parse_args()
    out=Path('outputs/benchmarks');out.mkdir(parents=True,exist_ok=True)
    records=[]
    for count in [2,4,8,16,20]:
        cfg=EnvConfig(scenario='open_field',gui=False,num_drones=count,max_episode_steps=args.steps)
        cfg.sensors.camera_enabled=args.camera
        cfg.obstacles.count=args.moving
        fingerprint=hashlib.sha256()
        with MultiDroneEnv(cfg) as env:
            env.reset(seed=42)
            profile=cProfile.Profile()
            profile.enable(); start=time.perf_counter()
            while env.agents:
                actions={a:np.array([.05,0,0],dtype=np.float32) for a in env.agents}
                obs,rewards,term,trunc,infos=env.step(actions)
                for a in obs:
                    fingerprint.update(obs[a].tobytes())
                    fingerprint.update(env.drones[a].physical_state().tobytes())
                    fingerprint.update(np.float64(rewards[a]).tobytes())
            elapsed=time.perf_counter()-start;profile.disable()
            profile.dump_stats(str(out/f'{args.label}_{count}.prof'))
            stats=pstats.Stats(profile)
            def total(names):
                return sum(value[2] for key,value in stats.stats.items() if any(name in key[2] for name in names))
            records.append(dict(fleet=count,steps=env.step_count,seconds=elapsed,steps_per_second=env.step_count/elapsed,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                physics_seconds=total(['stepSimulation']),controller_seconds=total(['compute','body_rotation_error','rotation_quaternion','attitude_torque']),
                collision_seconds=total(['_swept_collisions','getClosestPoints','getContactPoints','getOverlappingObjects','getAABB','getCollisionShapeData']),
                sensor_seconds=total(['rayTestBatch','getCameraImage','acquire','metric_depth']),
                observation_reward_seconds=total(['encode_observation','reward_components','_observations']),
                physics_substep_mean_seconds=total(['stepSimulation'])/(env.step_count*(cfg.physics_hz//cfg.control_hz)),
                sensor_update_seconds=sum(v[3] for k,v in stats.stats.items() if k[0].endswith('/env/sensors.py') and k[2]=='update'),
                dynamic_obstacle_seconds=sum(v[3] for k,v in stats.stats.items() if k[0].endswith('/env/obstacles.py') and k[2]=='update'),
                controller_total_seconds=sum(v[3] for k,v in stats.stats.items() if k[0].endswith('/env/drone_controller.py') and k[2]=='compute'),
                fingerprint=fingerprint.hexdigest()))
            with (out/f'{args.label}_{count}.txt').open('w') as stream:
                pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(35)
    (out/f'{args.label}.json').write_text(json.dumps(records,indent=2)+'\n')
    with (out/f'{args.label}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()
