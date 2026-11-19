"""Headless physical controller diagnostics; no learning or town changes."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
from config import DroneConfig, EnvConfig
from env import MultiDroneEnv
from env.drone import Drone
from env.drone_controller import VelocityController, body_rotation_error


def main():
    output=Path('outputs/dynamics')
    output.mkdir(parents=True,exist_ok=True)
    traces=[]
    summaries=[]
    cases=dict(hover=[(8.,[0,0,0])], directions=[(6.,v) for v in
        [[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,1],[0,0,-1],[0,0,0]]],
        recovery=[(10.,[0,0,0])],wind=[(10.,[0,0,0])])
    for name,segments in cases.items():
        client=p.connect(p.DIRECT)
        try:
            dt=1/240
            p.setTimeStep(dt,physicsClientId=client)
            cfg=DroneConfig()
            drone=Drone(client,[0,0,20],[.3,.7,.9,1],cfg)
            if name=='recovery':
                p.resetBasePositionAndOrientation(drone.body_id,[0,0,20],p.getQuaternionFromEuler([.15,-.12,.25]),physicsClientId=client)
            controller=VelocityController(cfg)
            errors=[]
            steady=[]
            started=time.perf_counter()
            tick=0
            for duration,command in segments:
                for local in range(round(duration/dt)):
                    thrust,torque=controller.compute(drone.physical_state(),command,dt)
                    wind=np.array([.2,0,0]) if name=='wind' else np.zeros(3)
                    drone.apply_wrench(thrust,torque,cfg.mass*wind)
                    p.stepSimulation(physicsClientId=client)
                    state=drone.physical_state()
                    assert np.isfinite(state).all()
                    error=float(np.linalg.norm(state[3:6]-command))
                    errors.append(error)
                    if local*dt>duration-1:
                        steady.append(error)
                    rpy=p.getEulerFromQuaternion(state[6:10])
                    traces.append(dict(case=name,time=tick*dt,position=json.dumps(state[:3].tolist()),
                        velocity=json.dumps(state[3:6].tolist()),attitude_rpy=json.dumps(rpy),
                        command=json.dumps(command),tracking_error=error,thrust=thrust,
                        torque=json.dumps(torque.tolist())))
                    tick+=1
            elapsed=time.perf_counter()-started
            state=drone.physical_state()
            summaries.append(dict(case=name,physics_steps=tick,steps_per_second=tick/elapsed,
                mean_tracking_error=float(np.mean(errors)),steady_mean_tracking_error=float(np.mean(steady)),
                final_position=state[:3].tolist(),displacement=float(np.linalg.norm(state[:3]-[0,0,20])),
                final_attitude_error=float(np.linalg.norm(body_rotation_error(state[6:10],[0,0,0,1])))))
        finally:
            p.disconnect(physicsClientId=client)
    for count in [2,4,8,16,20]:
        cfg=EnvConfig(scenario='open_field',num_drones=count,gui=False,max_episode_steps=100)
        with MultiDroneEnv(cfg) as env:
            obs,_=env.reset(seed=42)
            starts={a:d.state()[0] for a,d in env.drones.items()}
            started=time.perf_counter()
            while env.agents:
                obs,_,term,trunc,_=env.step({a:np.zeros(3,dtype=np.float32) for a in env.agents})
                assert all(env.observation_spaces[a].contains(value) for a,value in obs.items())
            elapsed=time.perf_counter()-started
            summaries.append(dict(case=f'fleet_{count}',steps=env.step_count,steps_per_second=env.step_count/elapsed,
                max_hover_drift=max(float(np.linalg.norm(d.state()[0]-starts[a])) for a,d in env.drones.items()),
                truncated=trunc['__all__'],terminated=term['__all__']))
    with (output/'traces.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=traces[0])
        writer.writeheader()
        writer.writerows(traces)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
