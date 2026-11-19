"""Matched deterministic PD/PID controller comparisons; no training."""
import csv
import json
from pathlib import Path
import numpy as np
import pybullet as p
from config import DroneConfig
from env.drone import Drone
from env.drone_controller import VelocityController


def main():
    rows=[]; summaries=[]
    for case in ['ideal','constant','varying','directional','changes','recovery','saturation']:
        for mode in ['pd','pid']:
            client=p.connect(p.DIRECT)
            try:
                cfg=DroneConfig(velocity_mode=mode)
                if case=='saturation':
                    cfg.max_collective_thrust=cfg.mass*cfg.gravity*1.05
                    cfg.max_roll=cfg.max_pitch=.08
                dt=1/120
                p.setTimeStep(dt,physicsClientId=client)
                drone=Drone(client,[0,0,20],[1,0,0,1],cfg)
                if case=='recovery':
                    p.resetBasePositionAndOrientation(drone.body_id,[0,0,20],p.getQuaternionFromEuler([.15,-.12,.25]),physicsClientId=client)
                controller=VelocityController(cfg)
                errors=[]; attitudes=[]; overshoots=[]; positions=[]
                for tick in range(2400):
                    t=tick*dt
                    command=np.zeros(3)
                    if case=='directional': command=np.array([2.,0,0])
                    if case=='changes': command=np.array([2. if t<6 else -2. if t<12 else 0.,0,0])
                    if case=='saturation' and t<8: command=np.array([5.,0,2.5])
                    wind=np.array([.2,0,0]) if case=='constant' else np.array([.2+.1*np.sin(t),0,0]) if case=='varying' else np.zeros(3)
                    thrust,torque=controller.compute(drone.physical_state(),command,dt)
                    drone.apply_wrench(thrust,torque,cfg.mass*wind)
                    p.stepSimulation(physicsClientId=client)
                    state=drone.physical_state(); assert np.isfinite(state).all()
                    error=float(np.linalg.norm(state[3:6]-command)); errors.append(error)
                    rpy=np.array(p.getEulerFromQuaternion(state[6:10])); attitudes.append(float(np.max(np.abs(rpy))))
                    overshoots.append(float(max(0.,np.linalg.norm(state[3:6])-np.linalg.norm(command))))
                    positions.append(state[:3].copy())
                    rows.append(dict(case=case,mode=mode,seed=42,time=t,position=json.dumps(state[:3].tolist()),velocity=json.dumps(state[3:6].tolist()),
                        attitude=json.dumps(rpy.tolist()),command=json.dumps(command.tolist()),tracking_error=error,thrust=thrust,torque=json.dumps(torque.tolist()),integral=json.dumps(controller.integral_error.tolist())))
                errors=np.array(errors)
                # Settling after final command transition: within .05 m/s for remainder.
                start=12 if case=='changes' else 8 if case=='saturation' else 0
                candidates=np.where((np.maximum.accumulate(errors[::-1])[::-1]<.05)&(np.arange(2400)*dt>=start))[0]
                summaries.append(dict(case=case,mode=mode,seed=42,mean_error=float(errors.mean()),last_second_error=float(errors[-120:].mean()),
                    drift=float(np.linalg.norm(positions[-1]-[0,0,20])),drift_10s=float(np.linalg.norm(positions[1199]-[0,0,20])),
                    settling_time=None if not len(candidates) else float(candidates[0]*dt-start),overshoot=max(overshoots),max_attitude=max(attitudes)))
            finally: p.disconnect(physicsClientId=client)
    out=Path('outputs/dynamics');out.mkdir(parents=True,exist_ok=True)
    for name,data in [('wind_comparison',summaries),('wind_traces',rows)]:
        with (out/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    (out/'wind_comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
