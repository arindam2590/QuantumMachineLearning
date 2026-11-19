"""Real PyBullet sensor, obstacle and battery integration."""
import numpy as np
import pytest
pytest.importorskip('pybullet')
pytest.importorskip('gymnasium')
import pybullet as p
from config import EnvConfig
from env import MultiDroneEnv


def test_geometry_sensors_obstacles_and_battery():
    cfg = EnvConfig(scenario='open_field',num_drones=2,gui=False)
    cfg.obstacles.count=2
    cfg.sensors.camera_enabled=True
    cfg.sensors.lidar_vertical_angles=(0.,)
    cfg.sensors.lidar_rays=4
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        original=[item['points'].copy() for item in env.moving_obstacles.items]
        readings=env.get_sensor_readings('uav_0')
        assert readings['imu']['value'].shape==(6,)
        assert readings['gps']['value'].shape==(3,)
        assert readings['camera']['value']['rgb'].shape==(64,64,3)
        depth=readings['camera']['value']['depth_m']
        assert depth.shape==(64,64) and np.isfinite(depth).all()
        d=env.drones['uav_0']
        position=d.state()[0]
        collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.5,.5,.5],physicsClientId=env.client)
        p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision,basePosition=position+[3,0,0],physicsClientId=env.client)
        lidar=env.sensors['uav_0']['lidar']
        lidar.update(.1,position,[0,0,0,1])
        assert lidar.read()['value']['distance_m'].min()<3
        before=[np.array(p.getBasePositionAndOrientation(item['body'],physicsClientId=env.client)[0]) for item in env.moving_obstacles.items]
        env.step({a:np.zeros(3) for a in env.agents})
        assert all(d.battery<1 for d in env.drones.values())
        assert any(not np.array_equal(pos,p.getBasePositionAndOrientation(item['body'],physicsClientId=env.client)[0]) for pos,item in zip(before,env.moving_obstacles.items))
        env.reset(seed=42)
        assert all(np.array_equal(points,item['points']) for points,item in zip(original,env.moving_obstacles.items))


def test_real_swept_building_and_uav_collisions():
    cfg=EnvConfig(scenario='open_field',num_drones=2,gui=False)
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        a,b=env.drones.values()
        a._position=np.array([0.,0.,10.])
        a._velocity=np.array([5.,0.,0.])
        b._position=np.array([2.,0.,10.])
        p.resetBasePositionAndOrientation(a.body_id,a._position,[0,0,0,1],physicsClientId=env.client)
        p.resetBasePositionAndOrientation(b.body_id,b._position,[0,0,0,1],physicsClientId=env.client)
        assert b.body_id in a._swept_collisions(a._position,np.array([5.,0.,10.]))
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.01,1,1],physicsClientId=env.client)
        wall=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[1,0,10],physicsClientId=env.client)
        assert wall in a._swept_collisions(a._position,np.array([5.,0.,10.]))


def test_battery_termination_real_environment():
    cfg=EnvConfig(scenario='open_field',num_drones=2,gui=False)
    cfg.battery.initial_charge=.000001
    cfg.battery.hover_rate=1.
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        _,rewards,term,trunc,infos=env.step({a:np.zeros(3) for a in env.agents})
        assert term['__all__'] and not trunc['__all__'] and not env.agents
        assert all(info['battery_depleted'] for info in infos.values())
        assert all(np.isclose(rewards[a],sum(info['reward_components'].values())) for a,info in infos.items())


def test_source_to_destination_motion_without_teleport():
    from main import demo_policy
    cfg=EnvConfig(scenario='open_field',num_drones=2,gui=False,max_episode_steps=400)
    cfg.source.center=(-3.,0.,12.)
    cfg.source.half_size=(1.,3.,3.)
    cfg.destination.center=(3.,0.,12.)
    cfg.destination.half_size=(1.,3.,3.)
    with MultiDroneEnv(cfg) as env:
        obs,_=env.reset(seed=42)
        start={a:d.state()[0].copy() for a,d in env.drones.items()}
        while env.agents:
            obs,_,_,_,_=env.step(demo_policy(obs,env))
        assert env.episode_statistics['team_success']
        assert all(np.linalg.norm(d.state()[0]-start[a])>3 for a,d in env.drones.items())
