"""Headless rendering and geometry regression checks, not subjective GUI checks."""
import numpy as np
import pytest
pytest.importorskip('pybullet')
pytest.importorskip('gymnasium')
import pybullet as p
from config import EnvConfig, DroneConfig
from env import MultiDroneEnv
from env.drone import Drone
from main import scripted_policy


@pytest.mark.parametrize('scenario',['urban','open_field','cluttered'])
@pytest.mark.parametrize('fleet',[2,8,20])
def test_render_initialization_and_transform(scenario,fleet):
    with MultiDroneEnv(EnvConfig(scenario=scenario,num_drones=fleet,gui=False)) as env:
        env.cfg.obstacles.count=2
        env.reset(seed=42)
        for _ in range(5): env.step(scripted_policy(env))
        drone=env.drones['uav_0']
        pose=p.getBasePositionAndOrientation(drone.body_id,physicsClientId=env.client)
        state=drone.physical_state()
        assert np.allclose(pose[0],state[:3]) and np.allclose(pose[1],state[6:10])
        visual=p.getVisualShapeData(drone.body_id,physicsClientId=env.client)
        assert len(visual)==6 and all(v[0]==drone.body_id and v[1]==-1 for v in visual)
        for v in visual:
            world=p.multiplyTransforms(pose[0],pose[1],v[5],v[6])
            rotation=np.array(p.getMatrixFromQuaternion(pose[1])).reshape(3,3)
            assert np.allclose(world[0],state[:3]+rotation@np.array(v[5]),atol=1e-5)
        view=p.computeViewMatrixFromYawPitchRoll([0,0,12],env.town.side*.85,35,-62,0,2)
        projection=p.computeProjectionMatrixFOV(60,4/3,.1,500)
        image=p.getCameraImage(160,120,view,projection,renderer=p.ER_TINY_RENDERER,physicsClientId=env.client)
        rgba=np.asarray(image[2]).reshape(120,160,4)
        assert rgba[:,:,:3].std()>1
        assert len(env.moving_obstacles.items)==2
        assert env.source.center[0]<env.destination.center[0]


def test_visual_scaling_preserves_collision_and_inertia():
    client=p.connect(p.DIRECT)
    try:
        a=Drone(client,[0,0,10],[1,0,0,1],DroneConfig(visual_scale=1))
        b=Drone(client,[10,0,10],[1,0,0,1],DroneConfig(visual_scale=8))
        assert p.getCollisionShapeData(a.body_id,-1,physicsClientId=client)[0][3]==p.getCollisionShapeData(b.body_id,-1,physicsClientId=client)[0][3]
        assert p.getDynamicsInfo(a.body_id,-1,physicsClientId=client)[2]==p.getDynamicsInfo(b.body_id,-1,physicsClientId=client)[2]
    finally: p.disconnect(physicsClientId=client)
