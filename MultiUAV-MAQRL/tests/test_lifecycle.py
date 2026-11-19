import numpy as np
import pytest
pytest.importorskip('pybullet')
pytest.importorskip('gymnasium')
from config import EnvConfig
from env import MultiDroneEnv
import pybullet as p


def actions(env):
    return {a: np.zeros(3, dtype=np.float32) for a in env.agents}


@pytest.mark.parametrize('count', [2,4,8,16,20])
def test_steps_truncation_reset_close(count):
    env = MultiDroneEnv(EnvConfig(scenario='open_field', num_drones=count, gui=False, max_episode_steps=2))
    try:
        with pytest.raises(RuntimeError):
            env.step({})
        first, _ = env.reset(seed=42)
        for i in range(2):
            active = set(env.agents)
            obs, rewards, term, trunc, infos = env.step(actions(env))
            assert set(obs) == set(rewards) == set(infos) == active
            assert set(term)-{'__all__'} == set(trunc)-{'__all__'} == active
        assert not env.agents and trunc['__all__']
        assert not infos['uav_0']['episode']['team_success']
        again, _ = env.reset(seed=42)
        assert all(np.array_equal(first[a],again[a]) for a in first)
        assert env.step_count == 0
    finally:
        env.close()
        env.close()
    assert not env.agents


def test_individual_arrival_and_team_success():
    with MultiDroneEnv(EnvConfig(scenario='open_field', num_drones=2, gui=False)) as env:
        env.reset(seed=42)
        drone = env.drones['uav_0']
        p.resetBasePositionAndOrientation(drone.body_id, env.destination.center, [0,0,0,1], physicsClientId=env.client)
        obs, rewards, term, trunc, infos = env.step(actions(env))
        assert term['uav_0'] and env.agents == ['uav_1']
        assert not infos['uav_0']['episode']['team_success']
        with pytest.raises(KeyError):
            env.step({'uav_0':np.zeros(3), 'uav_1':np.zeros(3)})
        p.resetBasePositionAndOrientation(env.drones['uav_1'].body_id, env.destination.center, [0,0,0,1], physicsClientId=env.client)
        _, _, term, _, infos = env.step(actions(env))
        assert not env.agents and term['__all__']
        assert infos['uav_1']['episode']['team_success']


def test_control_rates_and_rigid_body_feedback():
    from unittest.mock import patch
    cfg=EnvConfig(scenario='open_field', num_drones=2, gui=False, low_level_hz=60)
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        controller=env.controllers['uav_0']
        before=env.drones['uav_0'].physical_state().copy()
        with patch.object(controller,'compute',wraps=controller.compute) as compute:
            env.step(actions(env))
            assert compute.call_count==3  # 20 Hz policy, 60 Hz controller, 120 Hz physics
        assert env._physics_tick==6
        assert np.allclose(env.drones['uav_0'].physical_state()[:6],before[:6],atol=1e-6)
        assert env.get_sensor_readings('uav_0')['imu']['timestamp']>0


@pytest.mark.parametrize('collision_kind',['uav','building'])
def test_physical_collision_termination(collision_kind):
    cfg=EnvConfig(scenario='open_field',num_drones=2,gui=False)
    with MultiDroneEnv(cfg) as env:
        env.reset(seed=42)
        location=[0,0,10]
        p.resetBasePositionAndOrientation(env.drones['uav_0'].body_id,location,[0,0,0,1],physicsClientId=env.client)
        if collision_kind=='uav':
            p.resetBasePositionAndOrientation(env.drones['uav_1'].body_id,[.9,0,10],[0,0,0,1],physicsClientId=env.client)
        else:
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.5,.5,2],physicsClientId=env.client)
            p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=[.9,0,10],physicsClientId=env.client)
        _,_,terminated,_,infos=env.step(actions(env))
        assert terminated['uav_0'] and infos['uav_0']['collision']
        assert 'uav_0' not in env.agents
