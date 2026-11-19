"""Rigid-body plant tests; velocity controller is intentionally not implemented."""
import numpy as np
import pytest
from config import DroneConfig
pytest.importorskip('pybullet')
import pybullet as p
from env.drone import Drone


@pytest.fixture
def client():
    connection=p.connect(p.DIRECT)
    p.setTimeStep(1/120,physicsClientId=connection)
    yield connection
    p.disconnect(physicsClientId=connection)


def test_initialization_state_and_inertia(client):
    cfg=DroneConfig()
    drone=Drone(client,[1,2,10],[.3,.7,.9,1],cfg)
    state=drone.physical_state()
    assert state.shape==(13,)
    assert np.allclose(state[:3],[1,2,10])
    assert np.allclose(state[3:6],0) and np.allclose(state[10:],0)
    assert np.linalg.norm(state[6:10])==pytest.approx(1)
    dynamics=p.getDynamicsInfo(drone.body_id,-1,physicsClientId=client)
    assert dynamics[0]==cfg.mass
    assert np.allclose(dynamics[2],cfg.inertia_diagonal)


def test_gravity_and_no_teleportation(client):
    drone=Drone(client,[0,0,10],[1,0,0,1])
    for _ in range(12):
        p.stepSimulation(physicsClientId=client)
    state=drone.physical_state()
    assert state[5]==pytest.approx(-.981,abs=1e-6)
    assert 9.94<state[2]<10
    assert np.linalg.norm(state[6:10])==pytest.approx(1)
    with pytest.raises(NotImplementedError):
        drone.control(np.zeros(3),1/120)


def test_balanced_thrust_is_plant_input_not_controller(client):
    cfg=DroneConfig()
    drone=Drone(client,[0,0,10],[1,0,0,1],cfg)
    for _ in range(12):
        drone.apply_wrench(cfg.mass*cfg.gravity,[0,0,0])
        p.stepSimulation(physicsClientId=client)
    assert np.allclose(drone.physical_state()[:6],[0,0,10,0,0,0],atol=1e-8)


@pytest.mark.parametrize('kwargs',[{'mass':0},{'mass':-1},{'radius':float('nan')},{'gravity':0}])
def test_invalid_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        DroneConfig(**kwargs).validate()


def test_visual_scale_independent_physics():
    a,b=DroneConfig(),DroneConfig(visual_scale=8)
    assert a.radius==b.radius and a.inertia_diagonal==b.inertia_diagonal


def test_tilted_thrust_direction(client):
    drone=Drone(client,[0,0,10],[1,0,0,1])
    quaternion=p.getQuaternionFromEuler([0,np.pi/2,0])
    # Pose reset is test initialization only, never a flight update.
    p.resetBasePositionAndOrientation(drone.body_id,[0,0,10],quaternion,physicsClientId=client)
    drone.apply_control(drone.cfg.mass*2.,[0,0,0])
    assert np.allclose(drone.physical_state()[:3],[0,0,10])
    p.stepSimulation(physicsClientId=client)
    velocity=drone.physical_state()[3:6]
    assert np.allclose(velocity,[2/120,0,-drone.cfg.gravity/120],atol=1e-8)
    assert np.allclose(drone.physical_state()[10:],0,atol=1e-8)


@pytest.mark.parametrize('axis',[0,1,2])
def test_body_torque_response(client,axis):
    drone=Drone(client,[0,0,10],[1,0,0,1])
    quaternion=p.getQuaternionFromEuler([.2,-.3,.7])
    p.resetBasePositionAndOrientation(drone.body_id,[0,0,10],quaternion,physicsClientId=client)
    torque=np.zeros(3)
    torque[axis]=.1
    rotation=np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
    drone.apply_control(0,torque)
    p.stepSimulation(physicsClientId=client)
    expected=rotation@(torque/np.array(drone.cfg.inertia_diagonal))/120
    assert np.allclose(drone.physical_state()[10:],expected,atol=1e-8)
    assert np.linalg.norm(drone.physical_state()[6:10])==pytest.approx(1)


def test_force_free_uniform_motion(client):
    drone=Drone(client,[0,0,10],[1,0,0,1])
    p.setGravity(0,0,0,physicsClientId=client)
    p.resetBaseVelocity(drone.body_id,[1,2,3],[0,0,0],physicsClientId=client)
    for _ in range(12):
        drone.apply_control(0,[0,0,0])
        p.stepSimulation(physicsClientId=client)
    assert np.allclose(drone.physical_state()[:6],[.1,.2,10.3,1,2,3],atol=1e-8)


@pytest.mark.parametrize('thrust,torque',[(-1,[0,0,0]),(float('nan'),[0,0,0]),
                                         (1,[0,0]),(1,[0,float('inf'),0])])
def test_invalid_controls(client,thrust,torque):
    drone=Drone(client,[0,0,10],[1,0,0,1])
    with pytest.raises(ValueError):
        drone.apply_control(thrust,torque)
