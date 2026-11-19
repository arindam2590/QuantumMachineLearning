import numpy as np
import pytest
pytest.importorskip('pybullet')
import pybullet as p
from config import DroneConfig
from env.drone import Drone
from env.drone_controller import HoverController, body_rotation_error


@pytest.mark.parametrize('angles',[(.01,-.01,.02),(.15,-.12,.25)])
def test_hover_recovery(angles):
    client=p.connect(p.DIRECT)
    try:
        p.setTimeStep(1/240,physicsClientId=client)
        cfg=DroneConfig()
        drone=Drone(client,[0,0,10],[.3,.7,.9,1],cfg)
        p.resetBasePositionAndOrientation(drone.body_id,[0,0,10],p.getQuaternionFromEuler(angles),physicsClientId=client)
        controller=HoverController(cfg,10)
        for _ in range(2400):
            thrust,torque=controller.apply(drone)
            assert 0<=thrust<=cfg.max_thrust_factor*cfg.mass*cfg.gravity
            assert np.max(np.abs(torque))<=cfg.max_torque
            p.stepSimulation(physicsClientId=client)
        state=drone.physical_state()
        assert abs(state[2]-10)<.01
        assert abs(state[5])<.01
        assert np.linalg.norm(body_rotation_error(state[6:10],[0,0,0,1]))<.001
        assert np.linalg.norm(state[10:])<.001
    finally:
        p.disconnect(physicsClientId=client)


def test_sign_invariance_and_yaw_target():
    cfg=DroneConfig(yaw_target=.7)
    controller=HoverController(cfg,10)
    q=np.array(p.getQuaternionFromEuler([.1,.2,.3]))
    target=p.getQuaternionFromEuler([0,0,.7])
    assert np.allclose(body_rotation_error(q,target),body_rotation_error(-q,target))
    state=np.array([0,0,10,0,0,0,*target,0,0,0])
    thrust,torque=controller.compute(state)
    assert thrust==pytest.approx(cfg.mass*cfg.gravity)
    assert np.allclose(torque,0,atol=1e-10)


def test_saturation_and_rate_damping():
    cfg=DroneConfig()
    controller=HoverController(cfg,10)
    state=np.array([0,0,0,0,0,0,0,0,0,1,100,0,0],dtype=float)
    thrust,torque=controller.compute(state)
    assert thrust==pytest.approx(cfg.max_thrust_factor*cfg.mass*cfg.gravity)
    assert torque[0]==-cfg.max_torque


@pytest.mark.parametrize('command',[[2,0,0],[0,0,1],[0,0,-1],[0,0,0]])
def test_velocity_tracking(command):
    from env.drone_controller import VelocityController
    client=p.connect(p.DIRECT)
    try:
        dt=1/240
        p.setTimeStep(dt,physicsClientId=client)
        cfg=DroneConfig()
        drone=Drone(client,[0,0,20],[1,0,0,1],cfg)
        controller=VelocityController(cfg)
        for _ in range(2400):
            controller.apply(drone,command,dt)
            assert np.linalg.norm(controller.desired_acceleration[:2])<=cfg.max_accel_xy+1e-9
            assert abs(controller.desired_acceleration[2])<=cfg.max_accel_z+1e-9
            p.stepSimulation(physicsClientId=client)
        assert np.allclose(drone.physical_state()[3:6],command,atol=.08)
        if command==[0,0,0]:
            assert np.allclose(drone.physical_state()[:3],[0,0,20],atol=.01)
        # Reverse command, then brake without resetting pose or velocity.
        for target in [-np.array(command),np.zeros(3)]:
            for _ in range(2400):
                controller.apply(drone,target,dt)
                p.stepSimulation(physicsClientId=client)
            assert np.allclose(drone.physical_state()[3:6],target,atol=.08)
    finally:
        p.disconnect(physicsClientId=client)


def test_command_actuator_tilt_and_jerk_limits():
    from env.drone_controller import VelocityController, attitude_torque
    cfg=DroneConfig(max_roll=.1,max_pitch=.15,max_collective_thrust=8.,
                    max_body_torques=(.1,.2,.3),max_jerk=2.)
    controller=VelocityController(cfg)
    state=np.array([0,0,10,0,0,0,0,0,0,1,0,0,0],dtype=float)
    previous=np.zeros(3)
    for _ in range(100):
        thrust,torque=controller.compute(state,[100,100,100],.01)
        assert thrust<=8 and np.all(np.abs(torque)<=np.array([.1,.2,.3]))
        assert np.linalg.norm(controller.limited_velocity_command[:2])<=cfg.max_speed_xy+1e-9
        assert controller.limited_velocity_command[2]<=cfg.max_speed_z
        assert np.linalg.norm(controller.desired_acceleration-previous)<=.02000001
        previous=controller.desired_acceleration.copy()
        angles=p.getEulerFromQuaternion(__import__('env.drone_controller',fromlist=['rotation_quaternion']).rotation_quaternion(controller.desired_rotation))
        assert abs(angles[0])<=cfg.max_roll+1e-8 and abs(angles[1])<=cfg.max_pitch+1e-8
    controller.reset()
    assert np.allclose(controller.desired_acceleration,0)
    inertia=np.asarray(cfg.inertia_diagonal)
    torque=attitude_torque(cfg,np.ones(3)*100,np.zeros(3))
    inferred_rate=torque/(inertia*cfg.attitude_kd)
    assert np.all(inferred_rate<=np.array([cfg.max_angular_rate,cfg.max_angular_rate,cfg.max_yaw_rate]))


def test_wind_force_and_controller_response():
    from env.drone_controller import VelocityController
    client=p.connect(p.DIRECT)
    try:
        dt=1/240
        p.setTimeStep(dt,physicsClientId=client)
        cfg=DroneConfig()
        drone=Drone(client,[0,0,20],[1,0,0,1],cfg)
        controller=VelocityController(cfg)
        for _ in range(2400):
            thrust,torque=controller.compute(drone.physical_state(),[0,0,0],dt)
            drone.apply_wrench(thrust,torque,external_force=[cfg.mass*.2,0,0])
            p.stepSimulation(physicsClientId=client)
        state=drone.physical_state()
        assert np.isfinite(state).all()
        assert 0<state[3]<.3  # PD has a finite steady velocity error under constant wind.
        assert abs(state[5])<.1
    finally:
        p.disconnect(physicsClientId=client)


def test_integral_reset_limits_and_antiwindup():
    from env.drone_controller import VelocityController
    cfg=DroneConfig(velocity_mode='pid',velocity_integral_limit=.2)
    controller=VelocityController(cfg)
    state=np.array([0,0,20,0,0,0,0,0,0,1,0,0,0],dtype=float)
    for _ in range(100): controller.compute(state,[.1,0,0],.01)
    assert 0<controller.integral_error[0]<=.2
    controller.reset()
    assert np.allclose(controller.integral_error,0) and controller.previous_velocity is None
    cfg.max_collective_thrust=1.
    for _ in range(100): controller.compute(state,[0,0,2],.01)
    assert abs(controller.integral_error[2])<1e-8
    cfg.max_collective_thrust=None
    cfg.max_roll=cfg.max_pitch=.01
    for _ in range(100): controller.compute(state,[5,0,0],.01)
    assert abs(controller.integral_error[0])<1e-8


def test_specialized_cross_preserves_outputs():
    from env.drone_controller import cross3
    rng=np.random.default_rng(42)
    for _ in range(100):
        a,b=rng.normal(size=(2,3))
        assert np.array_equal(cross3(a,b),np.cross(a,b))
