"""Nonlinear rigid-body plant integrated by PyBullet; no flight controller.

World axes are z-up. Collective thrust acts along body +z; torque is body-frame.
Newton-Euler dynamics: m*v_dot = m*g + R*[0,0,T] + F_external,
I*omega_dot + omega cross (I*omega) = torque (body-frame omega).
Physical mass distribution is approximated by a uniform collision sphere.
"""
import numpy as np
import pybullet as p


from .environment_config import DroneConfig


class Drone:
    """Rigid-body quadrotor plant with independent enlarged display geometry."""
    def __init__(self, client_id, position, color, config=None):
        self.client_id = client_id
        self.cfg = config or DroneConfig()
        self.cfg.validate()
        c = self.cfg
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError('position must contain three finite coordinates')
        p.setGravity(0, 0, -c.gravity, physicsClientId=client_id)
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=c.radius,
                                            physicsClientId=client_id)
        # Enlarge the visible drone independently from its collision sphere.
        # This changes rendering only, leaving navigation dynamics and the
        # existing MARL collision model exactly as before.
        scale = c.visual_scale
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('visual_scale must be a finite positive number')
        arms = [([0.60, .055, .035], [0, 0, 0.03]),
                ([.055, 0.60, .035], [0, 0, 0.03])]
        shape_types = [p.GEOM_BOX, p.GEOM_BOX] + [p.GEOM_CYLINDER]*4
        half_extents = [[scale * v for v in dimensions] for dimensions, _ in arms] + [[0, 0, 0]]*4
        lengths = [0., 0.] + [scale*.045]*4
        radii = [0., 0.] + [scale*.16]*4
        positions = [[scale * v for v in offset] for _, offset in arms] + [
            [scale*x, scale*y, scale*.08] for x,y in
            [(-.55,0),(.55,0),(0,-.55),(0,.55)]]
        # Bright, contrasting arms and rotor disks remain visible when the
        # camera is zoomed out to show the entire town.
        colors = [list(color), [.15, .19, .25, 1]] + [
            [.98, .71, .16, 1], [.98, .71, .16, 1],
            [.07, .80, .95, 1], [.07, .80, .95, 1]]
        visuals = p.createVisualShapeArray(shapeTypes=shape_types,
                          halfExtents=half_extents, lengths=lengths, radii=radii,
                          visualFramePositions=positions, rgbaColors=colors,
                          physicsClientId=client_id)
        self.body_id = p.createMultiBody(baseMass=c.mass, baseCollisionShapeIndex=collision,
                        baseVisualShapeIndex=visuals, basePosition=position,
                        physicsClientId=client_id)
        p.changeDynamics(self.body_id, -1, localInertiaDiagonal=c.inertia_diagonal,
                         linearDamping=0., angularDamping=0., restitution=0.,
                         physicsClientId=client_id)
        self.collision_bodies = set()

    def physical_state(self):
        """13 values: world position/velocity, xyzw quaternion, world angular velocity."""
        position, quaternion = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        velocity, omega = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return np.asarray([*position, *velocity, *quaternion, *omega], dtype=float)

    def state(self):
        """Compatibility tuple: position, velocity, Euler angles, world omega, xyzw."""
        value = self.physical_state()
        return (value[:3].copy(), value[3:6].copy(),
                np.asarray(p.getEulerFromQuaternion(value[6:10])),
                value[10:13].copy(), value[6:10].copy())

    def apply_control(self, thrust, body_torques):
        """Queue body +Z thrust (N) and right-handed body torques (Nm).

        R maps body coordinates to the right-handed, Z-up world frame.
        PyBullet reports angular velocity in world coordinates; the omega in
        R_dot = R*skew(omega) is R.T @ omega_world. Forces are applied at the
        centre of mass, so collective thrust introduces no extra moment.
        Call before every fixed physics step; PyBullet clears applied wrenches
        after stepping. This method neither steps nor resets the rigid body.
        """
        thrust = np.asarray(thrust, dtype=float)
        torques = np.asarray(body_torques, dtype=float)
        if (thrust.shape != () or not np.isfinite(thrust) or thrust < 0
                or torques.shape != (3,) or not np.all(np.isfinite(torques))):
            raise ValueError('Require a finite nonnegative scalar thrust and three finite body torques')
        state = self.physical_state()
        rotation = np.asarray(p.getMatrixFromQuaternion(state[6:10])).reshape(3, 3)
        p.applyExternalForce(self.body_id, -1, (rotation[:, 2]*float(thrust)).tolist(),
                             state[:3].tolist(), p.WORLD_FRAME, physicsClientId=self.client_id)
        p.applyExternalTorque(self.body_id, -1, (rotation@torques).tolist(),
                              p.WORLD_FRAME, physicsClientId=self.client_id)

    def apply_wrench(self, thrust, torque, external_force=None):
        """Apply collective thrust (N), body torque (Nm), optional world force (N).

        Call once per physics step before the environment steps PyBullet.
        This is an actuator/plant interface, not a velocity flight controller.
        """
        torque = np.asarray(torque, dtype=float)
        if not np.isfinite(thrust) or thrust < 0 or torque.shape != (3,) or not np.all(np.isfinite(torque)):
            raise ValueError('Require finite nonnegative thrust and three finite torques')
        force = np.zeros(3) if external_force is None else np.asarray(external_force, dtype=float)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError('external_force must be a finite 3-vector')
        self.apply_control(thrust, torque)
        if np.any(force):
            p.applyExternalForce(self.body_id, -1, force.tolist(), self.physical_state()[:3].tolist(),
                                 p.WORLD_FRAME, physicsClientId=self.client_id)

    def control(self, action, dt, wind=None):
        """Velocity controller intentionally deferred; do not silently ignore actions."""
        raise NotImplementedError('Rigid-body plant requires a desired-velocity flight controller')

    def _swept_collisions(self, start, end):
        """Conservative sphere sweep against expanded obstacle AABBs.

        This prevents thin-obstacle tunnelling, at the cost of false positives
        near curved shapes and compound-body AABB corners.
        """
        radius = self.cfg.radius
        candidates = p.getOverlappingObjects((np.minimum(start, end)-radius).tolist(),
                                             (np.maximum(start, end)+radius).tolist(),
                                             physicsClientId=self.client_id) or []
        hits = set()
        delta = end-start
        for body, _ in candidates:
            if body == self.body_id or not p.getCollisionShapeData(body, -1, physicsClientId=self.client_id):
                continue
            low, high = p.getAABB(body, physicsClientId=self.client_id)
            low, high = np.asarray(low)-radius, np.asarray(high)+radius
            enter, leave = 0., 1.
            for axis in range(3):
                if abs(delta[axis]) < 1e-12:
                    if start[axis] < low[axis] or start[axis] > high[axis]:
                        leave = -1.
                        break
                else:
                    a, b = sorted(((low[axis]-start[axis])/delta[axis],
                                   (high[axis]-start[axis])/delta[axis]))
                    enter, leave = max(enter, a), min(leave, b)
            if enter <= leave:
                hits.add(body)
        return hits
