"""Quaternion PD attitude and altitude hover controller; no navigation policy."""
import numpy as np
import pybullet as p


def cross3(a, b):
    """Three-vector cross product without general axis/broadcast dispatch."""
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])


def body_rotation_error(quaternion, target):
    """Shortest rotation from current to target, expressed in current body axes.

    Quaternions use xyzw and map body vectors into world coordinates.
    q_error = inverse(q_current) * q_target; q and -q are equivalent.
    """
    current, desired = np.asarray(quaternion, dtype=float), np.asarray(target, dtype=float)
    for value in (current, desired):
        if value.shape != (4,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) < 1e-12:
            raise ValueError('Require finite nonzero xyzw quaternions')
    current = current/np.linalg.norm(current)
    desired = desired/np.linalg.norm(desired)
    a, b = -current[:3], desired[:3]
    vector = current[3]*b+desired[3]*a+cross3(a,b)
    scalar = current[3]*desired[3]-np.dot(a,b)
    if scalar < 0:
        vector, scalar = -vector, -scalar
    length = np.linalg.norm(vector)
    return 2*vector if length < 1e-10 else vector*(2*np.arctan2(length,scalar)/length)


def thrust_limit(config):
    return (config.max_collective_thrust if config.max_collective_thrust is not None
            else config.mass*config.gravity*config.max_thrust_factor)


def attitude_torque(config, error, omega_body):
    """Bound body-rate references; do not overwrite physical angular velocity."""
    limits = np.array([config.max_angular_rate,config.max_angular_rate,config.max_yaw_rate])
    rate_target = np.clip(config.attitude_kp/config.attitude_kd*error,-limits,limits)
    torque = np.asarray(config.inertia_diagonal)*config.attitude_kd*(rate_target-omega_body)
    torque_limits = config.max_body_torques if config.max_body_torques is not None else config.max_torque
    return np.clip(torque,-np.asarray(torque_limits),np.asarray(torque_limits))


class HoverController:
    """Outputs collective N and body-frame Nm for a fixed altitude/yaw target.

    Stabilizes roll/pitch/yaw and vertical motion; does not hold horizontal
    position or track desired velocities. Call compute/apply once per fixed
    physics step before p.stepSimulation().
    """
    def __init__(self, config, altitude_target, yaw_target=None):
        config.validate()
        self.config = config
        self.altitude_target = float(altitude_target)
        self.yaw_target = config.yaw_target if yaw_target is None else float(yaw_target)
        if not np.isfinite(self.altitude_target) or not config.min_altitude <= self.altitude_target <= config.max_altitude:
            raise ValueError('Hover altitude must lie within flight bounds')
        if not np.isfinite(self.yaw_target):
            raise ValueError('yaw_target must be finite')

    def compute(self, physical_state):
        state = np.asarray(physical_state, dtype=float)
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise ValueError('Expected finite 13-element rigid-body state')
        c = self.config
        quaternion = state[6:10]
        target = p.getQuaternionFromEuler([0.,0.,self.yaw_target])
        error = body_rotation_error(quaternion,target)
        quaternion = quaternion/np.linalg.norm(quaternion)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
        omega_body = rotation.T@state[10:13]
        # Gains specify angular acceleration feedback; inertia converts to Nm.
        torque = attitude_torque(c,error,omega_body)
        acceleration = c.hover_altitude_kp*(self.altitude_target-state[2])-c.hover_vertical_kd*state[5]
        thrust = c.mass*(c.gravity+acceleration)/max(c.min_thrust_projection,rotation[2,2])
        thrust = float(np.clip(thrust,0.,thrust_limit(c)))
        return thrust, torque

    def apply(self, drone):
        thrust, torque = self.compute(drone.physical_state())
        drone.apply_control(thrust,torque)
        return thrust, torque


class VelocityController:
    """Cascaded world-velocity PD and body-attitude PD controller.

    Optional bounded integral with conditional anti-windup. Zero velocity brakes toward hover,
    without holding an exact position. Command limits constrain references;
    actual acceleration also depends on attitude lag and external forces.
    """
    def __init__(self, config, yaw_target=None):
        config.validate()
        self.config = config
        self.yaw_target = config.yaw_target if yaw_target is None else float(yaw_target)
        if not np.isfinite(self.yaw_target):
            raise ValueError('yaw_target must be finite')
        self.integral_error = np.zeros(3)
        self.previous_velocity = None
        self.desired_acceleration = np.zeros(3)

    def reset(self):
        self.integral_error = np.zeros(3)
        self.previous_velocity = None
        self.desired_acceleration[:] = 0.

    def compute(self, physical_state, desired_velocity, dt):
        c = self.config
        state = np.asarray(physical_state, dtype=float)
        command = np.asarray(desired_velocity, dtype=float).copy()
        if state.shape != (13,) or not np.all(np.isfinite(state)):
            raise ValueError('Expected finite 13-element physical state')
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError('Expected finite desired 3D velocity')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        horizontal = np.linalg.norm(command[:2])
        if horizontal > c.max_speed_xy:
            command[:2] *= c.max_speed_xy/horizontal
        command[2] = np.clip(command[2], -c.max_speed_z, c.max_speed_z)
        self.limited_velocity_command = command.copy()
        velocity = state[3:6]
        acceleration_feedback = (np.zeros(3) if self.previous_velocity is None else
                                 (velocity-self.previous_velocity)/dt)
        velocity_error = command-velocity
        previous_integral = self.integral_error.copy()
        if c.velocity_mode == 'pid' and np.linalg.norm(velocity_error) <= c.velocity_integral_error_gate:
            self.integral_error = np.clip(self.integral_error+velocity_error*dt,
                                          -c.velocity_integral_limit,c.velocity_integral_limit)
        acceleration = c.velocity_kp*velocity_error-c.velocity_kd*acceleration_feedback
        if c.velocity_mode == 'pid':
            acceleration += c.velocity_ki*self.integral_error
        unconstrained_acceleration = acceleration.copy()
        self.previous_velocity = velocity.copy()
        norm = np.linalg.norm(acceleration[:2])
        if norm > c.max_accel_xy:
            acceleration[:2] *= c.max_accel_xy/norm
        acceleration[2] = np.clip(acceleration[2],-c.max_accel_z,c.max_accel_z)
        if c.max_jerk is not None:
            delta = acceleration-self.desired_acceleration
            norm = np.linalg.norm(delta)
            if norm > c.max_jerk*dt:
                acceleration = self.desired_acceleration+delta*(c.max_jerk*dt/norm)
        self.desired_acceleration = acceleration.copy()
        specific_thrust = acceleration+np.array([0.,0.,c.gravity])
        # Keep upward thrust feasible, even for extreme descent requests.
        specific_thrust[2] = max(specific_thrust[2],c.minimum_specific_thrust)
        horizontal = np.linalg.norm(specific_thrust[:2])
        max_horizontal = specific_thrust[2]*np.tan(min(c.max_tilt,c.max_roll,c.max_pitch))
        if horizontal > max_horizontal:
            specific_thrust[:2] *= max_horizontal/horizontal
        magnitude = np.linalg.norm(specific_thrust)
        z_axis = (np.array([0.,0.,1.]) if magnitude < c.minimum_specific_thrust else specific_thrust/magnitude)
        # Desired body X aligns with the yaw heading projected onto the thrust plane.
        heading = np.array([np.cos(self.yaw_target),np.sin(self.yaw_target),0.])
        y_axis = cross3(z_axis,heading)
        y_axis /= max(np.linalg.norm(y_axis),1e-12)
        x_axis = cross3(y_axis,z_axis)
        desired_rotation = np.column_stack([x_axis,y_axis,z_axis])
        self.desired_rotation = desired_rotation.copy()
        quaternion = state[6:10]
        if np.linalg.norm(quaternion) < 1e-12:
            raise ValueError('Invalid quaternion')
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion/np.linalg.norm(quaternion))).reshape(3,3)
        # Quaternion conversion through PyBullet's matrix-equivalent Euler extraction
        # is avoided: construct quaternion directly from an orthonormal matrix.
        target = rotation_quaternion(desired_rotation)
        error = body_rotation_error(quaternion,target)
        omega_body = rotation.T@state[10:13]
        torque = attitude_torque(c,error,omega_body)
        thrust = c.mass*float(np.dot(specific_thrust,rotation[:,2]))
        raw_thrust = thrust
        thrust = float(np.clip(thrust,0.,thrust_limit(c)))
        if c.velocity_mode == 'pid':
            # Reject integration that drives a constrained acceleration axis further
            # into saturation. Include tilt, jerk and thrust feasibility.
            feasible = specific_thrust-np.array([0.,0.,c.gravity])
            blocked = (unconstrained_acceleration-feasible)*velocity_error > 1e-10
            if abs(raw_thrust-thrust) > 1e-9:
                blocked |= velocity_error*(unconstrained_acceleration) > 0
            torque_limits = np.asarray(c.max_body_torques if c.max_body_torques is not None else c.max_torque)
            if np.any(np.abs(torque) >= torque_limits-1e-9):
                blocked[:] = True
            self.integral_error[blocked] = previous_integral[blocked]
        return thrust,torque

    def apply(self, drone, desired_velocity, dt):
        thrust,torque = self.compute(drone.physical_state(),desired_velocity,dt)
        drone.apply_control(thrust,torque)
        return thrust,torque


def rotation_quaternion(matrix):
    """Convert an orthonormal body-to-world rotation matrix to xyzw."""
    # Stable largest-component construction, including near-pi rotations.
    m = np.asarray(matrix)
    candidates = np.array([1+m[0,0]-m[1,1]-m[2,2],
                           1-m[0,0]+m[1,1]-m[2,2],
                           1-m[0,0]-m[1,1]+m[2,2],1+np.trace(m)])
    index = int(np.argmax(candidates))
    q = np.zeros(4)
    q[index] = .5*np.sqrt(max(0.,candidates[index]))
    divisor = 4*q[index]
    if index == 3:
        q[:3] = [m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1]]
        q[:3] /= divisor
    else:
        j,k = (index+1)%3,(index+2)%3
        q[j] = (m[j,index]+m[index,j])/divisor
        q[k] = (m[k,index]+m[index,k])/divisor
        q[3] = (m[k,j]-m[j,k])/divisor
    return q/np.linalg.norm(q)
