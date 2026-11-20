"""Temporary world-frame safety intervention, executed by the existing controller."""
import numpy as np


class BoundaryRecovery:
    def __init__(self, lower, upper, drone, navigation):
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.drone, self.nav = drone, navigation
        self.active = False
        self.faces = np.zeros(3)
        self.count = 0
        self.elapsed = 0.
        self.yaw = None
        self.previous_yaw = None

    def update(self, position, velocity, yaw, dt):
        position, velocity = np.asarray(position), np.asarray(velocity)
        contact = np.where(position <= self.lower, 1., np.where(position >= self.upper, -1., 0.))
        new = bool(np.any(contact) and not self.active)
        if new:
            self.active = True
            self.count += 1
            self.elapsed = 0.
            self.previous_yaw = yaw
            self.yaw = (yaw + 2*np.pi) % (2*np.pi) - np.pi  # wrap(yaw + pi)
            self.faces = contact
        if self.active:
            self.elapsed += dt
            self.faces = np.where(contact != 0, contact, self.faces)
            # Release only after position AND outward braking reserve are safe.
            accel = np.array([self.drone.max_accel_xy]*2+[self.drone.max_accel_z])
            reserve = self.nav.boundary_recovery_margin + velocity**2/(2*accel)
            safe = np.all(position > self.lower+reserve) and np.all(position < self.upper-reserve)
            if safe:
                self.active = False
        failed = self.active and (self.elapsed > self.nav.boundary_recovery_timeout or
                    np.any(position < self.lower-self.nav.boundary_max_overshoot) or
                    np.any(position > self.upper+self.nav.boundary_max_overshoot))
        return new, bool(failed)

    def desired_velocity(self, position):
        # Aim inside every contacted face; brake tangential motion as well.
        target = np.clip(position, self.lower+self.nav.boundary_recovery_margin*2,
                         self.upper-self.nav.boundary_recovery_margin*2)
        desired = target-np.asarray(position)
        limits = np.array([self.drone.max_speed_xy]*2+[self.drone.max_speed_z])
        return np.clip(desired, -limits, limits)
