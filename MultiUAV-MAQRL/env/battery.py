"""Normalized charge approximation, not motor power or aerodynamic modelling."""
import numpy as np


class Battery:
    def __init__(self, config):
        self.config = config
        self.level = config.initial_charge
        self.consumed = 0.

    def update(self, velocity, acceleration, dt):
        c = self.config
        rate = (c.hover_rate+c.motion_rate*np.dot(velocity, velocity)+
                c.acceleration_rate*np.dot(acceleration, acceleration)+
                c.climb_rate*max(0., velocity[2]))
        used = min(self.level, max(0., rate*dt))
        self.level = max(0., self.level-used)
        self.consumed += used
        return used

    @property
    def depleted(self):
        return self.level <= 0.
