"""Sensor measurement utilities for the pending sensor integration."""
import numpy as np


def metric_depth(buffer, near, far):
    """Convert an OpenGL depth buffer to metric camera-axis depth."""
    if not 0 < near < far:
        raise ValueError("Require 0 < near < far")
    return far*near/(far-(far-near)*np.asarray(buffer))


class StateSensor:
    """Cached world-frame measurements, timestamped in simulation seconds.

    IMU acceleration is linear acceleration (m/s²), not accelerometer specific
    force: gravity is not added. Angular velocity is world-frame rad/s.
    GPS position uses the simulator's local Cartesian metres.
    """
    def __init__(self, kind, config, rng):
        self.kind, self.config, self.rng = kind, config, rng
        self.frequency = config.imu_hz if kind == 'imu' else config.gps_hz
        self.noise = config.imu_noise_std if kind == 'imu' else config.gps_noise_std
        self.bias = np.asarray(config.imu_bias if kind == 'imu' else config.gps_bias, dtype=float).copy()
        self.cached = None
        self.last_update = None
        self.next_update = 0.

    def update(self, timestamp, position, acceleration, angular_velocity):
        if self.cached is not None and timestamp+1e-10 < self.next_update:
            return self.read()
        truth = np.concatenate([acceleration, angular_velocity]) if self.kind == 'imu' else np.asarray(position)
        elapsed = 0. if self.last_update is None else timestamp-self.last_update
        drift = self.config.imu_bias_drift if self.kind == 'imu' else 0.
        if drift and elapsed > 0:
            self.bias += self.rng.normal(0., drift*np.sqrt(elapsed), self.bias.shape)
        self.cached = np.asarray(truth)+self.bias+self.rng.normal(0., self.noise, truth.shape)
        self.last_update = float(timestamp)
        self.next_update = (np.floor((timestamp+1e-10)*self.frequency)+1)/self.frequency
        return self.read()

    def read(self):
        return dict(value=None if self.cached is None else self.cached.copy(),
                    timestamp=self.last_update, frequency_hz=self.frequency,
                    frame='local_world', units=('m/s², rad/s' if self.kind == 'imu' else 'm'),
                    bias=self.bias.copy(), kind=self.kind)


class IMUSensor(StateSensor):
    def __init__(self, config, rng):
        super().__init__('imu', config, rng)


class GPSSensor(StateSensor):
    def __init__(self, config, rng):
        super().__init__('gps', config, rng)


class GeometrySensor:
    """Simulation-clock cache for PyBullet geometry sensors."""
    def __init__(self, client, body, config, rng, kind):
        self.client, self.body, self.config, self.rng, self.kind = client, body, config, rng, kind
        self.frequency = config.lidar_hz if kind == 'lidar' else config.camera_hz
        self.cached, self.timestamp, self.next_update = None, None, 0.

    def read(self):
        return dict(value=None if self.cached is None else {k: v.copy() for k, v in self.cached.items()},
                    timestamp=self.timestamp, frequency_hz=self.frequency, kind=self.kind,
                    frame='body' if self.kind == 'lidar' else 'camera',
                    enabled=self.config.lidar_enabled if self.kind == 'lidar' else self.config.camera_enabled)

    def update(self, timestamp, position, quaternion):
        enabled = self.config.lidar_enabled if self.kind == 'lidar' else self.config.camera_enabled
        if not enabled or (self.cached is not None and timestamp+1e-10 < self.next_update):
            return
        self.cached = self.acquire(np.asarray(position), quaternion)
        self.timestamp = float(timestamp)
        self.next_update = (np.floor((timestamp+1e-10)*self.frequency)+1)/self.frequency


class LiDARSensor(GeometrySensor):
    def __init__(self, client, body, config, rng, radius):
        super().__init__(client, body, config, rng, 'lidar')
        self.radius = radius
        c = config
        angles = np.linspace(-np.deg2rad(c.lidar_fov)/2, np.deg2rad(c.lidar_fov)/2,
                             c.lidar_rays, endpoint=c.lidar_fov < 360)
        elevations = np.deg2rad(c.lidar_vertical_angles)
        self.directions = np.array([[np.cos(v)*np.cos(h), np.cos(v)*np.sin(h), np.sin(v)]
                               for v in elevations for h in angles])


    def acquire(self, position, quaternion):
        import pybullet as p
        c = self.config
        directions = self.directions
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
        directions = directions @ rotation.T
        # Start beyond the physical collision sphere, independently of visual scale.
        offset = self.radius+1e-4
        starts = position+directions*offset
        ends = position+directions*c.lidar_range
        results = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=self.client)
        truth = np.array([c.lidar_range if hit[0] < 0 or hit[0] == self.body else
                          offset+hit[2]*(c.lidar_range-offset) for hit in results])
        measured = np.clip(truth+self.rng.normal(0, c.lidar_noise_std, truth.shape), 0, c.lidar_range)
        return dict(ground_truth_m=truth.astype(np.float32), distance_m=measured.astype(np.float32),
                    normalized=(measured/c.lidar_range).astype(np.float32))


class RGBDCamera(GeometrySensor):
    def __init__(self, client, body, config, rng):
        super().__init__(client, body, config, rng, 'camera')

    def acquire(self, position, quaternion):
        import pybullet as p
        c = self.config
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3,3)
        eye = position+rotation@np.asarray(c.camera_offset)
        view = p.computeViewMatrix(eye.tolist(), (eye+rotation[:,0]).tolist(), rotation[:,2].tolist())
        projection = p.computeProjectionMatrixFOV(c.camera_fov, c.camera_width/c.camera_height,
                                                 c.camera_near, c.camera_far)
        result = p.getCameraImage(c.camera_width, c.camera_height, viewMatrix=view,
                                  projectionMatrix=projection, renderer=p.ER_TINY_RENDERER,
                                  physicsClientId=self.client)
        rgb = np.asarray(result[2], dtype=np.uint8).reshape(c.camera_height,c.camera_width,4)[...,:3].copy()
        depth = metric_depth(np.asarray(result[3]).reshape(c.camera_height,c.camera_width),
                             c.camera_near,c.camera_far).astype(np.float32)
        return dict(rgb=rgb, depth_m=depth)
