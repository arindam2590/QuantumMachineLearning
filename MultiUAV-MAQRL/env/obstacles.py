"""Trajectory utilities for the pending dynamic-obstacle integration."""
import numpy as np


def ping_pong_position(start, end, speed, elapsed):
    """Constant-speed back-and-forth motion between two waypoints."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = np.linalg.norm(end-start)
    if speed < 0 or elapsed < 0:
        raise ValueError("Speed and elapsed time must be nonnegative")
    if length == 0:
        return start.copy()
    phase = (elapsed*speed/length) % 2
    return start+(end-start)*(1-abs(phase-1))


def trajectory_position(points, speed, elapsed, mode):
    """Linear stops at the last waypoint; waypoint loops; back_and_forth reverses."""
    points = np.asarray(points, dtype=float)
    if mode == 'back_and_forth':
        return ping_pong_position(points[0], points[-1], speed, elapsed)
    if mode == 'waypoint':
        points = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = lengths.sum()
    if total == 0:
        return points[0].copy()
    distance = elapsed*speed
    distance = distance % total if mode == 'waypoint' else min(distance, total)
    for i, length in enumerate(lengths):
        if distance <= length and length > 0:
            return points[i]+(points[i+1]-points[i])*distance/length
        distance -= length
    return points[-1].copy()


def wind_acceleration(config, elapsed):
    """World-frame external acceleration in m/s², independent of rendering."""
    if not config.enabled:
        return np.zeros(3)
    return (np.asarray(config.wind_acceleration)+np.asarray(config.variation_amplitude)*
            np.sin(2*np.pi*config.variation_hz*elapsed))


class MovingObstacles:
    """Seeded kinematic obstacles with collision-free reserved path envelopes."""
    def __init__(self, client, config, rng, regions, static_bodies, drones, world_half, altitude, gui=False):
        import pybullet as p
        self.client, self.config, self.elapsed = client, config, 0.
        self.items = []
        self.gui = gui
        low = np.array(config.bounds_min if config.bounds_min is not None else [-world_half*.8, -world_half*.8, 3.])
        high = np.array(config.bounds_max if config.bounds_max is not None else [world_half*.8, world_half*.8, altitude-2.])
        if np.any(low >= high) or np.any(np.abs([low[:2], high[:2]]) > world_half) or low[2] < 0 or high[2] > altitude:
            raise ValueError('Invalid moving obstacle bounds')
        for _ in range(config.count):
            for attempt in range(2000):
                size = float(rng.uniform(config.min_size, config.max_size))
                points = rng.uniform(low+size, high-size, size=(3, 3))
                # Reserve the entire swept path box, preventing scenery/region intersections.
                a, b = points.min(axis=0)-size, points.max(axis=0)+size
                if any(r.overlaps_box((a+b)/2, (b-a)/2) for r in regions):
                    continue
                bodies = list(static_bodies)+[d.body_id for d in drones.values()]+[item['body'] for item in self.items]
                if any(np.all(np.asarray(p.getAABB(body, physicsClientId=client)[1]) >= a) and
                       np.all(np.asarray(p.getAABB(body, physicsClientId=client)[0]) <= b) for body in bodies):
                    continue
                break
            else:
                raise ValueError('Cannot place moving obstacles safely; reduce count/size or adjust bounds')
            shape = str(rng.choice(config.shapes))
            kind = {'box': p.GEOM_BOX, 'sphere': p.GEOM_SPHERE, 'cylinder': p.GEOM_CYLINDER}[shape]
            args = dict(halfExtents=[size]*3) if shape == 'box' else dict(radius=size)
            collision_args = dict(args)
            visual_args = dict(args)
            if shape == 'cylinder':
                collision_args['height'] = 2*size
                visual_args['length'] = 2*size
            collision = p.createCollisionShape(kind, physicsClientId=client, **collision_args)
            visual = p.createVisualShape(kind, rgbaColor=[1., .2, .1, 1.], physicsClientId=client, **visual_args)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                     baseVisualShapeIndex=visual, basePosition=points[0], physicsClientId=client)
            item = dict(body=body, points=points, size=size)
            self.items.append(item)
            if gui:
                path = points if config.trajectory != 'back_and_forth' else points[[0, -1]]
                if config.trajectory == 'waypoint':
                    path = np.vstack([path, path[0]])
                for start, end in zip(path[:-1], path[1:]):
                    p.addUserDebugLine(start, end, [1., .15, .05], lineWidth=3,
                                       physicsClientId=client)

    def update(self, dt):
        import pybullet as p
        self.elapsed += dt
        for index, item in enumerate(self.items):
            position = trajectory_position(item['points'], self.config.speed, self.elapsed, self.config.trajectory)
            p.resetBasePositionAndOrientation(item['body'], position, [0, 0, 0, 1], physicsClientId=self.client)
