"""Fixed decentralized observation encoding; all features clipped to [-1, 1]."""
import numpy as np


def observation_size(config):
    return 21+7*config.neighbour_count+config.sensors.lidar_rays*len(config.sensors.lidar_vertical_angles)


def encode_observation(agent, states, readings, destination, config, battery=1.):
    position, velocity, rpy, _, _ = states[agent]
    half = config.town.blocks_per_side*config.town.plot_spacing/2+config.world_margin
    position_scale = np.array([half, half, config.drone.max_altitude])
    velocity_scale = np.array([config.drone.max_speed_xy]*2+[config.drone.max_speed_z])
    gps = readings['gps']['value']
    imu = readings['imu']['value']
    # Neighbours model range-limited relative-state sensing (no global list in output).
    neighbours = []
    for other, state in states.items():
        if other == agent:
            continue
        delta = state[0]-position
        distance = np.linalg.norm(delta)
        if distance <= config.neighbour_radius:
            neighbours.append((distance, other, delta, state[1]-velocity))
    neighbours.sort(key=lambda item: (item[0], item[1]))
    slots = np.zeros((config.neighbour_count, 6))
    mask = np.zeros(config.neighbour_count)
    for i, (_, _, delta, relative_velocity) in enumerate(neighbours[:config.neighbour_count]):
        slots[i] = np.r_[delta/config.neighbour_radius, relative_velocity/(2*velocity_scale)]
        mask[i] = 1
    lidar = readings['lidar']['value']
    ranges = np.ones(config.sensors.lidar_rays*len(config.sensors.lidar_vertical_angles)) if lidar is None else lidar['normalized']
    features = np.concatenate([gps/position_scale, velocity/velocity_scale,
        [np.sin(rpy[2]), np.cos(rpy[2])], [battery], (destination-gps)/(2*position_scale),
        slots.ravel(), mask, ranges,
        imu[:3]/config.imu_acceleration_scale, imu[3:]/config.imu_angular_scale,
        gps/position_scale])
    if not np.all(np.isfinite(features)):
        raise ValueError('Nonfinite sensor observation')
    return np.clip(features, -1, 1).astype(np.float32)
