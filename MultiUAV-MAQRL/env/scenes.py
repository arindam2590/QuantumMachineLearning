"""Scene dispatch; urban geometry remains owned by the original town builder."""
import numpy as np
import pybullet as p
from .town_sim import TownEnvironment


def build_scene(client, config, rng, regions):
    scene = TownEnvironment(client, config.town, rng, reserved_regions=regions)
    if config.scenario == 'urban':
        scene.build()
        return scene
    half = scene.side/2+config.world_margin
    scene._box([half+1, half+1, .25], [0, 0, -.25], [.23, .40, .26, 1])
    if config.scenario == 'open_field':
        return scene
    top = config.drone.max_altitude+config.drone.radius
    # Interior wall faces coincide with the existing horizontal world bounds.
    for axis in (0, 1):
        for sign in (-1, 1):
            dimensions = [half+1, half+1, top/2]
            dimensions[axis] = .5
            position = [0., 0., top/2]
            position[axis] = sign*(half+.5)
            scene._box(dimensions, position, [.55, .58, .64, 1])
    scene._box([half+1, half+1, .25], [0, 0, top+.25], [.6, .62, .68, 1])
    # Keep a central x-directed corridor free at every flight altitude.
    clearance = max(3., 3*config.drone.radius)
    for x in np.linspace(-scene.side*.35, scene.side*.35, 7):
        for sign in (-1, 1):
            width, depth = rng.uniform(1.5, 3.5, 2)
            height = float(rng.uniform(4., min(20., top*.6)))
            y = sign*float(rng.uniform(clearance+depth, max(clearance+depth, scene.side*.4)))
            center = np.array([x, y, height/2])
            extent = np.array([width/2, depth/2, height/2])
            if np.any(np.abs(center[:2])+extent[:2] >= half):
                continue
            if any(region.overlaps_box(center, extent) for region in regions):
                continue
            shape = str(rng.choice(['box', 'cylinder']))
            color = [.75, .48, .24, 1]
            if shape == 'box':
                body = scene._box(extent, center, color)
            else:
                radius = min(width, depth)/2
                collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client)
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=client)
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                         baseVisualShapeIndex=visual, basePosition=center, physicsClientId=client)
                scene.static_body_ids.append(body)
            scene.buildings.append(dict(id=body, center=(x, y), width=float(width),
                                        depth=float(depth), height=height, shape=shape))
    return scene
