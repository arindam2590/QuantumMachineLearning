"""Reusable static procedural town builder. No simulation loops or CLI here."""
import numpy as np
import pybullet as p


from .environment_config import TownConfig


class TownEnvironment:
    """Town geometry only; uses the PyBullet client provided by its owner.

    build() may be invoked after resetSimulation; it never connects/disconnects.
    """
    def __init__(self, client_id: int, config: TownConfig, rng=None, reserved_regions=()):
        config.validate()
        self.client_id = client_id
        self.cfg = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.reserved_regions = reserved_regions
        self.buildings = []
        self.static_body_ids = []
        self.side = config.blocks_per_side * config.plot_spacing

    def _box(self, half, position, color):
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client_id)
        vid = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color,
                                  physicsClientId=self.client_id)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cid,
                                 baseVisualShapeIndex=vid, basePosition=position,
                                 physicsClientId=self.client_id)
        self.static_body_ids.append(body)
        return body

    def build(self):
        self.buildings.clear()
        self.static_body_ids.clear()
        cfg = self.cfg
        side = self.side
        self._box([side / 2 + 8, side / 2 + 8, .25], [0, 0, -.25], [.23, .40, .26, 1])
        # Narrow, intersecting road strips through the entire city.
        for k in range(cfg.blocks_per_side + 1):
            v = (k - cfg.blocks_per_side / 2) * cfg.plot_spacing
            self._box([cfg.road_width / 2, side / 2 + 2, .015],
                      [v, 0, .016], [.16, .17, .19, 1])
            self._box([side / 2 + 2, cfg.road_width / 2, .015],
                      [0, v, .018], [.16, .17, .19, 1])
        plots = [(i, j) for i in range(cfg.blocks_per_side)
                 for j in range(cfg.blocks_per_side)]
        # Reserve only plots whose maximum building envelope intersects a region.
        footprint = (cfg.plot_spacing-cfg.road_width-1)/2
        roof = max(cfg.max_building_height+2.4, cfg.max_building_height*1.26)
        plots = [(i, j) for i, j in plots if not any(
            region.overlaps_box([(i+.5-cfg.blocks_per_side/2)*cfg.plot_spacing,
                                 (j+.5-cfg.blocks_per_side/2)*cfg.plot_spacing, roof/2],
                                np.array([footprint, footprint, roof/2]))
            for region in self.reserved_regions)]
        self.eligible_plot_count = len(plots)
        self.rng.shuffle(plots)
        count = round(len(plots) * cfg.building_density)
        footprint_limit = cfg.plot_spacing - cfg.road_width - 1
        for i, j in plots[:count]:
            x = (i + .5 - cfg.blocks_per_side / 2) * cfg.plot_spacing
            y = (j + .5 - cfg.blocks_per_side / 2) * cfg.plot_spacing
            w, d = self.rng.uniform(3., footprint_limit, size=2)
            h = float(self.rng.uniform(cfg.min_building_height, cfg.max_building_height))
            rgba = self.rng.uniform([.35, .38, .43], [.8, .8, .85]).tolist() + [1.]
            shape = str(self.rng.choice(['box', 'tower', 'stepped', 'cylinder']))
            if shape == 'cylinder':
                radius = min(w, d) * .5
                collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=h,
                                                    physicsClientId=self.client_id)
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=h,
                                              rgbaColor=rgba, physicsClientId=self.client_id)
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                         baseVisualShapeIndex=visual, basePosition=[x, y, h/2],
                                         physicsClientId=self.client_id)
                self.static_body_ids.append(body)
            else:
                body = self._box([w/2, d/2, h/2], [x, y, h/2], rgba)
                if shape == 'tower':
                    self._box([w*.23, d*.23, 1.2], [x, y, h+1.2], rgba)
                elif shape == 'stepped':
                    self._box([w*.33, d*.33, h*.13], [x, y, h+h*.13], rgba)
            self.buildings.append(dict(id=body, center=(x, y), height=h,
                                       width=float(w), depth=float(d), shape=shape))
        return self.buildings

    def height_at(self, x, y):
        """Conservative footprint-height query, useful for spawn clearance."""
        z = 0.
        for b in self.buildings:
            bx, by = b['center']
            radius_x = b['width']/2
            radius_y = b['depth']/2
            if abs(x-bx) <= radius_x and abs(y-by) <= radius_y:
                extra = 2.4 if b['shape'] == 'tower' else (.26*b['height'] if b['shape'] == 'stepped' else 0.)
                z = max(z, b['height'] + extra)
        return z
