"""Box navigation regions and seeded, separated fleet placement."""
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass
class Region:
    center: np.ndarray
    half_size: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).copy()
        self.half_size = np.asarray(self.half_size, dtype=float).copy()
        if (self.center.shape != (3,) or self.half_size.shape != (3,)
                or not np.all(np.isfinite(self.center))
                or not np.all(np.isfinite(self.half_size)) or np.any(self.half_size <= 0)):
            raise ValueError("Region needs finite 3D center and positive half_size")

    def contains(self, position, radius=0.):
        """Require the complete spherical collision envelope inside the box."""
        return bool(np.all(np.abs(np.asarray(position)-self.center)+radius <= self.half_size+1e-9))

    def overlaps_box(self, center, half_size):
        return bool(np.all(np.abs(np.asarray(center)-self.center) < self.half_size+half_size))

    def draw(self, client, color):
        import pybullet as p
        corners = list(itertools.product([-1, 1], repeat=3))
        for i, a in enumerate(corners):
            for b in corners[i+1:]:
                if sum(x != y for x, y in zip(a, b)) == 1:
                    p.addUserDebugLine(self.center+self.half_size*a,
                                       self.center+self.half_size*b, color,
                                       physicsClientId=client)


def spawn_positions(region, count, separation, rng=None, radius=0., is_free=None):
    """Sample a shuffled lattice; fail clearly rather than overlap UAVs."""
    if count < 1 or separation <= 0 or radius < 0 or separation < 2*radius:
        raise ValueError("Invalid fleet size, separation, or collision radius")
    usable = region.half_size-radius
    if np.any(usable < 0):
        raise ValueError("Region is smaller than the UAV collision envelope")
    axes = []
    for extent in usable:
        n = int(np.floor(2*extent/separation))+1
        axes.append((np.arange(n)-(n-1)/2)*separation)
    points = np.array(list(itertools.product(*axes)))+region.center
    rng = np.random.default_rng(0) if rng is None else rng
    rng.shuffle(points)
    points = [point for point in points if is_free is None or is_free(point)]
    if len(points) < count:
        raise ValueError(f"Region capacity {len(points)} is insufficient for {count} UAVs")
    return np.array(points[:count])
