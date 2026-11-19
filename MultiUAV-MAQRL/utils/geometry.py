"""Reusable Cartesian geometry operations."""
import numpy as np


def aabb_distance(point, center, half_size):
    """Euclidean distance to an axis-aligned box (zero inside)."""
    return float(np.linalg.norm(np.maximum(np.abs(np.asarray(point)-center)-half_size, 0)))
