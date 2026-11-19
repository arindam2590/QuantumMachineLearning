"""Geometry checks run even when the optional simulator is unavailable."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('navigation_geometry', Path(__file__).parents[1]/'env/navigation.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Region, spawn_positions = module.Region, module.spawn_positions

@pytest.mark.parametrize('count', [2, 4, 8, 16, 20])
def test_spawning(count):
    region = Region([-52, 0, 12], [1, 52, 3])
    points = spawn_positions(region, count, 1.375, np.random.default_rng(42), .55)
    assert all(region.contains(point, .55) for point in points)
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 1.375-1e-9
    assert np.array_equal(points, spawn_positions(region, count, 1.375, np.random.default_rng(42), .55))


def test_destination_envelope():
    region = Region([10, 0, 5], [2, 2, 2])
    assert region.contains([11.5, 0, 5], .5)
    assert not region.contains([11.6, 0, 5], .5)
    assert not region.contains([10, 0, 7.1])


def test_insufficient_capacity_and_obstacles():
    region = Region([0, 0, 5], [1, 1, 1])
    with pytest.raises(ValueError, match='capacity'):
        spawn_positions(region, 20, 2, radius=.5)
    with pytest.raises(ValueError, match='capacity'):
        spawn_positions(region, 1, 2, is_free=lambda point: False)


def test_region_validation():
    with pytest.raises(ValueError):
        Region([0, 0, 0], [1, -1, 1])
