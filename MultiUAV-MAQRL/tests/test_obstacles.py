import importlib.util
from pathlib import Path
import numpy as np
from config.environment_config import DisturbanceConfig

spec = importlib.util.spec_from_file_location('obstacle_geometry', Path(__file__).parents[1]/'env/obstacles.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trajectories():
    points = [[0,0,0], [2,0,0], [2,2,0]]
    assert np.allclose(module.trajectory_position(points, 1, 100, 'linear'), points[-1])
    assert np.allclose(module.trajectory_position(points, 1, 3, 'waypoint'), [2,1,0])
    assert np.allclose(module.trajectory_position(points[:2], 1, 3, 'back_and_forth'), [1,0,0])


def test_wind_disabled_constant_and_varying():
    config = DisturbanceConfig(wind_acceleration=(1,0,0))
    assert np.allclose(module.wind_acceleration(config, 1), 0)
    config.enabled = True
    assert np.allclose(module.wind_acceleration(config, 1), [1,0,0])
    config.variation_amplitude = (1,0,0)
    config.variation_hz = .25
    assert np.allclose(module.wind_acceleration(config, 1), [2,0,0])
