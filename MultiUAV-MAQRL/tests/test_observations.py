import importlib.util
from pathlib import Path
import numpy as np
import pytest
from config import EnvConfig
spec = importlib.util.spec_from_file_location('local_observations', Path(__file__).parents[1]/'env/observations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state(x):
    return np.array([x,0.,10.]), np.zeros(3), np.zeros(3), np.zeros(3), np.array([0,0,0,1])


@pytest.mark.parametrize('count', [2,4,8,16,20])
def test_fixed_shape_normalization_and_padding(count):
    config = EnvConfig()
    states = {f'uav_{i}': state(i*100) for i in range(count)}
    readings = dict(gps=dict(value=np.array([0,0,10.])), imu=dict(value=np.zeros(6)), lidar=dict(value=None))
    obs = module.encode_observation('uav_0', states, readings, np.array([90,0,10]), config)
    assert obs.shape == (module.observation_size(config),) == (157,)
    assert obs.dtype == np.float32
    assert np.all(np.abs(obs) <= 1)
    assert np.all(obs[12:12+7*config.neighbour_count] == 0)


def test_nearest_neighbours_sorted_and_masked():
    config = EnvConfig(neighbour_count=2)
    states = {'ego':state(0), 'b':state(-1), 'a':state(1), 'far':state(100)}
    readings = dict(gps=dict(value=np.array([0,0,10.])), imu=dict(value=np.zeros(6)), lidar=dict(value=None))
    obs = module.encode_observation('ego',states,readings,np.zeros(3),config)
    assert obs[12] > 0  # Equal distance: ID a before b.
    assert obs[18] < 0
    assert np.array_equal(obs[24:26], [1,1])
