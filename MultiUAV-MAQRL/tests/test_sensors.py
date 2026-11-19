"""Sensor tests need NumPy only; no renderer or physics connection."""
import importlib.util
from pathlib import Path
import numpy as np
from config.environment_config import SensorConfig

spec = importlib.util.spec_from_file_location('state_sensors', Path(__file__).parents[1]/'env/sensors.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dimensions_and_world_coordinates():
    config = SensorConfig()
    imu = module.IMUSensor(config, np.random.default_rng(1))
    gps = module.GPSSensor(config, np.random.default_rng(2))
    args = (0., [1,2,3], [4,5,6], [7,8,9])
    assert np.array_equal(imu.update(*args)['value'], [4,5,6,7,8,9])
    assert np.array_equal(gps.update(*args)['value'], [1,2,3])
    assert gps.read()['frame'] == 'local_world'


def test_cache_frequency_and_copy():
    gps = module.GPSSensor(SensorConfig(gps_hz=10), np.random.default_rng(1))
    gps.update(0., [1,2,3], [], [])
    assert np.array_equal(gps.update(.09, [4,5,6], [], [])['value'], [1,2,3])
    assert np.array_equal(gps.update(.1, [4,5,6], [], [])['value'], [4,5,6])
    result = gps.read()
    result['value'][:] = 0
    assert np.array_equal(gps.read()['value'], [4,5,6])


def test_noise_bias_drift_and_reproducibility():
    config = SensorConfig(imu_noise_std=.1, imu_bias=(1.,)*6, imu_bias_drift=.2)
    a = module.IMUSensor(config, np.random.default_rng(42))
    b = module.IMUSensor(config, np.random.default_rng(42))
    c = module.IMUSensor(config, np.random.default_rng(43))
    for t in [0., .02, .04]:
        args = (t, np.zeros(3), np.zeros(3), np.zeros(3))
        av, bv, cv = [sensor.update(*args)['value'] for sensor in (a,b,c)]
        assert np.array_equal(av, bv)
        assert not np.array_equal(av, cv)
    assert not np.array_equal(a.bias, np.ones(6))


def test_gps_bias_keeps_truth_separate():
    gps = module.GPSSensor(SensorConfig(gps_bias=(1,2,3)), np.random.default_rng(1))
    truth = np.array([10.,20.,30.])
    reading = gps.update(0., truth, [], [])
    assert np.array_equal(reading['value'], [11,22,33])
    assert np.array_equal(truth, [10,20,30])


def test_depth_existing_utility():
    assert np.allclose(module.metric_depth([0,1], .1, 100), [.1,100])


def test_lidar_detection_dimensions_cache_and_reset():
    from unittest.mock import MagicMock, patch
    import sys
    backend = MagicMock()
    backend.getMatrixFromQuaternion.return_value = np.eye(3).ravel()
    config = SensorConfig(lidar_rays=2, lidar_vertical_angles=(0.,), lidar_range=10.)
    backend.rayTestBatch.return_value = [(3,-1,.5,None,None), (-1,-1,1,None,None)]
    with patch.dict(sys.modules, {'pybullet': backend}):
        lidar = module.LiDARSensor(0, 1, config, np.random.default_rng(42), .5)
        lidar.update(0., np.zeros(3), [0,0,0,1])
        value = lidar.read()['value']
        assert value['normalized'].shape == (2,)
        assert np.isclose(value['distance_m'][0], .5001+.5*9.4999)
        assert value['distance_m'][1] == 10
        lidar.update(.01, np.zeros(3), [0,0,0,1])
        assert backend.rayTestBatch.call_count == 1
        lidar.update(.1, np.zeros(3), [0,0,0,1])
        assert backend.rayTestBatch.call_count == 2
        fresh = module.LiDARSensor(0, 1, config, np.random.default_rng(42), .5)
        assert fresh.read()['timestamp'] is None


def test_camera_shapes_and_disabled_acquisition():
    from unittest.mock import MagicMock, patch
    import sys
    backend = MagicMock()
    backend.getMatrixFromQuaternion.return_value = np.eye(3).ravel()
    config = SensorConfig(camera_width=4, camera_height=3)
    backend.getCameraImage.return_value = (4,3,np.zeros((3,4,4)),np.zeros((3,4)),None)
    with patch.dict(sys.modules, {'pybullet': backend}):
        camera = module.RGBDCamera(0,1,config,np.random.default_rng(42))
        camera.update(0., np.zeros(3), [0,0,0,1])
        backend.getCameraImage.assert_not_called()
        config.camera_enabled = True
        camera.update(0., np.zeros(3), [0,0,0,1])
        value = camera.read()['value']
        assert value['rgb'].shape == (3,4,3)
        assert value['depth_m'].shape == (3,4)
        assert np.allclose(value['depth_m'], config.camera_near)
        camera.update(.01, np.zeros(3), [0,0,0,1])
        backend.getCameraImage.assert_called_once()
