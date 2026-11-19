from config import EnvConfig

def test_default_config():
    EnvConfig().validate()

import pytest
from config import TownConfig

@pytest.mark.parametrize("kwargs", [{"num_drones": 0}, {"num_drones": 2.5},
    {"physics_hz": 0}, {"control_hz": 7}, {"physics_hz": float("inf")}])
def test_invalid_environment_config(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs).validate()

@pytest.mark.parametrize("density", [-.1, 1.1, float("nan")])
def test_invalid_density(density):
    with pytest.raises(ValueError):
        TownConfig(building_density=density).validate()

def test_timing_and_independent_defaults():
    a, b = EnvConfig(), EnvConfig()
    assert a.physics_dt == 1/120
    assert a.control_dt == 1/20
    a.sensors.enabled = True
    assert not b.sensors.enabled


def test_region_grid_defaults():
    config = EnvConfig()
    assert config.source.grid_shape == (2, 3)
    assert config.destination.grid_shape == (2, 3)
    config.source.grid_shape = (0, 3)
    with pytest.raises(ValueError, match="grid_shape"):
        config.validate()
