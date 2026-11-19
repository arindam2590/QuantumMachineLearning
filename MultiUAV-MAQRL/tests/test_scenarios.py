"""Actual headless scenario integration tests; require simulator dependencies."""
import numpy as np
import pytest
pytest.importorskip('pybullet')
pytest.importorskip('gymnasium')
import pybullet as p
from config import EnvConfig
from env import MultiDroneEnv


@pytest.mark.parametrize('scenario', ['urban', 'cluttered', 'open_field'])
@pytest.mark.parametrize('count', [2, 4, 8, 16, 20])
def test_scenario_spawns_and_reproducibility(scenario, count):
    with MultiDroneEnv(EnvConfig(scenario=scenario, num_drones=count, gui=False)) as env:
        env.reset(seed=42)
        original = np.array([d.state()[0] for d in env.drones.values()])
        buildings = [b.copy() for b in env.town.buildings]
        assert len(original) == count
        for drone in env.drones.values():
            assert env.source.contains(drone.state()[0], env.cfg.drone.radius)
            for body in env.town.static_body_ids:
                assert not p.getClosestPoints(drone.body_id, body, 0, physicsClientId=env.client)
        distances = np.linalg.norm(original[:, None]-original[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 2*env.cfg.drone.radius
        env.reset(seed=42)
        assert np.array_equal(original, [d.state()[0] for d in env.drones.values()])
        assert buildings == env.town.buildings
        if scenario == 'urban':
            assert len(buildings) == round(env.town.eligible_plot_count*env.cfg.town.building_density)
        if scenario == 'cluttered':
            assert buildings
            for b in buildings:
                assert abs(b['center'][1])-b['depth']/2 >= 3
