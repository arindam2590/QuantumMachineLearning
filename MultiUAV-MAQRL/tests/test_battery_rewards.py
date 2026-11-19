import importlib.util
from pathlib import Path
import numpy as np
import pytest
from config.environment_config import BatteryConfig, RewardWeights


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'env'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

Battery = load('battery').Battery
reward_components = load('rewards').reward_components


def test_energy_calculation_and_independence():
    config = BatteryConfig(hover_rate=.1, motion_rate=.01, acceleration_rate=.02, climb_rate=.03)
    a, b = Battery(config), Battery(config)
    used = a.update(np.array([1.,0,2]), np.array([0.,3,0]), .5)
    assert used == pytest.approx((.1+.01*5+.02*9+.03*2)*.5)
    assert a.level == pytest.approx(1-used)
    assert b.level == 1


def test_depletion_and_reset():
    config = BatteryConfig(initial_charge=.01, hover_rate=1.)
    battery = Battery(config)
    assert battery.update(np.zeros(3), np.zeros(3), 1) == .01
    assert battery.level == 0 and battery.depleted
    assert battery.update(np.zeros(3), np.zeros(3), 1) == 0
    assert Battery(config).level == .01


def test_reward_components_and_one_time_arrival():
    weights = RewardWeights()
    result = reward_components(10,8,reached=True,collision=True,energy=.1,
                               proximity=.5,dt=.05,weights=weights)
    assert result['progress'] == pytest.approx(.2)
    assert result['arrival'] == 0.0
    assert result['collision'] == weights.collision
    assert result['energy'] < 0 and result['proximity'] < 0
    assert reward_components(8,8,reached=True,already_arrived=True)['arrival'] == 0
    assert reward_components(8,8,terminal=True,collision=False)['collision'] == 0
    assert reward_components(10, 8, reached=True, collision=False,
                             weights=weights)['arrival'] == 100.0
    completed = reward_components(10, 8, reached=True, already_arrived=True,
                                  collision=True, energy=.1, proximity=.5,
                                  effort=1., weights=weights)
    assert all(value == 0.0 for value in completed.values())
