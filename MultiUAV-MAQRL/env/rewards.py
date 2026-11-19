"""Per-UAV navigation rewards; callers sum the returned signed components.

Coefficient mapping (legacy defaults are supplied by RewardWeights):
    progress: +0.1 * distance reduction
    arrival: +100 once (non-default configured arrival magnitudes override)
    proximity: -abs(weights.proximity) * normalized severity * policy dt
    collision: -abs(weights.collision) * reported collision event
    energy: -abs(weights.energy) * actual normalized charge consumed

The caller supplies only one proximity scalar and one collision flag. Thus the
requested UAV/UAV and UAV/obstacle terms cannot be identified separately here.
They remain single category-agnostic components, never duplicated or guessed.
Legacy time/effort keys and arguments remain for compatibility, with zero
contributions because they are absent from the proposed navigation objective.
"""
import math
from config.environment_config import RewardWeights


DEFAULT_ARRIVAL_REWARD = 100.0
_CONFIG_DEFAULT_ARRIVAL = RewardWeights().arrival


def reward_components(previous_distance, distance, effort=0., terminal=False,
                      reached=False, weights=None, *, collision=False,
                      energy=0., proximity=0., dt=1., already_arrived=False):
    """Return the existing dictionary of finite, signed reward contributions.

    Penalty weights accept either legacy negative coefficients or nonnegative
    magnitudes: abs() converts once, and the returned contribution is negative.
    ``energy`` is existing per-step battery consumption, not a new energy model.
    Proximity retains the caller's normalized surface-clearance severity and
    existing dt scaling; it includes all queried body types, even ground/roads.

    ``terminal`` denotes the current transition, not prior completion. A new
    terminal arrival may earn its bonus; a current terminal failure may incur
    penalties. Previously terminated agents must be excluded by the caller.
    ``already_arrived`` suppresses every term. Collision is an actual caller
    event and is never inferred from proximity. Collision retirement provides
    one-event handling in MultiDroneEnv; no cross-agent/global history is kept.

    Nonfinite used inputs, invalid negative dt, or overflowing contributions
    raise ValueError rather than sending NaN/Inf into replay. Legacy unused
    effort/time weights do not participate in this objective.
    """
    result = dict(progress=0., time=0., effort=0., arrival=0., collision=0.,
                  proximity=0., energy=0.)
    if already_arrived:
        return result
    weights = weights or RewardWeights()

    def finite(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{name} must be a finite scalar') from error
        if not math.isfinite(value):
            raise ValueError(f'{name} must be a finite scalar')
        return value

    previous = finite('previous_distance', previous_distance)
    current = finite('distance', distance)
    severity = min(1., max(0., finite('proximity', proximity)))
    energy_cost = max(0., finite('energy', energy))
    period = finite('dt', dt)
    if period < 0:
        raise ValueError('dt must be nonnegative')
    progress_weight = finite('weights.progress', weights.progress)
    configured_arrival = finite('weights.arrival', weights.arrival)
    goal_weight = (DEFAULT_ARRIVAL_REWARD if configured_arrival == _CONFIG_DEFAULT_ARRIVAL
                   else abs(configured_arrival))
    proximity_weight = abs(finite('weights.proximity', weights.proximity))
    collision_weight = abs(finite('weights.collision', weights.collision))
    energy_weight = abs(finite('weights.energy', weights.energy))

    # Each nonnegative penalty magnitude is negated exactly once.
    result.update(progress=progress_weight*(previous-current),
                  arrival=goal_weight if reached and not collision else 0.,
                  proximity=-proximity_weight*severity*period,
                  collision=-collision_weight if collision else 0.,
                  energy=-energy_weight*energy_cost)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Reward contribution overflow')
    try:
        total = math.fsum(result.values())
    except OverflowError as error:
        raise ValueError('Total reward overflow') from error
    if not math.isfinite(total) or not math.isfinite(sum(result.values())):
        raise ValueError('Total reward overflow')
    return result
