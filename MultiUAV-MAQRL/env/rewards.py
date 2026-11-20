"""Per-UAV navigation rewards; callers sum the returned signed components.

Coefficient mapping (legacy defaults are supplied by RewardWeights):
    progress: weighted signed change in distance to the admissible goal volume
    arrival: +100 once (non-default configured arrival magnitudes override)
    proximity: -abs(weights.proximity) * normalized severity * policy dt
    collision: -abs(weights.collision) * reported collision event
    energy: -abs(weights.energy) * actual normalized charge consumed

The legacy function receives one aggregate proximity scalar and collision flag.
The optional inter-UAV helper below uses explicit active-pair distances; its
caller excludes UAV bodies from aggregate proximity to avoid double penalties.
Legacy time/effort keys and arguments remain for compatibility, with zero
contributions because they are absent from the proposed navigation objective.
"""
import math
from .environment_config import RewardWeights


DEFAULT_PROGRESS_REWARD = 20.0
_CONFIG_DEFAULT_PROGRESS = RewardWeights().progress
DEFAULT_ARRIVAL_REWARD = 100.0
_CONFIG_DEFAULT_ARRIVAL = RewardWeights().arrival
GOAL_REWARD_FORMULATION = 'admissible_box_signed_progress_v2'


def inter_uav_proximity_penalty(distances, safety_radius, weight=1., dt=1.):
    """C1 threshold penalty: -weight*dt*sum(max(0,1-d/R)**2).

    Distances are 3D centers of active other UAVs; zero distance is finite.
    The caller must exclude UAVs from the aggregate proximity term when used.
    """
    if (not all(math.isfinite(x) for x in (safety_radius, weight, dt))
            or safety_radius <= 0 or weight < 0 or dt < 0):
        raise ValueError('Require positive radius and finite nonnegative weight/dt')
    terms = []
    for distance in distances:
        if not math.isfinite(distance) or distance < 0:
            raise ValueError('Separation must be finite and nonnegative')
        terms.append(max(0., 1-distance/safety_radius)**2)
    result = -weight*dt*math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('Separation reward overflow')
    return result


def reward_components(previous_distance, distance, effort=0., terminal=False,
                      reached=False, weights=None, *, collision=False,
                      energy=0., proximity=0., dt=1., already_arrived=False,
                      goal_radius=2., already_terminated=False):
    """Return the existing dictionary of finite, signed reward contributions.

    Penalty weights accept either legacy negative coefficients or nonnegative
    magnitudes: abs() converts once, and the returned contribution is negative.
    ``energy`` is existing per-step battery consumption, not a new energy model.
    Proximity retains the caller's normalized surface-clearance severity and
    existing dt scaling, including ground/roads. With separate UAV proximity
    enabled, the caller excludes UAVs from this aggregate query.
    The ``progress`` slot is wp*(previous-current), with caller-supplied distances
    to the destination box contracted by the UAV collision radius. No recurring
    inside-volume reward or absolute-distance cost is used. Legacy goal_radius,
    constant_goal_region_reward, distance_weight and distance_scale remain for
    signature/config compatibility and do not contribute to guidance.

    ``terminal`` and ``effort`` remain compatibility arguments. Arrival eligibility
    comes from the caller's reached/collision flags, not from a generic terminal
    flag. The caller adds boundary components using the actual failure reason.
    ``already_arrived`` or ``already_terminated`` suppress every term. Collision
    retirement in MultiDroneEnv supplies one-event handling; no episode history
    or privileged collision classification is invented inside this function.

    Nonfinite used inputs, invalid negative dt, or overflowing contributions
    raise ValueError rather than sending NaN/Inf into replay. Legacy unused
    effort/time weights do not participate in this objective.
    """
    result = dict(progress=0., time=0., effort=0., arrival=0., collision=0.,
                  proximity=0., energy=0.)
    if already_arrived or already_terminated:
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
    if previous < 0 or current < 0:
        raise ValueError('Goal-volume distances must be nonnegative')
    severity = min(1., max(0., finite('proximity', proximity)))
    energy_cost = max(0., finite('energy', energy))
    period = finite('dt', dt)
    if period < 0:
        raise ValueError('dt must be nonnegative')
    configured_progress = finite('weights.progress', weights.progress)
    progress_weight = (DEFAULT_PROGRESS_REWARD
                       if configured_progress == _CONFIG_DEFAULT_PROGRESS
                       else configured_progress)
    if progress_weight < 0:
        raise ValueError('Progress weight must be nonnegative')
    configured_arrival = finite('weights.arrival', weights.arrival)
    goal_weight = (DEFAULT_ARRIVAL_REWARD if configured_arrival == _CONFIG_DEFAULT_ARRIVAL
                   else abs(configured_arrival))
    proximity_weight = abs(finite('weights.proximity', weights.proximity))
    collision_weight = abs(finite('weights.collision', weights.collision))
    energy_weight = abs(finite('weights.energy', weights.energy))

    # Pure signed progress: no movement means no positive guidance, even inside.
    shaping = progress_weight*(previous-current)
    result.update(progress=shaping,
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
