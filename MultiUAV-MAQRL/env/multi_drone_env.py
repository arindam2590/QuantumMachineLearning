"""Parallel dictionary API using Gymnasium spaces, not a single-agent Gym API.

Each step returns entries for exactly the agents active at its start, including
final observations for agents finishing that step. Those agents are removed
from ``agents`` before the next call. ``__all__`` flags are legacy extensions
on termination/truncation dictionaries only; no actions are accepted for them.
Team success means every possible agent arrived safely. No PettingZoo dependency.
"""
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from .town_sim import TownConfig, TownEnvironment
from .drone import Drone, DroneConfig
from .rewards import reward_components, inter_uav_proximity_penalty
from .boundary_recovery import BoundaryRecovery
from .observations import observation_size, encode_observation


from .environment_config import EnvConfig


class MultiDroneEnv(gym.Env):
    """Shared scenes with local fixed-size observations and normalized 3D actions."""
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 20}

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.cfg.validate()
        self.render_mode = 'human' if self.cfg.gui else 'none'
        self.possible_agents = [f'uav_{i}' for i in range(self.cfg.num_drones)]
        self.agents = []
        self.observation_spaces = {a: spaces.Box(-1., 1., shape=(observation_size(self.cfg),), dtype=np.float32)
                                   for a in self.possible_agents}
        self.action_spaces = {a: spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
                              for a in self.possible_agents}
        # Convenience single-agent templates; per-agent dicts are the authoritative API.
        self.observation_space = self.observation_spaces[self.possible_agents[0]]
        self.action_space = self.action_spaces[self.possible_agents[0]]
        self.client = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError('Could not connect to PyBullet')
        self.town = None
        self.drones = {}
        self.goals = {}
        self._done = {}
        self.step_count = 0
        self._closed = False
        self.np_random = np.random.default_rng(self.cfg.town.seed)

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError('Cannot reset a closed environment')
        if options not in (None, {}):
            raise ValueError('No reset options are supported yet')
        super().reset(seed=seed)
        self.agents = []
        self.cfg.validate()
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0., 0., -self.cfg.drone.gravity, physicsClientId=self.client)
        p.setTimeStep(1./self.cfg.physics_hz, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        from .navigation import Region, spawn_positions
        half = self.cfg.town.blocks_per_side*self.cfg.town.plot_spacing/2
        regions = []
        for spec, x, altitude in [(self.cfg.source, -half, self.cfg.spawn_altitude),
                                  (self.cfg.destination, half, self.cfg.goal_altitude)]:
            spacing = self.cfg.town.plot_spacing
            depth, width = spec.grid_shape
            # Align the reservation with entire boundary plots and inset its
            # spawn volume slightly from the outer world boundary.
            if spec.half_size is None and (2*depth >= self.cfg.town.blocks_per_side or
                                           width > self.cfg.town.blocks_per_side):
                raise ValueError("Town is too small for opposite source/destination grid reservations")
            default_half = [depth*spacing/2, width*spacing/2, 3.]
            default_center = [np.sign(x)*(half-default_half[0]), 0, altitude]
            region = Region(spec.center if spec.center is not None else default_center,
                            spec.half_size if spec.half_size is not None else default_half)
            if (region.center[2]-region.half_size[2] < self.cfg.drone.min_altitude or
                    region.center[2]+region.half_size[2] > self.cfg.drone.max_altitude or
                    np.any(np.abs(region.center[:2])+region.half_size[:2] > half+self.cfg.world_margin)):
                raise ValueError("Region intersects ground or exceeds world bounds")
            spawn_positions(region, self.cfg.num_drones, 2.5*self.cfg.drone.radius,
                            radius=self.cfg.drone.radius)
            regions.append(region)
        self.source, self.destination = regions
        if self.cfg.navigation.stable_hover_enabled:
            spacing = max(2.5*self.cfg.drone.radius,
                          self.cfg.navigation.separation_radius if self.cfg.navigation.inter_uav_penalty_enabled else 0.)
            # Constructive, conservative capacity proof; never change the region.
            spawn_positions(self.destination, self.cfg.num_drones, spacing,
                            radius=self.cfg.drone.radius)
        if self.source.overlaps_box(self.destination.center, self.destination.half_size):
            raise ValueError("Source and destination regions must not overlap")
        from .scenes import build_scene
        self.town = build_scene(self.client, self.cfg, self.np_random, regions)
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self.drones = {}
        self.goals = {}
        self._done = {a: False for a in self.possible_agents}
        limit = self.town.side/2 + self.cfg.world_margin
        self._boundary = {a: BoundaryRecovery(
            [-limit, -limit, self.cfg.drone.min_altitude],
            [limit, limit, self.cfg.drone.max_altitude], self.cfg.drone, self.cfg.navigation)
            for a in self.possible_agents}
        self._goal_dwell = {a: 0 for a in self.possible_agents}
        def is_free(point):
            radius = self.cfg.drone.radius
            for body in self.town.static_body_ids:
                lower, upper = p.getAABB(body, physicsClientId=self.client)
                delta = np.maximum(np.maximum(np.asarray(lower)-point, point-np.asarray(upper)), 0)
                if np.linalg.norm(delta) <= radius:
                    return False
            return True
        starts = spawn_positions(self.source, len(self.agents), 2.5*self.cfg.drone.radius,
                                 self.np_random, self.cfg.drone.radius, is_free)
        self.status = {a: 'active' for a in self.possible_agents}
        self.episode_statistics = dict(steps=0, duration=0., arrivals=0, failures=0, team_success=False)
        self.cumulative_rewards = {a: 0. for a in self.possible_agents}
        self.arrival_times = {}
        self.sensors = {}
        from .drone_controller import VelocityController
        self.controllers = {}
        self._wrenches = {}
        self._physics_tick = 0
        from .sensors import IMUSensor, GPSSensor, LiDARSensor, RGBDCamera
        for i, agent in enumerate(self.agents):
            self.drones[agent] = Drone(self.client, starts[i], [.3, .7, .9, 1.], self.cfg.drone)
            self.controllers[agent] = VelocityController(self.cfg.drone)
            from .battery import Battery
            self.drones[agent].battery_model = Battery(self.cfg.battery)
            self.drones[agent].battery = self.drones[agent].battery_model.level
            self.goals[agent] = self.destination.center.copy()
            seeds = np.random.SeedSequence(int(self.np_random.integers(2**32))).spawn(2)
            self.sensors[agent] = dict(imu=IMUSensor(self.cfg.sensors, np.random.default_rng(seeds[0])),
                                       gps=GPSSensor(self.cfg.sensors, np.random.default_rng(seeds[1])))
            for sensor in self.sensors[agent].values():
                sensor.update(0., starts[i], np.zeros(3), np.zeros(3))
        from .obstacles import MovingObstacles
        self.moving_obstacles = MovingObstacles(self.client, self.cfg.obstacles, self.np_random,
            regions, self.town.static_body_ids, self.drones,
            half+self.cfg.world_margin, self.cfg.drone.max_altitude, gui=self.cfg.gui)
        for agent, drone in self.drones.items():
            seed = int(self.np_random.integers(2**32))
            self.sensors[agent]['lidar'] = LiDARSensor(self.client, drone.body_id, self.cfg.sensors,
                                                     np.random.default_rng(seed), self.cfg.drone.radius)
            self.sensors[agent]['camera'] = RGBDCamera(self.client, drone.body_id, self.cfg.sensors,
                                                      np.random.default_rng(seed+1))
            state = drone.state()
            for name in ('lidar', 'camera'):
                self.sensors[agent][name].update(0., state[0], state[4])
        for region, color in [(self.source, [0, 1, 0]), (self.destination, [0, .3, 1])]:
            region.draw(self.client, color)
        if self.cfg.gui:
            p.resetDebugVisualizerCamera(cameraDistance=self.town.side*.85, cameraYaw=35,
                cameraPitch=-62, cameraTargetPosition=[0,0,self.cfg.town.max_building_height*.35], physicsClientId=self.client)
        return self._observations(), {a: {'goal': self.goals[a].copy(), 'sensors': self.get_sensor_readings(a)} for a in self.agents}

    def _nearest_building_distance(self, pos):
        # Approximate 3D AABB distance: inexpensive sensing placeholder for future lidar.
        best = float('inf')
        for b in self.town.buildings:
            bx, by = b['center']
            roof = b['height'] + (2.4 if b['shape']=='tower' else
                    (.26*b['height'] if b['shape']=='stepped' else 0.))
            delta = np.maximum(np.abs(pos[:2]-[bx,by])-[b['width']/2,b['depth']/2], 0)
            vertical = max(0., pos[2]-roof, -pos[2])
            best = min(best, float(np.linalg.norm([delta[0],delta[1],vertical])))
        return best if np.isfinite(best) else self.town.side

    def _observations(self):
        states = {a: d.state() for a, d in self.drones.items() if self.status[a] == 'active'}
        observations = {}
        for agent in self.agents:
            value = encode_observation(agent, states, self.get_sensor_readings(agent),
                                       self.destination.center, self.cfg, self.drones[agent].battery)
            if not self.observation_spaces[agent].contains(value):
                raise ValueError(f"Invalid observation for {agent}")
            observations[agent] = value
        return observations

    def global_state(self):
        """Privileged state, separate from decentralized actor observations."""
        return dict(uavs={a: dict(position=d.state()[0], velocity=d.state()[1],
                                 active=a in self.agents, battery=d.battery) for a, d in self.drones.items()},
                    static_obstacles=[dict(b) for b in self.town.buildings],
                    moving_obstacles=[dict(body=item['body'], position=np.asarray(
                        p.getBasePositionAndOrientation(item['body'], physicsClientId=self.client)[0]))
                                      for item in self.moving_obstacles.items],
                    destination=self.destination.center.copy())

    def step(self, actions):
        if self._closed or not self.agents:
            raise RuntimeError('Call reset() before stepping / after an episode')
        if not isinstance(actions, dict):
            raise TypeError('actions must be a dict keyed by agent ID')
        unexpected = set(actions)-set(self.agents)
        if unexpected:
            raise KeyError(f'Unknown/inactive agents: {sorted(unexpected)}')
        controls = {}
        previous_dist = {}
        for agent in self.agents:
            if agent not in actions:
                raise KeyError(f'Missing action for {agent}')
            arr = np.asarray(actions[agent], dtype=float)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                raise ValueError(f'{agent} action must be a finite length-3 vector')
            if np.any(np.abs(arr) > 1):
                raise ValueError(f'{agent} action must be in [-1, 1]')
            controls[agent] = arr
            pos = self.drones[agent].state()[0]
            previous_dist[agent] = self.destination.distance_to_volume(pos, self.cfg.drone.radius)
        # Integrate drone position/velocity at physics_hz; keep the original
        # MARL control rate, actions, and reward/termination logic.
        n_substeps = self.cfg.physics_hz//self.cfg.control_hz
        collided = {a: False for a in self.agents}
        energy_used = {a: 0. for a in self.agents}
        dt = 1. / self.cfg.physics_hz
        from .obstacles import wind_acceleration
        controller_stride = self.cfg.physics_hz//self.cfg.low_level_hz
        boundary_events = {a: 0 for a in self.agents}
        boundary_failed = {a: False for a in self.agents}
        recovery_enabled = self.cfg.scenario == 'urban' and self.cfg.navigation.boundary_recovery_enabled
        for _ in range(n_substeps):
            before = {a: self.drones[a].state() for a in self.agents}
            self.moving_obstacles.update(dt)
            wind = wind_acceleration(self.cfg.disturbances, self.moving_obstacles.elapsed)
            for agent in self.agents:
                drone = self.drones[agent]
                recovery = self._boundary[agent]
                if recovery_enabled:
                    was_active = recovery.active
                    event, failed = recovery.update(before[agent][0], before[agent][1],
                        p.getEulerFromQuaternion(before[agent][4])[2], dt)
                    if event:
                        recovery.previous_yaw = self.controllers[agent].yaw_target
                    boundary_events[agent] += int(event)
                    boundary_failed[agent] |= failed
                    if was_active and not recovery.active:
                        self.controllers[agent].yaw_target = recovery.previous_yaw
                depleted = drone.battery_model.depleted and self.cfg.battery.terminate_on_depletion
                if depleted:
                    wrench = (0., np.zeros(3))
                else:
                    if self._physics_tick % controller_stride == 0:
                        if self.cfg.navigation.goal_heading_enabled and not (recovery_enabled and recovery.active):
                            nav = self.cfg.navigation
                            self.controllers[agent].update_goal_heading(
                                before[agent][0], self.goals[agent], 1/self.cfg.low_level_hz,
                                nav.goal_heading_rate, nav.heading_hold_radius)
                        desired = controls[agent]*np.array([self.cfg.drone.max_speed_xy]*2+
                                                          [self.cfg.drone.max_speed_z])
                        if recovery_enabled and recovery.active:
                            self.controllers[agent].yaw_target = recovery.yaw
                            desired = recovery.desired_velocity(before[agent][0])
                        self._wrenches[agent] = self.controllers[agent].compute(
                            drone.physical_state(), desired, 1/self.cfg.low_level_hz)
                    wrench = self._wrenches[agent]
                # PyBullet clears force accumulators each tick: reapply held commands.
                drone.apply_wrench(*wrench, external_force=self.cfg.drone.mass*wind)
            p.stepSimulation(physicsClientId=self.client)
            self._physics_tick += 1
            for agent in self.agents:
                drone = self.drones[agent]
                after = drone.state()
                if recovery_enabled:
                    recovery = self._boundary[agent]
                    was_active = recovery.active
                    event, failed = recovery.update(after[0], after[1],
                        p.getEulerFromQuaternion(after[4])[2], 0.)
                    if event:
                        recovery.previous_yaw = self.controllers[agent].yaw_target
                    boundary_events[agent] += int(event)
                    boundary_failed[agent] |= failed
                    if was_active and not recovery.active:
                        self.controllers[agent].yaw_target = recovery.previous_yaw
                acceleration = (after[1]-before[agent][1])/dt
                energy_used[agent] += drone.battery_model.update(
                    (before[agent][1]+after[1])/2, acceleration, dt)
                drone.battery = drone.battery_model.level
                for name in ('imu', 'gps'):
                    self.sensors[agent][name].update(self.moving_obstacles.elapsed,
                                                    after[0], acceleration, after[3])
                contacts = p.getContactPoints(bodyA=drone.body_id, physicsClientId=self.client)
                hits = {contact[2] for contact in contacts}
                # Preserve conservative segment checks as collision-event backup.
                hits |= drone._swept_collisions(before[agent][0], after[0])
                drone.collision_bodies = hits
                collided[agent] |= bool(hits)
                for other in self.agents:
                    if self.drones[other].body_id in hits:
                        collided[other] = True
            # Acquire only after all UAV poses and obstacle poses are current.
            for agent in self.agents:
                state = self.drones[agent].state()
                for name in ('lidar', 'camera'):
                    self.sensors[agent][name].update(self.moving_obstacles.elapsed, state[0], state[4])
        self.step_count += 1
        observations = self._observations()
        rewards, terminated, truncated, infos = {}, {}, {}, {}
        limit = self.town.side/2 + self.cfg.world_margin
        nav = self.cfg.navigation
        # Snapshot before any retirement: pair penalties must be symmetric even
        # when the first agent in iteration finishes on this transition.
        separation_positions = ({a: self.drones[a].state()[0].copy() for a in self.agents}
                                if nav.inter_uav_penalty_enabled else {})
        for agent in self.agents:
            pos = self.drones[agent].state()[0]
            distance = float(np.linalg.norm(self.goals[agent]-pos))
            volume_distance = self.destination.distance_to_volume(pos, self.cfg.drone.radius)
            out_of_bounds = bool(abs(pos[0])>limit or abs(pos[1])>limit or
                                 pos[2]<self.cfg.drone.min_altitude or pos[2]>self.cfg.drone.max_altitude)
            terminal_boundary = boundary_failed[agent] if recovery_enabled else out_of_bounds
            reached = self.destination.contains(pos, self.cfg.drone.radius) and not collided[agent] and not out_of_bounds
            done = self._done[agent]
            depleted = self.drones[agent].battery_model.depleted
            reached = reached and not (depleted and self.cfg.battery.terminate_on_depletion)
            if nav.stable_hover_enabled:
                stable = reached and np.linalg.norm(self.drones[agent].state()[1]) <= nav.hover_speed
                self._goal_dwell[agent] = self._goal_dwell[agent]+1 if stable else 0
                reached = self._goal_dwell[agent] >= nav.hover_dwell_steps
            new_terminal = (reached or collided[agent] or terminal_boundary or
                            (depleted and self.cfg.battery.terminate_on_depletion)) and not done
            body = self.drones[agent].body_id
            proximity = 0.
            for other in range(p.getNumBodies(physicsClientId=self.client)):
                other_body = p.getBodyUniqueId(other, physicsClientId=self.client)
                if other_body == body:
                    continue
                if nav.inter_uav_penalty_enabled and other_body in {d.body_id for d in self.drones.values()}:
                    continue  # replaced by the separate center-distance UAV term
                points = p.getClosestPoints(body, other_body, self.cfg.proximity_distance,
                                             physicsClientId=self.client)
                if points:
                    proximity = max(proximity, 1-max(0., min(point[8] for point in points))/self.cfg.proximity_distance)
            components = reward_components(previous_dist[agent], volume_distance,
                                           float(np.sum(controls[agent]**2)),
                                           new_terminal, reached, self.cfg.rewards,
                                           collision=collided[agent], energy=energy_used[agent],
                                           proximity=proximity, dt=self.cfg.control_dt,
                                           already_arrived=agent in self.arrival_times,
                                           already_terminated=done, goal_radius=self.cfg.goal_radius)
            if nav.inter_uav_penalty_enabled:
                distances = [float(np.linalg.norm(separation_positions[agent]-position))
                             for other, position in separation_positions.items() if other != agent]
                components['proximity_uav'] = inter_uav_proximity_penalty(
                    distances, nav.separation_radius, nav.separation_weight, self.cfg.control_dt)
            if recovery_enabled:
                components['boundary'] = (-self.cfg.navigation.boundary_contact_penalty*boundary_events[agent]
                                          if not done and not reached else 0.)
            else:
                boundary_failure = (new_terminal and terminal_boundary and not reached
                                    and not collided[agent]
                                    and not (depleted and self.cfg.battery.terminate_on_depletion))
                components['boundary'] = -self.cfg.rewards.boundary_exit if boundary_failure and not done else 0.
            rewards[agent] = sum(components.values()) if not done else 0.
            self._done[agent] = done or new_terminal
            if new_terminal and reached:
                drone = self.drones[agent]
                p.resetBaseVelocity(drone.body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
                p.changeVisualShape(drone.body_id, -1, rgbaColor=[0.1, 1., 0.2, 1.], physicsClientId=self.client)
                self.arrival_times[agent] = self.step_count/self.cfg.control_hz
            if new_terminal:
                self.status[agent] = ('arrived' if reached else 'collision' if collided[agent]
                                      else 'battery_depleted' if depleted and self.cfg.battery.terminate_on_depletion else 'out_of_bounds')
                self._retire_drone(agent, reached)
            elif self.step_count >= self.cfg.max_episode_steps:
                self.status[agent] = 'truncated'
                self._retire_drone(agent, False)
            self.cumulative_rewards[agent] += rewards[agent]
            terminated[agent] = self._done[agent]
            truncated[agent] = self.step_count >= self.cfg.max_episode_steps and not self._done[agent]
            infos[agent] = dict(battery=self.drones[agent].battery, battery_depleted=depleted,
                                energy_consumed=energy_used[agent],
                                total_energy_consumed=self.drones[agent].battery_model.consumed,
                                distance_to_destination_volume=volume_distance,
                                previous_distance_to_destination_volume=previous_dist[agent], sensors=self.get_sensor_readings(agent), reward_components=components, goal=self.goals[agent].copy(), distance_to_goal=distance,
                                reached_goal=bool(reached), collision=bool(collided[agent]),
                                out_of_bounds=out_of_bounds, success=bool(reached))
            if recovery_enabled:
                infos[agent].update(boundary_penalty_managed=True,
                    boundary_contact_count=boundary_events[agent],
                    boundary_contacts_total=self._boundary[agent].count,
                    boundary_exit_count=int(new_terminal and terminal_boundary and not collided[agent] and not reached
                                            and not (depleted and self.cfg.battery.terminate_on_depletion)),
                    boundary_recovery_active=self._boundary[agent].active)
        self.episode_statistics = dict(steps=self.step_count, duration=self.step_count*self.cfg.control_dt,
            arrivals=len(self.arrival_times),
            failures=sum(s not in ('active', 'arrived', 'truncated') for s in self.status.values()),
            team_success=len(self.arrival_times) == len(self.possible_agents))
        for agent in infos:
            if nav.stable_hover_enabled:
                infos[agent]['goal_dwell_steps'] = self._goal_dwell[agent]
            if nav.goal_heading_enabled:
                infos[agent]['yaw_reference'] = self.controllers[agent].yaw_target
            infos[agent].update(status=self.status[agent],
                                cumulative_reward=self.cumulative_rewards[agent],
                                episode=dict(self.episode_statistics))
        terminated['__all__'] = all(self._done.values())
        truncated['__all__'] = self.step_count >= self.cfg.max_episode_steps and not terminated['__all__']
        if self.cfg.realtime and self.cfg.gui:
            time.sleep(1./self.cfg.control_hz)
        self.agents = [a for a in self.agents if not (terminated[a] or truncated[a])]
        if terminated['__all__'] or truncated['__all__']:
            self.agents = []  # Return final observations/rewards on the terminal step.
        if any(terminated[a] or truncated[a] for a in observations):
            # Keep retiring agents' final observations, but refresh survivors so
            # neighbour slots exclude peers retired on this same transition.
            observations.update(self._observations())
        return observations, rewards, terminated, truncated, infos

    def _retire_drone(self, agent, arrived):
        """Keep a stationary visible marker with no collision geometry."""
        drone = self.drones[agent]
        final_state = drone.state()
        p.removeBody(drone.body_id, physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=drone.cfg.radius,
            rgbaColor=[.1, 1., .2, 1.] if arrived else [.8, .2, .2, .5], physicsClientId=self.client)
        drone.body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
            basePosition=final_state[0], baseOrientation=final_state[4], physicsClientId=self.client)

    def get_sensor_readings(self, agent):
        """Copies of latest noisy measurements; ground-truth state is unchanged."""
        return {name: sensor.read() for name, sensor in self.sensors[agent].items()}

    def render(self):
        if self.cfg.gui:
            return None  # GUI is live while p.stepSimulation() executes.
        return None

    def close(self):
        if not self._closed:
            if p.isConnected(physicsClientId=self.client):
                p.disconnect(physicsClientId=self.client)
            self.agents = []
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


MultiDroneTownEnv = MultiDroneEnv
