"""Runnable renderer + illustrative goal-directed multi-drone velocity policy.

Example: python main.py --building-density .65 --num-drones 12
         python main.py --headless --episodes 2 --steps 500
"""
import argparse
import time
import json
from pathlib import Path
import pybullet as p
import numpy as np
from env import TownConfig, EnvConfig, MultiDroneEnv


def demo_policy(observations, env):
    """Naive go-to-goal controller with altitude clearance; NOT learned MARL."""
    actions = {}
    # This reactive demonstration is not a guaranteed collision-free planner.
    c = env.cfg
    half = env.town.side/2+c.world_margin
    scale = np.array([half, half, c.drone.max_altitude])
    for agent in env.agents:
        obs = observations[agent]
        pos = obs[:3]*scale
        delta = obs[9:12]*(2*scale)
        cruise = c.spawn_altitude if c.scenario != 'urban' else c.town.max_building_height*1.27+5
        cruise = np.clip(cruise, c.drone.min_altitude+1, c.drone.max_altitude-1)
        desired_z = cruise if np.linalg.norm(delta[:2]) > env.destination.half_size[0] else env.destination.center[2]
        velocity = np.array([delta[0]*.3, delta[1]*.3, (desired_z-pos[2])*.55])
        if c.scenario == 'urban' and pos[2] < cruise-2 and np.linalg.norm(delta[:2]) > env.destination.half_size[0]:
            velocity[:2] = 0
        reading = env.get_sensor_readings(agent)['lidar']['value']
        if reading is not None:
            angles = np.linspace(-np.deg2rad(c.sensors.lidar_fov)/2,
                                 np.deg2rad(c.sensors.lidar_fov)/2,
                                 c.sensors.lidar_rays, endpoint=c.sensors.lidar_fov < 360)
            yaw = np.arctan2(obs[6], obs[7])
            directions = np.array([[np.cos(v)*np.cos(h+yaw), np.cos(v)*np.sin(h+yaw), np.sin(v)]
                for v in np.deg2rad(c.sensors.lidar_vertical_angles) for h in angles])
            strength = np.maximum(0., 1-reading['distance_m']/6.)
            velocity -= 2*np.sum(directions*strength[:,None], axis=0)/max(1, np.count_nonzero(strength))
        actions[agent] = np.clip(velocity/np.array([c.drone.max_speed_xy]*2+[c.drone.max_speed_z]), -1, 1).astype(np.float32)
    return actions


def scripted_policy(env):
    """Fixed simulation-time velocity script; no planner or obstacle avoidance.

    0–2s climb, 2–4s hover, 4–6s forward, 6–8s lateral,
    8–10s reverse, then stop. Initial spawn is already airborne.
    """
    phase = min(int(env.step_count*env.cfg.control_dt//2), 5)
    velocity = np.asarray([[0,0,1],[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,0]][phase])
    scale = np.array([env.cfg.drone.max_speed_xy]*2+[env.cfg.drone.max_speed_z])
    return {a: (velocity/scale).astype(np.float32) for a in env.agents}


def build_parser():
    parser = argparse.ArgumentParser(description='Multi-drone 3D PyBullet MARL town')
    parser.add_argument('--scenario', choices=['urban', 'open_field', 'cluttered', 'indoor'], default='urban')
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument('--render', dest='render', action='store_true',
                           help='enable PyBullet GUI rendering (default)')
    rendering.add_argument('--no-render', '--headless', dest='render', action='store_false',
                           help='disable rendering and use PyBullet DIRECT mode')
    parser.set_defaults(render=True)
    parser.add_argument('--building-density', type=float, default=.55)
    parser.add_argument('--num-drones', type=int, default=8)
    parser.add_argument('--blocks-per-side', type=int, default=15)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', '--max-steps', type=int, default=600, help='control steps per episode')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--realtime', action='store_true', help='sleep to match wall clock')
    parser.add_argument('--num-moving-obstacles', type=int, default=0)
    parser.add_argument('--overview', action='store_true', help='wider whole-town camera instead of the default close view')
    parser.add_argument('--scripted-demo', action='store_true', help='fixed velocity script, use --seed 42 for repeatability')
    parser.add_argument('--velocity-vectors', action='store_true', help='show measured velocity in green (one-second displacement)')
    parser.add_argument('--trails', action='store_true', help='show trails lasting 5 simulated seconds')
    parser.add_argument('--directions', action='store_true', help='show destination direction lines')
    parser.add_argument('--output', type=Path, help='save episode statistics as JSON')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    town = TownConfig(building_density=args.building_density,
                      blocks_per_side=args.blocks_per_side, seed=args.seed)
    cfg = EnvConfig(scenario=args.scenario, num_drones=args.num_drones, town=town, gui=args.render,
                    max_episode_steps=args.steps, realtime=args.realtime)
    cfg.obstacles.count = args.num_moving_obstacles
    cfg.validate()
    summaries = []
    with MultiDroneEnv(cfg) as env:
        for episode in range(args.episodes):
            obs, infos = env.reset(seed=args.seed+episode)
            if cfg.gui and args.overview:
                p.resetDebugVisualizerCamera(cameraDistance=env.town.side*1.35,cameraYaw=35,
                    cameraPitch=-62,cameraTargetPosition=[0,0,12],physicsClientId=env.client)
            print(f'Episode {episode+1}: buildings={len(env.town.buildings)}, '
                  f'drones={len(env.drones)}, action_dim=3, obs_dim={env.observation_space.shape[0]}')
            total_reward = {a: 0. for a in env.possible_agents}
            successes = set()
            collisions = set()
            started = time.perf_counter()
            direction_ids = {}
            velocity_ids = {}
            for step in range(args.steps):
                if cfg.gui and not p.isConnected(physicsClientId=env.client):
                    return
                previous = {a: env.drones[a].state()[0] for a in env.agents}
                actions = scripted_policy(env) if args.scripted_demo else demo_policy(obs, env)
                obs, rewards, terminated, truncated, infos = env.step(actions)
                if cfg.gui:
                    for a, start in previous.items():
                        end = env.drones[a].state()[0]
                        if args.velocity_vectors:
                            velocity_ids[a] = p.addUserDebugLine(end, end+env.drones[a].state()[1], [0,1,.3],
                                replaceItemUniqueId=velocity_ids.get(a,-1), physicsClientId=env.client)
                        if args.trails:
                            p.addUserDebugLine(start, end, [.9,.6,.1], lifeTime=5., physicsClientId=env.client)
                        if args.directions:
                            direction_ids[a] = p.addUserDebugLine(end, env.destination.center, [.2,.5,1.],
                                replaceItemUniqueId=direction_ids.get(a, -1), physicsClientId=env.client)
                for a, r in rewards.items():
                    total_reward[a] += r
                successes.update(a for a, info in infos.items() if info['reached_goal'])
                collisions.update(a for a, info in infos.items() if info['collision'])
                if terminated['__all__'] or truncated['__all__']:
                    break
            print(f'  steps={step+1} successes={len(successes)}/{args.num_drones} '
                  f'collisions={len(collisions)} mean_reward={np.mean(list(total_reward.values())):.2f}')
            summary = dict(episode=episode+1, seed=args.seed+episode, **env.episode_statistics,
                collisions=len(collisions), mean_reward=float(np.mean(list(total_reward.values()))),
                mean_battery=float(np.mean([d.battery for d in env.drones.values()])),
                wall_seconds=time.perf_counter()-started)
            summaries.append(summary)
            print(f"  team_success={summary['team_success']} simulated_seconds={summary['duration']:.2f} "
                  f"mean_battery={summary['mean_battery']:.4f}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summaries, indent=2)+'\n')
        if cfg.gui:
            print('Simulation complete. Close the PyBullet window to exit.')
            try:
                while p.isConnected(physicsClientId=env.client):
                    # Poll GUI events without advancing the completed simulation.
                    p.getKeyboardEvents(physicsClientId=env.client)
                    time.sleep(.05)
            except p.error:
                if p.isConnected(physicsClientId=env.client):
                    raise
            except KeyboardInterrupt:
                pass



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except p.error:
        if p.isConnected():
            raise
