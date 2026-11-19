"""Shared entry-point preparation; intentionally performs no rollout or learning."""
import argparse
import json
from dataclasses import asdict
from config.maddpg_config import MADDPGConfig
from .registry import available_algorithms, create_algorithm


def prepare(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"{mode} entry-point scaffold; no learning implemented")
    parser.add_argument("--algo", required=True, choices=available_algorithms())
    parser.add_argument("--scenario", choices=["urban", "open_field", "cluttered"], default="urban")
    parser.add_argument("--num-drones", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    defaults = MADDPGConfig()
    for name in ('actor_lr','critic_lr','gamma','tau','exploration_noise_std',
                 'exploration_noise_decay','minimum_exploration_noise'):
        parser.add_argument('--'+name.replace('_','-'),type=float,default=getattr(defaults,name))
    for name in ('replay_capacity','batch_size','warmup_steps','policy_update_frequency',
                 'max_training_steps','validation_interval','checkpoint_interval','evaluation_episodes'):
        parser.add_argument('--'+name.replace('_','-'),type=int,default=getattr(defaults,name))
    parser.add_argument('--hidden-layers',type=int,nargs='+',default=defaults.hidden_layers)
    parser.add_argument('--optimizer',choices=['Adam'],default=defaults.optimizer)
    parser.add_argument('--device',choices=['cpu','cuda','mps','auto'],default=defaults.device)
    parser.add_argument('--gradient-clip-norm',type=float,default=None)
    args = parser.parse_args(argv)
    values = {name:getattr(args,name) for name in asdict(defaults)}
    values['hidden_layers'] = tuple(values['hidden_layers'])
    config = MADDPGConfig(**values)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.num_drones < 1:
        parser.error("--num-drones must be positive")
    spec = create_algorithm(args.algo)
    from config import EnvConfig
    from env import MultiDroneEnv
    from .env_adapter import ParallelEnvAdapter
    with MultiDroneEnv(EnvConfig(gui=False,scenario=args.scenario,num_drones=args.num_drones)) as env:
        adapter = ParallelEnvAdapter(env)
        adapter.reset(seed=args.seed)
        learner = create_algorithm(args.algo,adapter=adapter,config=config)
        state_dim = adapter.global_state_dim
    result = dict(mode=mode, learner_created=True, global_state_dim=state_dim, algorithm_config=asdict(config),
                  exploration_enabled=mode == "train", algorithm=spec.name, implemented=spec.implemented,
                  scenario=args.scenario, num_drones=args.num_drones, seed=args.seed,
                  observation_dim=spec.observation_dim, action_dim=spec.action_dim,
                  status="learner initialized; no rollout, training, evaluation, or checkpoint loading executed")
    print(json.dumps(result, indent=2))
    return result
