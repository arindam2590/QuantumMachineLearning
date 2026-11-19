"""Deterministic trained-model playback using the frozen public environment API."""
import argparse
import json
import pybullet as p
from pathlib import Path
from algorithms.registry import available_algorithms,load_algorithm
from algorithms.evaluation import checkpoint_environment,evaluate


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algo',required=True,choices=available_algorithms())
    parser.add_argument('--checkpoint',required=True,type=Path)
    rendering=parser.add_mutually_exclusive_group()
    rendering.add_argument('--render',action='store_true',help='enable existing PyBullet GUI')
    rendering.add_argument('--headless',dest='render',action='store_false')
    parser.set_defaults(render=False)
    parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--num-drones',type=int)
    parser.add_argument('--scenario',choices=['urban','open_field','cluttered'])
    parser.add_argument('--seed',type=int,default=200000)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--episode-steps',type=int)
    parser.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--close-on-finish',action='store_true',help='GUI exits after final episode instead of waiting')
    return parser


def run(args):
    learner=load_algorithm(args.algo,args.checkpoint,args.device)
    cfg=checkpoint_environment(learner)
    cfg.gui=args.render;cfg.realtime=args.realtime
    if args.num_drones is not None:cfg.num_drones=args.num_drones
    if args.scenario is not None:cfg.scenario=args.scenario
    if args.episode_steps is not None:cfg.max_episode_steps=args.episode_steps
    cfg.validate()
    def report(row):
        print(json.dumps(dict(seed=row['seed'],steps=row['steps'],team_success=row['team_success'],
            arrivals=row['arrivals'],collisions=row['collisions'],per_agent_status=row['per_agent_status'])),flush=True)
    from utils.results import export_evaluation
    result=evaluate(learner,cfg,episodes=args.episodes,seed=args.seed,
                    result_writer=lambda result:export_evaluation(result,args.output,args.algo,learner,cfg,args.checkpoint,'test'),
                    enforce_heldout=False,episode_callback=report,
                    keep_open=args.render and not args.close_on_finish)
    print(json.dumps({k:v for k,v in result.items() if k!='episode_metrics'},indent=2))
    return result


if __name__=='__main__':
    parser=build_parser()
    args=parser.parse_args()
    try:run(args)
    except ValueError as error:parser.error(str(error))
    except KeyboardInterrupt:pass
    except p.error:
        if args.render and not p.isConnected():
            print('PyBullet disconnected; playback interrupted before episode completion.')
        else:
            raise
