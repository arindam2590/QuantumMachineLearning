"""Check CLI parsing without requiring the optional simulation backend."""
import argparse
import ast
from pathlib import Path


def test_documented_cli_options():
    tree = ast.parse((Path(__file__).parents[1]/'main.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_parser')
    scope = dict(argparse=argparse, Path=Path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<driver>', 'exec'), scope)
    parser = scope['build_parser']()
    for scenario, count in [('urban',8), ('open_field',20), ('cluttered',16)]:
        args = parser.parse_args(['--scenario',scenario,'--num-drones',str(count),'--headless'])
        assert args.scenario == scenario and args.num_drones == count and not args.render
    args = parser.parse_args(['--render','--num-moving-obstacles','3','--trails','--directions',
                              '--output','results/run.json','--max-steps','100'])
    assert args.render and args.num_moving_obstacles == 3 and args.steps == 100
    assert args.trails and args.directions
