"""Execute only the matched boundary-penalty experiment; no gamma sweep."""
from pathlib import Path
import hashlib,json,shlex,time
from train import build_parser,run

OUT=Path('outputs/benchmarks/boundary_penalty_20_gamma_095')
MANIFEST=Path('experiments/maddpg/boundary_penalty_20_gamma_095.json')


def main():
    testlog=(OUT/'regression_tests.log').read_text()
    if '172 passed' not in testlog or 'failed' in testlog:
        raise RuntimeError('Complete regression suite must pass before this experiment')
    manifest=json.loads(MANIFEST.read_text())
    if manifest['status']=='completed':raise RuntimeError('Experiment already completed; refusing accidental duplicate')
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in protected.items()):
        raise RuntimeError('Protected source changed')
    argv=shlex.split(manifest['command'])[2:]
    args=build_parser().parse_args(argv)
    assert args.resume is None and args.out_of_bounds_penalty==20
    manifest['status']='running';MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.time()
    summary=run(args)
    verification={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items()}
    assert all(verification.values())
    (OUT/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'verification.json').write_text(json.dumps(dict(protected_sources_unchanged=verification,wall_seconds=time.time()-started),indent=2)+'\n')
    manifest.update(status='completed',actual_environment_steps=summary['environment_steps'],actual_updates=summary['updates'],run_directory=summary['run_directory'])
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
