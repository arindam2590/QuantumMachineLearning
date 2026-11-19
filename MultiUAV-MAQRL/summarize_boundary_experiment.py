"""Build a factual report from the completed controlled experiment artifacts."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
import numpy as np

OUT=Path('outputs/benchmarks/boundary_penalty_20_gamma_095')


def main():
    manifest=json.loads(Path('experiments/maddpg/boundary_penalty_20_gamma_095.json').read_text())
    if manifest['status']!='completed':raise ValueError('Training has not completed')
    comparison=json.loads((OUT/'comparison.json').read_text())
    root=Path(manifest['run_directory'])
    with next((root/'training').glob('*_training.csv')).open() as f:rows=list(csv.DictReader(f))
    assert rows and int(rows[-1]['environment_steps'])==186374
    penalty_count=0
    for row in rows:
        statuses=json.loads(row['agent_status']);penalties=json.loads(row['out_of_bounds_penalties'])
        assert math.isclose(float(row['team_reward']),float(row['raw_team_reward'])+sum(penalties.values()),abs_tol=1e-8)
        for a,value in penalties.items():
            assert value==(-20. if statuses[a]=='out_of_bounds' else 0.)
            penalty_count+=int(value==-20.)
    loss_count=0;max_loss=0.
    with next((root/'training').glob('*_actor_critic_loss.csv')).open() as f:
        for row in csv.DictReader(f):
            for name in ('actor_loss','critic_loss'):
                value=float(row[name]);assert math.isfinite(value);max_loss=max(max_loss,abs(value))
            loss_count+=1
    training_summary=json.loads((OUT/'training_summary.json').read_text())
    protection=json.loads((OUT/'protected_hashes.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in protection.items())
    scheduled=[]
    for path in (root/'validation').glob('*_validation.csv'):
        with path.open() as f:
            data=list(csv.DictReader(f))
        if data and 'evaluation_step' in data[0]:scheduled.extend(data)
    outcomes=Counter(s['status'] for r in scheduled for s in json.loads(r['per_agent_status']).values())
    audit=dict(environment_steps=training_summary['environment_steps'],updates=training_summary['updates'],
        one_time_penalty_events=penalty_count,reward_identity_verified=True,finite_loss_rows=loss_count,
        maximum_absolute_loss=max_loss,protected_sources_unchanged=True,
        scheduled_evaluation_episodes=len(scheduled),scheduled_evaluation_status_counts=dict(outcomes),
        scheduled_arrivals=sum(int(r['arrivals']) for r in scheduled))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    v=comparison['validation'];h=v['horizon_only'];b=v['boundary_penalty_20']
    def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    table='| Metric | Horizon only | Boundary penalty −20 |\n|---|---:|---:|\n'
    for label,key in [('Boundary failure rate','boundary_failure_rate'),('Negative-X saturation','negative_x_saturation'),('Individual arrival rate','individual_arrival_rate'),('Team success rate','team_success_rate'),('Collisions / 20 UAV outcomes','collisions'),('Mean destination progress (m)','mean_destination_progress'),('Mean episode steps','mean_steps'),('Mean raw team reward','mean_raw_reward'),('Mean learning team reward','mean_learning_reward')]:
        table+=f'| {label} | {fmt(h[key])} | {fmt(b[key])} |\n'
    report=f'''# Controlled out-of-bounds penalty experiment

Fresh Pure MADDPG run completed exactly **186,374 environment steps**, with γ=0.95, 2 UAVs, urban density 0.65, the PD controller, and a 1,600-step / 80-second horizon. The only learning-objective change is the opt-in −20 boundary penalty. Actor/critic architecture, optimizers, exploration, replay implementation, dynamics, observations, and baseline reward module are unchanged. No other gamma experiment was launched.

**Outcome:** The penalty changed the saturated direction and improved geometric progress, but did not produce successful navigation. In the paired final evaluation, negative-X saturation changed from 100% to 0%, while absolute action saturation remained 100% on all three axes. The new deterministic command was `[+1, -1, +1]`. Boundary failures declined from 18/20 to 15/20, but collisions increased from 2/20 to 5/20. Both individual arrivals and team success remained zero. The mean time to episode completion increased from 6.265 s to 29.29 s, still well below the 80-second limit because failures occurred first. These results support a behavioral change for this seed, not recovery from policy collapse or collision-free navigation.

## Interface and integration

`env/multi_drone_env.py` determines boundaries from world XY limits and configured minimum/maximum flight altitude. It publishes an explicit `out_of_bounds` Boolean and terminal `status`. The existing `env/rewards.py:reward_components` has no boundary/status argument; its generic terminal flag is insufficient. This limitation was reported before implementation. The minimal integration is an algorithm-independent runner adjustment, using explicit infos and per-agent termination/truncation flags. No frozen environment caller was edited.

A penalty requires status `out_of_bounds`, the explicit boundary flag, true termination, no truncation, no collision, and no successful/reached arrival. It applies only once to that UAV; a per-episode latch resets with each episode. Collision takes precedence when collision and boundary flags coincide. Magnitude defaults to zero. Original reward components are preserved and logged separately from adjusted learning return. See [INTEGRATION.md](INTEGRATION.md) for field definitions.

## Matched final deterministic evaluation

Both final checkpoints were loaded and evaluated without noise or parameter changes on seeds **100042–100051**, with the same 80-second limit. Rates use 20 individual UAV outcomes; team success uses 10 episodes. Saturation means commanded X ≤ −0.98, averaged equally across UAVs and episodes over each UAV's active steps.

{table}

Horizon-only mean action [X,Y,Z]: `{h['action_mean']}`.
Boundary-penalty mean action [X,Y,Z]: `{b['action_mean']}`.
Absolute saturation rates [X,Y,Z]: control `{h['action_saturation']}`; penalty `{b['action_saturation']}`.
Per-UAV progress: control `{h['per_agent_destination_progress']}`; penalty `{b['per_agent_destination_progress']}`.
Terminal counts: control `{h['status_counts']}`; penalty `{b['status_counts']}`.

Raw return is directly comparable across policies; learning return includes the changed objective and must not be interpreted as a like-for-like performance score.

## Training and scheduled validation

Training counters: `{training_summary['episodes_completed']}` complete episodes, `{training_summary['agent_transitions']}` agent transitions, `{training_summary['updates']}` updates, `{training_summary['wall_seconds']:.2f}` seconds. The budget may end mid-episode; incomplete agent statuses are not counted as completed failure outcomes. Episode reset seeds follow 42 + episode index, so policy-dependent episode lengths change actual seed coverage despite the same deterministic seed rule.

Training comparison:
```json
{json.dumps(comparison['training'],indent=2)}
```

Scheduled validation: {len(scheduled)} episodes (10 fixed seeds at each 10,000-step checkpoint), terminal counts `{dict(outcomes)}`, arrivals `{audit['scheduled_arrivals']}`. Reused validation seeds support paired comparisons but do not establish generalization.

## Verification

- Complete regression suite executed **before training: 172 passed**; repeated after training: **172 passed** (see `regression_tests.log`). This includes explicit boundary eligibility, collision, arrival, truncation, independent agents, once-only/reset behavior, default-off behavior, legacy checkpoint defaults, and enabled checkpoint/resume/evaluation.
- Checkpoint trainer metadata stores the adjustment config/version. Old checkpoints load with zero adjustment. Resume restores the saved setting and rejects a conflicting explicit override. Final evaluation asserted all model parameters unchanged.
- Audit verified {penalty_count} one-time boundary penalties and reward identity `learning = raw + adjustment` for every training row. Every logged actor/critic loss is finite across {loss_count} agent update records; no numerical exceptions were reported. Finite losses do not prove a stable/useful policy.
- SHA-256 checks confirm all protected `env/*.py`, `config/*.py`, and `algorithms/maddpg/*.py` sources unchanged; initial hashes also matched the horizon-only run. Historical result files were not renamed, moved, or overwritten.

## Interpretation and limitations

Use the paired results above to distinguish a changed saturated action direction, fewer boundary exits, positive geometric progress, and actual arrivals. A lower boundary rate alone is insufficient: it may reflect more collisions or stalled episodes. This is one training seed and ten repeatedly used evaluation seeds, not a significance study. γ remains 0.95 and no changes to critic state, network architecture, reward coefficients other than this explicit adjustment, or physical behavior were made. No follow-up experiment was started automatically.

## Changed code and scope

Modified: `train.py` (opt-in adjustment, checkpoint setting, raw/learning logging), `algorithms/evaluation.py` (restore adjustment and collect matching telemetry), `utils/training_telemetry.py` (signed saturation). Added: `algorithms/reward_adjustments.py`, `tests/test_boundary_reward_adjustment.py`, `experiments/maddpg/boundary_penalty_20_gamma_095.json`, `run_boundary_experiment.py`, `compare_boundary_policies.py`, and `summarize_boundary_experiment.py`. No environment, configuration-default, reward-module, adapter, replay, controller, sensor, building, or network implementation was changed.

The first warm-up episode exactly matched the horizon-only run for seed, step count, action means, initial and final goal distances, and statuses (`initial_trajectory_verification.json`). Saved algorithm and environment configurations matched exactly (`matched_configuration.json`).

## Artifacts

Manifest: `experiments/maddpg/boundary_penalty_20_gamma_095.json`.
Run: `{root}`.
Final checkpoint: `{training_summary['final_checkpoint']}`.
Standardized training CSV, actor/critic loss CSV, 300-DPI PNG plots, checkpoints, and validation CSV/JSON are under that run. Paired episode diagnostics are `horizon_only_evaluation.json` and `boundary_penalty_20_evaluation.json`; machine-readable comparison is `comparison.json`, with integrity audit in `audit.json`.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
