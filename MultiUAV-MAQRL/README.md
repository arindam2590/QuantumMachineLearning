# Multi-UAV PyBullet navigation

A parallel dictionary environment for later decentralized MARL experiments. No learning algorithm, critic, neural network, replay buffer, or training loop is included. This is not a standard single-agent Gymnasium environment or a strict PettingZoo implementation.

## Installation

Use Python 3.11 with a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The local `qmarl311` Conda environment contains PyBullet and Gymnasium. GUI execution requires a graphical desktop.

## Structure

```text
main.py                         CLI and reactive demonstration controller
validate.py                     headless validation, metrics and benchmarks
config/environment_config.py    centralized dataclass configuration
env/town_sim.py                 preserved procedural urban geometry
env/scenes.py                   urban, open-field and indoor scene dispatch
env/drone.py                    acceleration-limited kinematic UAVs
env/navigation.py               regions and seeded spawn placement
env/obstacles.py                moving obstacles and wind
env/sensors.py                  IMU, GPS, LiDAR and RGB-D caches
env/observations.py              local observation encoding
env/battery.py                  normalized charge approximation
env/rewards.py                  decomposed per-agent rewards
env/multi_drone_env.py          parallel environment lifecycle
utils/geometry.py               geometry utilities
tests/                          unit and headless integration checks
outputs/                        generated validation CSV/JSON
```

## Simulation

```bash
python main.py --scenario urban --num-drones 8 --building-density .65 --render
python main.py --scenario open_field --num-drones 20 --render
python main.py --scenario cluttered --num-drones 16 --headless
python main.py --scenario urban --num-moving-obstacles 2 --trails --directions --render
python main.py --headless --episodes 2 --steps 100 --seed 42 --output outputs/demo.json
```

`--no-render` aliases `--headless`; `--max-steps` aliases `--steps`. Rendering defaults on. GUI runs keep the final scene open until the window is closed or Ctrl+C is pressed. PyBullet provides mouse orbit/zoom controls and an overview camera. Trails expire after five seconds; direction lines are replaced. Other options include `--blocks-per-side` (15 by default) and `--realtime`.

Urban retains box, cylinder, tower and stepped buildings, roads, randomized dimensions and heights. Density is `round(eligible_plot_count * building_density)`; only plots intersecting reserved regions are excluded. The generator's building construction and style are preserved. Indoor (`cluttered` or `indoor`) has bounded walls, a ceiling, varied boxes/cylinders and a clear central corridor. Open field has only ground. The reactive controller seeks the destination with altitude adjustment and LiDAR repulsion; it is not a guaranteed collision-free planner.

## Regions and lifecycle

Source and destination reserve opposite 2 × 3 plot areas by default. Configure `source` and `destination` via `RegionConfig.grid_shape`, `center` and `half_size` (metres). Complete UAV collision spheres must fit inside the region for arrival. Seeded shuffled-lattice spawning checks capacity and scenery clearance. Arrivals stop and become non-colliding green markers. Team success requires every UAV to arrive without failure.

```python
from config import EnvConfig
from env import MultiDroneEnv

with MultiDroneEnv(EnvConfig(gui=False, num_drones=8)) as env:
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {a: env.action_spaces[a].sample() for a in env.agents}
        observations, rewards, terminated, truncated, infos = env.step(actions)
```

Step results contain the agents active at step start, including final observations. Finished agents leave `env.agents` before the next step. Actions must match active IDs exactly. Termination/truncation dictionaries additionally contain legacy `__all__` flags. Time limits truncate; arrivals, collisions, out-of-bounds flight and optionally depleted batteries terminate. Reset clears episode state; close is idempotent. `global_state()` supplies optional privileged information separately.

## Actions and observations

Each action is `Box(-1, 1, (3,), float32)`, mapped to desired world vx, vy, vz using horizontal and vertical speed limits. Horizontal velocity magnitude is capped.

All observation values are clipped to [-1,1] and checked against their Gymnasium Box. Layout:

- GPS-derived position (3), ego velocity (3), yaw sine/cosine (2), battery (1), destination minus GPS position (3).
- K neighbour relative positions/velocities (6K), then validity masks (K), sorted by distance and ID within the configured sensing radius.
- Normalized LiDAR ranges (L), IMU acceleration/angular velocity (6), GPS position (3).

Size is `21 + 7K + L`: defaults K=4, L=36×3 give **157**, independent of fleet size. Positions scale by horizontal world half-size and maximum altitude, destination vectors by twice those scales, velocities by their limits, relative velocities by twice those limits, neighbour positions by sensing radius, and IMU by configurable scales. Missing neighbours are zero; disabled LiDAR supplies ones. Neighbour sensing is ideal range-limited relative-state sensing, without occlusion or noise. Raw images are excluded. Detailed readings are available through `infos[a]['sensors']` and `get_sensor_readings(a)`.

## Motion, energy and rewards

PyBullet integrates the nonlinear rigid body at a fixed timestep:

```text
p_dot = v
v_dot = -g*e3 + (T/m)*R*e3 + F_wind/m
R_dot = R*skew(omega_body)
I*omega_body_dot = tau_body - omega_body × (I*omega_body)
```

World/body axes are right-handed, world Z points upward, thrust acts along body +Z, and R maps body to world. PyBullet returns xyzw quaternions and world-frame angular velocity; controller feedback rotates angular velocity into body axes. No normal-flight position or velocity resets are used.

The cascaded controller limits world-frame velocity references, produces PD acceleration commands, optionally limits command jerk, adds gravity compensation, limits desired tilt, and computes quaternion attitude error. A bounded body-rate reference and angular damping produce saturated torques. Collective thrust is projected onto current body +Z and saturated. No integral controller or MPC is implemented. Zero velocity brakes toward hover rather than restoring an exact position.

Default physical parameters are mass **1.2 kg**, collision-sphere radius **0.55 m**, diagonal inertia **0.1452 kg·m²** (uniform solid sphere), and gravity **9.81 m/s²**. Visual scale **4** affects appearance only. Default command bounds are horizontal speed 5 m/s, vertical speed 2.5 m/s, horizontal acceleration 3 m/s², vertical acceleration 4 m/s², roll/pitch/total desired tilt .35 rad, body roll/pitch reference rates 3 rad/s, yaw reference rate 1.2 rad/s, thrust 25.8984 N (2.2 mg), and torque 3 Nm per axis. Velocity PD gains are 1.5 and .05; attitude angular-acceleration gains are 9 and 3.5. Optional thrust/per-axis torque limits override the legacy defaults; jerk limiting defaults off. Configure these through `DroneConfig`.

Collisions combine physical contacts with conservative segment/AABB event checks. The latter may flag early contact near corners/curved geometry. Moving obstacles still lack swept collision checks. Contacts and wind can exceed commanded speed, acceleration, tilt and rate references; those references are not hard physical-state clamps.

Battery drain per second combines baseline hover, squared speed, squared acceleration and positive climb speed. Charge is clamped to [0,1]. Reward components are progress, arrival (once), collision, proximity, energy, time and command effort. Configuration coefficients are initial defaults, not tuned results.

## Sensors

Each UAV owns independent seeded sensor caches, reset each episode:

- IMU: 6 values, world-frame linear acceleration and angular velocity; excludes gravity/specific force. Default 50 Hz, Gaussian noise, bias and random-walk bias drift.
- GPS: local-world Cartesian position, default 10 Hz, Gaussian noise and bias.
- LiDAR: default 36 horizontal rays across 360°, elevations -20/0/20°, 30 m, 10 Hz. Batched PyBullet ray casts provide truth, noisy metres and normalized ranges. Origins outside the collision sphere avoid self-hits but introduce a near-body blind zone.
- RGB-D: disabled by default; configurable 64×64 at 5 Hz, 75° field of view, near .1 m/far 60 m. Body-mounted +x camera uses TinyRenderer; depth buffers convert to camera-axis metres.

Frequencies cannot exceed physics frequency (120 Hz by default). Acquisition occurs on physics ticks at or after scheduled deadlines. Camera and LiDAR costs increase with fleet size; no real-time guarantee is made.

## Tests and benchmarks

```bash
python -m pytest -q
python validate.py --steps 100 --output outputs/validation.csv
python validate.py --steps 100 --moving 2 --sensor-modes lidar --output outputs/moving.csv
```

The validator runs every combination of 3 scenes, fleets 2/4/8/16/20 and sensor modes `state_only`, `lidar`, `rgbd`. It repeats resets to check reproducibility and verifies initial scenery clearance. CSV and JSON include arrivals, team success, collision agent-events by category, minimum sampled inter-UAV distance, per-agent travel/arrival/energy, rewards, simulated duration and steps/second. Minimum distance is sampled at control steps; collision pair-events may count both UAVs. Unclassified endpoint collisions are reported as `collision_other`. Setup/reset time is excluded from throughput; sensor modes include all other work. Short runs are smoke benchmarks, not stable performance claims.

Potential bottlenecks include per-agent proximity queries over every body, conservative collision broadphase/narrowphase checks, copying raw sensor images into infos, ray casting, rendering, and Python loops. Compare repeated longer runs before drawing performance conclusions.

## Validation limits and future integration

The environment provides reusable action/observation spaces and parallel lifecycle components for later IPPO, MADDPG, MAPPO or MASAC integration. Readiness is conditional: add the target framework adapter, verify sustained multi-UAV missions, moving-obstacle sweeps, disturbance bounds, reward suitability and GUI behavior. It is not yet certified for reliable collision-free navigation or research comparisons. No MARL implementation is included.

Executed benchmark results and validation limitations are recorded in [outputs/VALIDATION.md](outputs/VALIDATION.md). CSV/JSON results include the 100-step sensor-mode matrix and moving-obstacle smoke runs.

## Rigid-body controller integration

UAV motion now uses positive-mass PyBullet rigid bodies and the cascaded velocity/attitude controller. `control_hz` is the MARL action rate (default 20 Hz), `low_level_hz` the controller rate (120 Hz), and `physics_hz` the integration rate (120 Hz). Physics frequency must be a multiple of controller frequency, itself a multiple of policy frequency. Controller wrenches are held between updates and reapplied each physics tick. IMU, battery and collision updates use post-physics states. The earlier analytical kinematic equations and benchmark numbers describe the prior implementation and are not current dynamics/performance guarantees.

## Controller constraints

`DroneConfig` retains `max_speed_xy`/`max_speed_z` (commanded m/s), `max_accel_xy`/`max_accel_z` (desired m/s²), and adds `max_roll`/`max_pitch` (rad), `max_angular_rate` (body roll/pitch reference rad/s, with existing `max_yaw_rate` for yaw), optional `max_collective_thrust` (N), `max_body_torques` (three Nm limits), and `max_jerk` (Euclidean commanded acceleration slew in m/s³). None for thrust/torques retains the existing factor/scalar limits; None disables jerk limiting. A conservative tilt cone uses the smallest roll, pitch and total tilt bound. Rate references are saturated before damping torque generation. These constrain references and actuator outputs, not physical state under contact/wind. No position/velocity clipping or DCAD MPC was introduced. Wind acceleration is multiplied by mass and applied as world-frame force at the centre of mass each physics tick.

## Executed rigid-body integration diagnostics

Run `python diagnose_dynamics.py` to save `outputs/dynamics/traces.csv` (position, velocity, roll/pitch/yaw, command, tracking error, thrust, body torque) and `summary.json`. Physics diagnostics use 240 Hz; fleet tests use the default 120 Hz physics/controller and 20 Hz policy rates with LiDAR enabled and camera disabled.

Executed results: ideal hover drift 0 m over 8 seconds; directional/command-change tests averaged 0.0153 m/s tracking error in the final second of each segment. Small roll/pitch/yaw disturbance recovery finished with 0.000150 rad attitude error and 0.000662 m/s steady velocity error. Its net position drift was 0.506 m; this controller is not a position holder. Constant world X wind acceleration .2 m/s² produced 0.1334 m/s residual velocity and 1.292 m displacement over 10 seconds. No divergent/oscillatory instability was observed in these tests; broader disturbances remain unvalidated.

Headless open-field hover throughput over 100 environment steps: fleets 2/4/8/16/20 achieved approximately 616/379/190/93/73 steps per second. Ideal hover drift was zero in all five runs. These short local timings exclude reset and overlap other validation activity; they are not sustained performance guarantees. Diagnostic plant throughput includes trace collection and should not be compared directly with environment throughput.

Integration tests also exercise actual UAV/UAV and UAV/static-box contact termination, source-to-destination travel, action validation, fixed observations, individual termination, truncation and reset. GUI behavior was not revalidated for this dynamics change. Older reports in `outputs/VALIDATION.md` describe the earlier kinematic implementation.

## Optional integral wind rejection

Select `DroneConfig(velocity_mode="pid")` for outer velocity PID, or `"pd"` for the unchanged default baseline. Integral parameters: `velocity_ki=.4`, `velocity_integral_limit=1.0` (integrated velocity error in metres per axis), `velocity_integral_error_gate=.3` m/s. The integral is bounded and conditionally frozen/rejected at acceleration, tilt, thrust, jerk or torque constraints. Controller reset and environment reset clear its state. Inner attitude control is unchanged.

Run `python compare_wind.py` for matched deterministic tests; see [wind comparison](outputs/dynamics/WIND_COMPARISON.md). Constant-wind 10-second displacement fell from 1.293 m to .480 m; final-second velocity error at 20 seconds fell from .1333 to .000274 m/s. PID has modest no-wind transient regressions, so PD remains selectable and default. This improves velocity rejection, not exact position holding.

## Profiled performance optimization

See [outputs/benchmarks/REPORT.md](outputs/benchmarks/REPORT.md) for matched baseline/optimized measurements, memory, timings and limitations. At 20 UAVs, cProfile-instrumented throughput improved from 42.5 to 60.8 steps/s (43%); with RGB-D and two moving obstacles, 37.1 to 49.5 steps/s. These are profiled short runs, not directly comparable to earlier unprofiled numbers. The only simulation changes specialize three-vector cross products and cache LiDAR directions. All ten deterministic fingerprints matched exactly and 96 tests passed.

Reproduce with `python benchmark_step.py --label repeat --steps 100` or add `--camera --moving 2`. CSV/JSON and detailed profiles are saved under `outputs/benchmarks/`.

## Visual validation and deterministic demo

`--scripted-demo --seed 42` runs fixed velocity commands: climb for 2 s, hover 2 s, forward 2 s, lateral 2 s, reverse 2 s, then stop. This begins from the existing airborne spawn, not a ground launch; it is not a planner and does not guarantee arrival. Arrival is checked separately by integration tests. `--velocity-vectors` draws measured world velocity as a one-second green displacement. `--directions` shows destination vectors; `--trails` shows expiring recent paths. All debug commands are guarded by GUI mode.

Manual GUI validation (repeat with fleets 2, 8, 20 and scenarios urban, open_field, cluttered):

```bash
python main.py --scenario urban --num-drones 8 --num-moving-obstacles 5 --seed 42 --steps 280 --scripted-demo --render --realtime --velocity-vectors --directions --trails --overview
python main.py --scenario open_field --num-drones 20 --seed 42 --render --realtime --directions
python main.py --scenario cluttered --num-drones 2 --seed 42 --render --realtime --velocity-vectors
```

Automated visual checks: `python -m pytest -q tests/test_visuals.py`. Nine scene/fleet combinations test TinyRenderer image initialization and rigid-body visual transforms; another test checks that scale leaves inertia and collision geometry unchanged. The full suite passed 106 tests before adding the optional camera flag. Saved 20-UAV renders are in `outputs/visuals/`.

Known visual distinctions: the requested close camera crops some town boundary corners; use `--overview` for a wider view. The indoor ceiling can obscure interior objects from above; orbit the camera inside the room for inspection. TinyRenderer images do not include GUI debug region/trajectory lines, so green/blue region boundary visibility requires GUI review. Enlarged arms/rotors extend beyond the .55 m physical sphere and can visually intersect scenery or each other without a physical collision. Collision contacts are tested against the physical envelope, not enlarged decorative geometry. Arrivals intentionally replace the drone model with a stationary non-colliding green sphere. No geometry or dynamics was changed to conceal these effects.

Interactive GUI inspection was not performed in this tool session; headless rendering is validated, but camera usability, GUI region colors and interactive closing still require manual verification.

## Final pre-MARL validation

See [outputs/benchmarks/FINAL_PRE_MARL.md](outputs/benchmarks/FINAL_PRE_MARL.md) for the freshly executed 106-test result, PD/PID comparison, two matched 200-step fleet benchmarks, RGB-D workload, physics/sensor profiles, deterministic fingerprints and remaining limitations. Start integration with PD, open field, 2–4 UAVs, 20 Hz actions/120 Hz controller/120 Hz physics, LiDAR enabled and camera disabled. PID is available for wind experiments with documented transient trade-offs. Interactive GUI remains unverified; headless rendering tests passed.

## Pure MADDPG baseline

The frozen environment is now used by a separate vanilla MADDPG implementation. Actors use local 157D observations and produce 3D normalized actions. Each UAV has an independent actor, centralized critic and target pair; the critic uses a deterministic 8N+3 public global-state projection and 3N joint actions. Joint replay masks preserve retired agents and distinguish true termination from time-limit truncation. No attention, parameter sharing, twin critics or entropy terms are used.

Algorithm modules:

```text
config/maddpg_config.py             algorithm-only dataclass
algorithms/registry.py             shared learner factory/checkpoint loader
algorithms/env_adapter.py          public parallel API integration
algorithms/evaluation.py           shared noise-free metrics/evaluation
algorithms/maddpg/networks.py       independent actor/critic/targets
algorithms/maddpg/replay_buffer.py  bounded joint replay
algorithms/maddpg/maddpg.py         vanilla updates and checkpoint state
train.py                          training runner
validate.py                       checkpoint validation (legacy benchmark mode retained)
test.py                           deterministic GUI/headless playback
```

Install the existing simulator requirements and `python -m pip install -r algorithms/maddpg/requirements.txt` for PyTorch. See [algorithm documentation](algorithms/maddpg/README.md) for full options and limitations.

```bash
python train.py --algo maddpg --scenario open_field --num-drones 2 --device cpu   --total-steps 24 --episode-steps 8 --warmup-steps 4 --batch-size 4   --checkpoint-interval 8 --output outputs/training/e2e_check
python validate.py --algo maddpg --checkpoint outputs/training/e2e_check/final.pt   --seeds 100000 100001 100002 --output outputs/validation/e2e_check
python test.py --algo maddpg --checkpoint outputs/training/e2e_check/final.pt   --headless --seed 200000 --output outputs/testing/e2e_check
# Optional manual GUI playback:
python test.py --algo maddpg --checkpoint outputs/training/e2e_check/final.pt --render --realtime
```

[End-to-end verification report](outputs/benchmarks/MADDPG_END_TO_END.md) records executed tests, parameter counts, tensor shapes and smoke metrics. A 24-step run performed 21 updates and loaded successfully for validation/testing. This is functional verification only: no convergence, collision-free navigation or long-run stability is established. No full-scale training was started.

## Experiment result naming

New MARL runner outputs use `utils/results.py`; existing artifacts are never moved or renamed. `--output` now selects an output ROOT directory, not a filename prefix. New runs are nested as:

```text
outputs/<algo>/<scenario>/n<drones>/<unique_run_id>/
```

The filename template is:

```text
<algo>_<scenario>_n<drones>_ep<episodes>_seed<seed>_run<run_id>[_step<actual_steps>][_seeds<seed_set_hash>]_<result_type>.<ext>
```

Tokens are lowercase/underscores; `open_field` is serialized as `openfield`. The full UUID run ID prevents repeated configurations from overwriting one another and is generated without consuming experiment RNG state. Training names use the actual number of completed episodes at artifact creation, NOT the requested step budget. Partial episodes remain marked incomplete in logs and do not increment that count. Streaming logs can span episode-count snapshots; final training CSV/JSON covers the current invocation. Resumed counters are cumulative, while parent-run metadata identifies earlier segments. Evaluation names use the actual evaluation episode count, never the training episode count; the seed token identifies the model's training seed and the seed-set hash identifies the ordered evaluation seed list.

Result types include `training` CSV/JSON/JSONL, `actor_critic_loss` CSV, streaming `loss` JSONL, `validation` CSV/JSON, `test` CSV/JSON, `reward_plot`, `success_rate_plot`, `collision_plot` SVGs, `checkpoint`/`final_checkpoint` PT, `summary` JSON and `metadata` JSON. SVG plots use logged metrics only, require no plotting dependency, and do not affect training. Success-rate plots show cumulative success fractions for the rows in that invocation. Evaluation data live in their own unique directories; scheduled evaluations are nested below the originating training run's evaluations directory. The legacy non-MARL simulation benchmark mode and direct low-level checkpoint APIs remain compatible and retain caller-specified paths.

Metadata records algorithm/environment dataclasses, scenario, fleet, seeds or deterministic seed rule, training budget, run identifier, and a SHA256-based reward-version identifier from env/rewards.py. Checkpoint training metadata stores the originating run ID/directory. External evaluation metadata records resolved checkpoint path, SHA256, checkpoint step, origin run when present, evaluation seeds and episode count. Legacy checkpoints without run IDs remain loadable and are linked by resolved path and content hash. Resume creates a NEW run segment with parent checkpoint/run provenance, preserving old files and optimizer/exploration/replay behavior. Its filename episode/step counters remain cumulative.

Find the final checkpoint via the returned training summary's `final_checkpoint` field, or the `*_summary.json` artifact; do not assume a literal final.pt filename. `checkpoint_paths` lists all saves from the invocation. To evaluate:

```bash
python validate.py --algo maddpg --checkpoint <full_checkpoint_path> --output outputs
python test.py --algo maddpg --checkpoint <full_checkpoint_path> --headless --output outputs
```

No standalone training experiment was started to introduce naming. Existing test-suite smoke cases exercise output creation, deterministic repeatability and resumed checkpoints in temporary directories. The naming scheme is independent of learning architecture and registry algorithm names; adding future algorithms does not require changing filenames in the runners.

Training also accepts `--building-density` (urban eligible-plot fraction, default .55) and `--episodes` (absolute completed training-episode cap, including resumed episodes). If --total-steps is omitted, the safety cap is 1000 steps without --episodes, or min(max_training_steps, episodes × episode_steps) with it. If both budgets are supplied, training stops at whichever is reached first. Actual completed counts—not requested episode budgets—are used in result filenames. Repeat the same density/scenario/fleet/episode-length options when resuming a checkpoint to satisfy environment compatibility checks.

```bash
python train.py --algo maddpg --scenario urban --num-drones 2 --building-density 0.65 --episodes 1000 --seed 42 --device cpu
```

This command can run a substantial experiment; only parsing and short temporary test-suite smoke cases were executed when adding these flags.

### Consolidated data and high-DPI figures

New runs write one consolidated training episode CSV and one actor/critic loss CSV, plus metadata/summary JSON. Per-episode JSONL fragments and duplicate episode-data JSON are no longer generated by the MARL runners. Validation/testing export one episode CSV and aggregate JSON per invocation; scheduled validation episodes are combined into one report per training invocation with evaluation_step in every row. Each metric has one 300-DPI PNG with episode X axis, metric Y axis, grid and legend. Training plots include team reward, cumulative success rate, colliding-UAV count, and per-agent actor/critic losses averaged by episode. Validation/testing have reward/success/collision plots but no network-loss plot, since inference does not calculate training losses. Artifacts are finalized at run completion; checkpoints retain their existing periodic schedule. Repeated independent invocations remain uniquely identified to prevent overwriting. Historical files are not converted.

## Corrective-experiment preparation

Read-only algorithm-side episode telemetry now records action means/saturation, initial/min/final destination distances, progress, final statuses, component returns, mean per-agent losses and sampled critic Q summaries. `--telemetry-q-interval 20` controls read-only Q sampling; 0 disables it. See [the controlled plan](outputs/benchmarks/corrective_experiment_plan/PLAN.md) for fields, independent ablations and the unimplemented boundary-penalty proposal.

`python benchmark_feasibility.py` runs only the existing rule controller. Three fixed-seed urban routes completed in61.85–65.45s, supporting a provisional1600-step/80s cap instead of600steps/30s. Frozen defaults were not changed. Prepared gamma .95/.99/.995 JSON manifests under `experiments/maddpg/` include CLI commands and matched seeds; they are not auto-loaded and no extended training has been run.
