# Pure MADDPG scaffolding

No networks, optimizers, replay buffers, exploration, training or evaluation logic exist yet. Registry factories return specifications, not trained agents. No new dependencies are required at this stage.

```bash
python train.py --algo maddpg --scenario urban --num-drones 2 --seed 42
python validate.py --algo maddpg --scenario open_field --num-drones 4 --seed 42
python test.py --algo maddpg --scenario cluttered --num-drones 8 --seed 42
```

These commands print preparation metadata only. Existing `python validate.py` benchmark arguments continue to work when `--algo` is absent. `--algo maddpg --help` shows the shared algorithm CLI; bare `validate.py --help` shows the original benchmark CLI. Use the existing qmarl311 environment for commands importing the simulator.

## Frozen environment contract

```python
from config import EnvConfig
from env import MultiDroneEnv

with MultiDroneEnv(EnvConfig(gui=False, num_drones=2)) as env:
    observations, infos = env.reset(seed=42)
    actions = {a: env.action_spaces[a].sample() for a in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
```

Constructor: `MultiDroneEnv(config=None)`. Reset: `reset(self, *, seed=None, options=None)`; only None/empty options are supported. Step: `step(self, actions)`. Render: `render(self)` (GUI live; returns None). Close: `close(self)` (idempotent).

IDs are `uav_0` through `uav_{N-1}` in `possible_agents`; `agents` holds active agents. Do not lexicographically reorder IDs for future joint critic/replay layouts: use possible_agents order. Require exactly one finite length-three action in [-1,1] per active agent. Gymnasium action Box is float32, shape (3,). Actions map to world-frame desired velocities with configured speed limits.

Default observation Box is float32 [-1,1], shape (157,):

| Slice | Content |
|---|---|
| 0:3 | GPS-derived ego position |
| 3:6 | Ego velocity |
| 6:8 | sin/cos yaw |
| 8:9 | Actual normalized battery |
| 9:12 | Destination minus GPS position |
| 12:36 | Four neighbour relative position/velocity slots |
| 36:40 | Neighbour validity masks |
| 40:148 | 36 rays × 3 layers normalized LiDAR |
| 148:154 | IMU acceleration and angular velocity |
| 154:157 | GPS position |

All values are normalized and clipped by the frozen encoder. Local neighbours are radius-filtered, distance/ID sorted and padded. Dimensions depend on K/ray configuration (`21+7K+L`), not fleet size. Future MADDPG must validate the default dimensions rather than silently accepting incompatible configuration changes. Raw RGB-D and sensor metadata are in infos or get_sensor_readings(agent), outside actor observations.

`global_state()` returns a privileged dictionary of UAV states, static/moving obstacles and destination; it is not a fixed numerical tensor. Pure MADDPG centralized critic inputs should be defined separately later, without feeding privileged state to decentralized actors. Joint dimensions depend on fleet size even though local observations do not.

Step outputs use IDs active at step start, including their final observations. Finished agents leave agents immediately; no further actions are allowed. Termination/truncation dictionaries additionally contain `__all__`; never treat it as an agent. Time limits truncate; success, collision, boundary failure and configurable depletion terminate. Team success requires all UAVs to arrive safely. Arrived bodies become non-colliding markers. Episodes may end with mixed earlier terminations and remaining truncations. Future replay must retain final observations, distinguish masks for termination vs truncation, and account for inactive agents without stale actions.

## Conventions and scope

Configuration uses dataclasses in config/environment_config.py, frozen for this baseline. Existing logging uses JSON/CSV in main.py, validate.py and outputs/; tests use pytest with simulator tests skipped when dependencies are absent. Algorithm-side configuration/logging should be separate. Registry is algorithms/registry.py and shared preparation CLI is algorithms/cli.py. No environment modules were changed in this scaffolding step.

## Algorithm configuration

`config/maddpg_config.py` contains `MADDPGConfig`, separate from frozen `EnvConfig`. User-supplied initial paper defaults: actor/critic LR .001, gamma .95, tau .01, hidden layers (256,256,64), Adam. All are configurable; they are not fixed experimental settings. No specific paper citation was supplied or independently verified here.

Development defaults: replay capacity 100,000, batch 256, warm-up 1,000 transitions, update every transition, initial action noise std .2, multiplicative decay .99999 after warm-up, minimum .02, training budget 1,000,000 transitions, validation/checkpoint every 10,000 transitions, seed 42, CPU. Intervals count environment transitions, not per-agent samples. Noise parameters use normalized action units. No buffers or schedules are implemented yet.

```bash
python train.py --algo maddpg --actor-lr 0.0005 --critic-lr 0.001 \
  --batch-size 128 --replay-capacity 200000 --warmup-steps 2000 \
  --max-training-steps 500000 --device cpu --seed 42
python validate.py --algo maddpg --evaluation-episodes 20 --device mps
python test.py --algo maddpg --device cuda
```

Other overrides: `--gamma`, `--tau`, `--hidden-layers 256 256 64`, `--optimizer Adam`, `--policy-update-frequency`, `--exploration-noise-std`, `--exploration-noise-decay`, `--minimum-exploration-noise`, `--validation-interval`, and `--checkpoint-interval`. Configurations validate before preparation metadata is emitted. Training/validation/testing share the same config schema; validation/test metadata disables exploration and uses evaluation_episodes. There is no evaluation/training execution yet.

Device selection accepts cpu/cuda/mps/auto. It records a request only; accelerator availability and auto resolution will be implemented when a tensor backend is introduced. No PyTorch or other neural-network dependency is added now.

## Networks (implemented; training remains deferred)

Install algorithm-only dependencies with `python -m pip install -r algorithms/maddpg/requirements.txt`. `algorithms/maddpg/networks.py` provides `Actor`, `Critic`, `MADDPGNetworks`, `hard_update(target, source)` and `soft_update(target, source, tau)`.

```python
from config.maddpg_config import MADDPGConfig
from algorithms.maddpg.networks import MADDPGNetworks

cfg = MADDPGConfig()
models = MADDPGNetworks(num_agents=8, global_state_dim=104, config=cfg)
# 104 is an EXAMPLE state-vector size (8 × 13), not a frozen environment encoder.
# Define and validate a stable numeric global-state encoder before learning.
models.to('cpu')
```

Actors accept only local observations (...,157), output (...,3), and have ReLU hidden layers plus tanh outputs. Critics accept numeric global state (...,state_dim) and flattened joint actions (...,3N) in possible_agents order, returning (...,1). Each agent owns a distinct actor and single critic. Targets are deep copies, frozen and kept in eval mode; online parameters remain trainable. Configured device is not resolved by constructors: explicitly move the container and tensors to the selected available device. Random initialization follows PyTorch's RNG; seed it explicitly at the future experiment entry point. No optimizer, update loop, attention, twin critics, entropy regularization, or parameter sharing is implemented.

## Centralized replay

`ReplayBuffer(capacity, agent_ids, observation_dim=157, global_state_dim=...)` stores one-step joint transitions in the exact supplied agent order (pass `env.possible_agents`). It allocates payload only as transitions arrive and overwrites old transitions at capacity. Numerical data are float32, flags/masks bool. Sampling is uniform without replacement using a private seeded NumPy generator and returns independent PyTorch tensors.

Call `add(observations, global_state, actions, rewards, next_observations, next_global_state, terminated, truncated)`. Capture `active = list(env.agents)` and `executed_actions` BEFORE `env.step()`. Filter pre-step observations to those active agents (the prior step may contain final observations for retired agents). Pass exactly the actions actually sent to step, after any future exploration/clipping. Replay never reconstructs or clips actions. The buffer cannot independently verify that caller-provided actions were executed.

All maps must contain exactly the pre-step active IDs; returned post-step observations include final observations. `__all__` flags are ignored for agent axes; episode_done derives from per-agent flags. Missing inactive agents receive zero slots. `valid_agents` marks pre-step active slots, `next_valid_agents` marks available next/final observations, `next_active_agents` excludes newly finished agents. `bootstrap_mask = valid_agents & ~terminated` preserves time-limit bootstrapping; do not multiply it by next_active_agents or truncation would incorrectly suppress bootstrapping. A future learner must define next joint actions consistently for terminal and truncated slots. No reset observation may replace a final next observation. Each add is independent, so episode boundaries never connect automatically. `clear(seed=...)` optionally reseeds sampling; ordinary episode reset must NOT clear replay.

Sample shapes: observations [B,N,157], actions [B,N,3], global states [B,S], rewards/flags/masks [B,N], episode_done [B]. Flatten joint actions in canonical order for the critic. `nbytes` reports NumPy payload only, excluding Python overhead. Large joint replay can consume substantial memory (100,000 × 20 × 2 × 157 float32 observations alone is about 2.51 GB). No compression, prioritized sampling, n-step data, or learning updates are included.

## Vanilla learner (implemented)

`algorithms/maddpg/maddpg.py` implements `MADDPG(agent_ids, global_state_dim, config, observation_dim=157)`. It owns independent actors, critics, frozen targets, per-agent Adam optimizers, replay, Gaussian exploration RNG and counters. It introduces no environment changes or reward shaping.

- `select_actions(local_observation_dict, training=False)` returns only supplied active IDs; evaluation is deterministic. Training adds independent Gaussian action noise, then clips to [-1,1].
- `record_transition(...)` forwards exact executed actions and final next observations to replay, increments environment steps and decays exploration noise after warm-up.
- `update()` samples a joint minibatch after warm-up and at policy_update_frequency. Returns None if insufficient data/not due. `update(batch)` performs an explicit diagnostic update without scheduling checks.
- Targets are r_i + gamma*(1-terminated_i)*Q_target_i(s_next, joint_target_actions). Time-limit truncations bootstrap from final observations. Truly terminal/inactive agent target actions are zero. Inactive current slots are masked from individual losses.
- Critics minimize masked MSE. Actor i maximizes critic i with its current action and detached replay actions of all other agents. Critic parameters are frozen during actor optimization; target computations have no gradients. Targets receive Polyak updates after each agent update.
- Optional `gradient_clip_norm` (None by default, CLI `--gradient-clip-norm`) clips actor/critic gradient norms. No algorithm variants are included.

```python
from algorithms.maddpg.maddpg import MADDPG
from config.maddpg_config import MADDPGConfig
learner = MADDPG(['uav_0', 'uav_1'], global_state_dim=26,
                config=MADDPGConfig())
# 26 is an illustrative numeric state width, not an implemented state encoder.
learner.save_checkpoint('outputs/checkpoints/maddpg.pt')
restored = MADDPG.load_checkpoint('outputs/checkpoints/maddpg.pt', device='cpu')
```

Checkpoints include online/target weights, all optimizer states, configuration, canonical IDs/dimensions, replay contents/cursor/sampling RNG, exploration RNG/noise, and step/update counters. Load only trusted files; NumPy replay payloads use torch pickle. Checkpoint restoration reproduces the next CPU learning update in tests. It does not snapshot the PyBullet world or guarantee mid-episode simulator resume; resume collection at a fresh episode boundary. GPU/MPS numerical determinism is not certified.

The shared train/validate/test entry points still prepare metadata only; end-to-end rollout scheduling, state dictionary encoding, warm-up random-action collection and experiment logging remain future integration work. The registry's specification flag still refers to that deferred executable workflow. No training experiment was run here.

Reference: https://ieeexplore.ieee.org/document/11644773/ (Energy-Aware Multi-UAV Collaboration for Data Collection and Trajectory Planning With MADDPG). IEEE's access check prevented reading Section IV / Algorithm 1 during implementation. This code follows the requested vanilla MADDPG equations; exact paper-specific agreement is unverified. The paper's data-collection reward is not imported into the frozen navigation environment.

## Environment integration layer

`algorithms/env_adapter.py` provides `ParallelEnvAdapter`, using only public environment attributes/methods. It forwards reset/step returns without altering their rewards, observations or flags, and captures an independent `last_transition` ready for `learner.record_transition(**adapter.last_transition)`.

Canonical ordering always uses `env.possible_agents`. `active_observations()` filters stale final observations before action selection; `ordered_observations()` returns padded [N,O] values plus [N] active masks. `action_dictionary()` accepts canonical [N,3] arrays (filters inactive slots) or exact active-ID dictionaries. It validates rather than clips action values, so recorded float32 actions are exactly those sent to the environment.

Centralized state uses the public `global_state()` accessor: per agent, concatenate world position(3), velocity(3), battery(1), active flag(1), then destination(3), yielding **8N+3** float32 values in SI units. Static/moving obstacle lists and body IDs are deliberately omitted; this is a documented fixed projection, not a complete world-state encoding. Actors still receive only their local observations. This projection avoids scene-dependent critic dimensions; its sufficiency for learning is an experimental question.

If no accessor is available, concatenate canonical padded local observations and validity masks. Final observations remain present in transition next-state encoding even on truncation, preserving time-limit bootstrap inputs. On the subsequent step, retired slots are zeroed. `last_transition` retains final per-agent observations and separate termination/truncation dictionaries; no actions are emitted for retired agents. Reset clears adapter state but not a learner's replay. Adapters are not automatically interchangeable with arbitrary global-state schemas: another environment with a differently structured accessor needs its own documented encoder.

```python
from config import EnvConfig
from env import MultiDroneEnv
from algorithms.env_adapter import ParallelEnvAdapter
from algorithms.registry import create_algorithm

with MultiDroneEnv(EnvConfig(gui=False, num_drones=2)) as env:
    adapter = ParallelEnvAdapter(env)
    adapter.reset(seed=42)
    learner = create_algorithm('maddpg', adapter=adapter)
    actions = learner.select_actions(adapter.active_observations())
    adapter.step(actions)
    learner.record_transition(**adapter.last_transition)
```

Registry `create_algorithm(name, adapter=..., config=...)` creates the actual learner. Calling without an adapter preserves legacy specification lookup. Shared `train.py`, `validate.py --algo maddpg` and `test.py` now initialize/reset one headless environment and create the learner, then close resources; no rollout or training loop executes. The legacy `implemented` metadata describes the deferred entry-point workflow, while `learner_created` confirms learner construction. Use `--device cpu/cuda/mps/auto` for actual device resolution during creation. Existing benchmark-mode validate.py behavior remains unchanged without --algo.

## Training entry point (implemented)

`train.py` now collects rollouts and performs learning updates using the registry, public adapter and MADDPG learner. Default initial task: two UAVs, open field, PD controller, headless, camera disabled, no moving obstacles. `--total-steps` is an absolute environment-transition budget (default 1000 for safety), not an additional resumed budget; `--episode-steps` defaults to 600. All MADDPG dataclass options have CLI overrides. Configuration's max_training_steps remains a hard upper budget. No environment source/config definitions are edited.

Short smoke run with actual replay and gradients:

```bash
python train.py --algo maddpg --num-drones 2 --scenario open_field --seed 42 --device cpu \
  --total-steps 12 --episode-steps 6 --warmup-steps 4 --batch-size 4 \
  --hidden-layers 16 8 --checkpoint-interval 6 --validation-interval 1000 \
  --output outputs/training/smoke
python train.py --algo maddpg --total-steps 16 --episode-steps 6 \
  --resume outputs/training/smoke/final.pt --output outputs/training/smoke_resume
```

Warm-up uses seeded uniform normalized actions. After warm-up, local actors add training-only Gaussian noise. Replay receives exactly executed actions and final observations through the adapter. Update frequency/batch size are enforced by the learner. Independent counters track joint environment transitions, individual agent transitions, started/completed episodes, gradient updates and elapsed wall time. JSONL logs include per-agent/team reward, success/collision agents, episode lengths, losses and validation results. Config and summary JSONs accompany periodic checkpoint_N.pt and final.pt files.

Scheduled validation calls reusable `algorithms/evaluation.py:evaluate` in a separate environment with deterministic actors and held-out seeds; no validation implementation or testing workflow is embedded in the trainer. Validation time is included in wall-clock totals. Checkpoint intervals count environment steps, not updates.

Resume restores online/target weights, Adam states, replay and sampling RNG, exploration state, counters, Python/NumPy/Torch RNGs and CUDA RNGs when available. Loaded algorithm hyperparameters take precedence over CLI hyperparameters except the requested runtime device. Environment configuration must match the checkpoint. Collection resumes at a NEW seeded episode boundary; it does not restore mid-episode PyBullet state. Partial-budget episodes are logged as incomplete and are not artificially converted into truncated transitions. Replay may contain their valid one-step transitions. Checkpoints are saved in finally during an in-loop interruption/error. Use a distinct output directory for each fresh run to avoid appending unrelated logs. CPU repeatability is regression-tested; cross-device deterministic equivalence is not promised.

Executed smoke run: 12 environment / 24 agent transitions, 9 updates, periodic and final checkpoints; resume to 16 environment / 32 agent transitions and 13 updates. This checks functionality only, not learning performance. No million-step experiment was executed.

## Deterministic checkpoint validation

```bash
python validate.py --algo maddpg --checkpoint outputs/training/smoke/final.pt \
  --scenario open_field --num-drones 2 --seeds 100000 100001 100002 \
  --output outputs/validation/maddpg
```

Algorithm validation now requires a checkpoint; the earlier metadata-only `validate.py --algo maddpg` path has been replaced. Without --algo, validate.py retains its original simulation-benchmark mode. `--device` defaults cpu, validation is headless, and no exploration, replay insertions or network updates occur. Network modes are temporarily set to eval and restored; gradients are disabled. Environment parameters are restored from trainer checkpoint metadata where available; otherwise open field defaults are used. Optional --scenario, --num-drones and --episode-steps override those instance settings. Fleet/order/observation/global-state dimensions must match the checkpoint; selecting another fleet requires a compatible checkpoint.

Default held-out seeds are 100000–100004. Seeds must be unique, nonnegative and disjoint from known training seeds. Trainer metadata identifies previously used sequential training episode seeds; without it only the learner seed is known, so callers must supply additional training_seeds to the reusable evaluator if necessary. Selection of another scenario is an intentional cross-scenario test and is recorded in rows.

Reusable `algorithms.evaluation.evaluate(learner, environment_config, episodes=..., seed=..., seeds=..., output=..., training_seeds=...)` is used by train.py for scheduled validation. Each listed seed produces one COMPLETE episode, ending through the unchanged environment's termination/truncation behavior. Registry `load_algorithm(name, checkpoint, device)` encapsulates algorithm loading; evaluators need canonical IDs/dimensions and a deterministic select_actions(local_observations, training=False) interface, not MADDPG loss logic.

Outputs: `<output>.csv` contains per-seed episode rows; JSON contains those rows and pooled aggregates. Metrics are team success rate, individual arrival fraction, collision count (unique colliding UAV events, NOT unique physical contact pairs), mean successful arrival time (simulated seconds), mean team episode reward, mean team normalized battery use, successful path efficiency and mean episode duration. Travel distances are accumulated at control-step boundaries. Efficiency is shortest distance from initial position to the sphere-clearance destination volume divided by traveled distance, capped at 1; it ignores obstacles. Mean travel time and efficiency are null when nobody succeeds, rather than treating failures as fast/efficient arrivals. Per-agent rewards, distances, arrival times, energy and efficiency are retained in each row. Failed paths do not enter successful-path efficiency averages; interpret alongside arrival rate. Metrics are pooled over equally weighted one-episode seeds.

Validation regression tests verify deterministic repeated output, no gradient/parameter/RNG/replay/counter changes, mode restoration, correct checkpoint load, seed overlap rejection and metric aggregation. The short smoke checkpoint's two held-out episodes reported zero arrivals/collisions, not evidence of a trained policy. No simulator source/configuration definitions were changed.

## Trained-model GUI/headless testing

```bash
python test.py --algo maddpg --checkpoint outputs/training/smoke/final.pt --render --realtime
python test.py --algo maddpg --checkpoint outputs/training/smoke/final.pt \
  --headless --scenario open_field --num-drones 2 --seed 200000 --episodes 5 \
  --output outputs/testing/maddpg
```

Testing uses the shared registry loader, public adapter, checkpoint environment restoration and deterministic evaluator. Headless is the default; --render activates the existing GUI and original camera/visual configuration. --realtime enables the environment's existing GUI pacing. No manual navigation controls or renderer changes are introduced. Existing moving obstacles are shown if enabled in the checkpoint's environment configuration; testing does not add obstacles to a baseline that had none. Likewise, town/region/UAV visuals are generated by the unchanged environment. Fleet size must match checkpoint dimensions. --episode-steps optionally changes the episode limit.

Each complete episode prints per-agent status/arrival/collision and team success. CSV/JSON metrics default to outputs/testing/maddpg, separate from training/validation. After GUI episodes complete, metrics are saved and the final window remains open until closed or Ctrl+C; --close-on-finish exits immediately instead. Closing during an episode interrupts playback and cleans up resources; incomplete episodes are not reported as completed results. Testing allows an explicit training seed for visual reproduction; held-out seed enforcement remains enabled for validation. The shared metrics retain the legacy validation_seeds JSON key for the actual test seeds as well.

A two-episode headless smoke run using the short training checkpoint completed, with both agents truncated and zero arrivals/collisions; this is not a learned-policy quality claim. Regression tests exercise checkpoint loading, deterministic repeated inference, unchanged checkpoint bytes, metrics and GUI flag parsing. Interactive GUI window playback was not performed in this tool session; use the first command above for manual verification. No environment files were changed.
