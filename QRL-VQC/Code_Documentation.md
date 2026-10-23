# Quantum Deep Reinforcement Learning (QDRL) with VQC Framework
## Complete Code Documentation

---

## 1. Scope of this document

This document explains the complete modular **single-agent Quantum Deep Reinforcement Learning (QDRL)** implementation based on a **Variational Quantum Circuit Deep Q-Network (VQC-DQN)** for the Gymnasium FrozenLake environment, together with the reusable `VQC_framework`.

The documentation is strictly about the **software design and code behavior**.

It explains:

- the project structure;
- the responsibility of every source file;
- all classes;
- all public and internal functions;
- command-line arguments;
- configuration parameters;
- state encoding;
- VQC construction;
- replay memory;
- epsilon-greedy action selection;
- online and target Q-networks;
- DQN optimization;
- device handling;
- model saving;
- CSV logging;
- circuit visualization;
- plotting utilities;
- tensor shapes;
- module interaction;
- execution sequence;
- extension points.

This document intentionally does **not** interpret, compare, or discuss experimental results.

---

# 2. Current project architecture

The current implementation is modular. The QDRL algorithm and the reusable VQC implementation are separated.

```text
Quantum_ML_Project/
│
├── driver_frozenlake_dqn_fixed.py
├── dqn_vqc.py
├── util.py
├── plot_frozenlake_results.py
├── config_quantum_circuit.yaml
├── requirements.txt
│
└── VQC_framework/
    ├── __init__.py
    ├── vqc.py
    ├── circuit.py
    └── visualization.py
```

During execution, the project may additionally create a `results/` directory:

```text
results/
├── frozenlake_vqc_dqn.csv
├── frozenlake_vqc_dqn.pt
├── frozenlake_vqc_circuit.png
└── plots/
    ├── reward.png
    ├── success_rate.png
    ├── loss.png
    └── epsilon.png
```

These are generated artifacts. They are not source modules.

---

# 3. Responsibility of each file

| File | Responsibility |
|---|---|
| `driver_frozenlake_dqn_fixed.py` | Main executable program. Creates FrozenLake, constructs online/target VQC-DQN networks, runs training, target synchronization, saving, plotting, circuit export, and evaluation. |
| `dqn_vqc.py` | Core DQN components: replay memory, hybrid VQC-DQN model, state-to-quantum conversion, epsilon-greedy action selection, and one minibatch optimization step. |
| `util.py` | Shared runtime utilities: command-line parser, training-device resolution, YAML loading, and random seeding. |
| `plot_frozenlake_results.py` | Reads the training CSV and creates separate figures for reward, success rate, loss, and epsilon schedule. |
| `config_quantum_circuit.yaml` | Contains only the quantum-circuit architecture parameters. |
| `requirements.txt` | Lists external Python packages required by the project. |
| `VQC_framework/__init__.py` | Public package interface. Exposes `VQC`. |
| `VQC_framework/vqc.py` | Main reusable `VQC` PyTorch module. Creates PennyLane `default.qubit`, trainable circuit weights, QNode, forward execution, summary, and visualization interface. |
| `VQC_framework/circuit.py` | Low-level quantum-circuit building blocks: configuration validation, encoding gates, trainable gates, entanglement, and measurement observables. |
| `VQC_framework/visualization.py` | Text and Matplotlib circuit drawing functions. |

The most important separation is:

```text
QDRL algorithm
    ↓
dqn_vqc.py
    ↓
VQC_framework
    ↓
PennyLane
    ↓
default.qubit
```

The QDRL algorithm does not directly create PennyLane gates. It uses the reusable `VQC` object.

---

# 4. High-level execution flow

The complete runtime sequence is:

```text
Command line
    │
    ▼
util.parse_args()
    │
    ├── episodes
    ├── learning rate
    ├── gamma
    ├── replay settings
    ├── epsilon settings
    ├── target update
    ├── device
    ├── output paths
    └── other runtime options
    │
    ▼
driver_frozenlake_dqn_fixed.py
    │
    ├── set random seed
    ├── resolve training device
    ├── load config_quantum_circuit.yaml
    └── create FrozenLake
            │
            ▼
      determine state/action counts
            │
            ▼
      create policy VQC
      create target VQC
            │
            ▼
        VQCDQN(policy_vqc)
        VQCDQN(target_vqc)
            │
            ▼
      copy policy parameters
      into target network
            │
            ▼
      create Adam optimizer
      create ReplayBuffer
            │
            ▼
         TRAINING LOOP
            │
            ├── reset environment
            ├── compute epsilon
            ├── select action
            ├── env.step(action)
            ├── store transition
            ├── sample replay batch
            ├── calculate DQN loss
            ├── backpropagate
            ├── optimizer step
            └── periodically copy policy → target
            │
            ▼
       save CSV/checkpoint
            │
            ├── generate plots
            ├── save VQC diagram
            └── greedy evaluation
```

---

# 5. Hybrid quantum-classical model architecture

The implemented Q-network is not purely quantum.

It is a hybrid model:

```text
FrozenLake state ID
       │
       ▼
binary conversion
       │
       ▼
0 / π quantum angles
       │
       ▼
VQC_framework.VQC
       │
       ▼
n_qubits expectation values
       │
       ▼
classical nn.Linear
       │
       ▼
Q-value for each action
```

For the default four-qubit FrozenLake configuration:

```text
state
  ↓
4 quantum input angles
  ↓
4 VQC expectation values
  ↓
Linear(4 → 4)
  ↓
4 action Q-values
```

The quantum component is used as a nonlinear trainable feature extractor.

The classical linear head converts quantum measurements to action values.

---

# 6. FrozenLake interface used by the code

The driver creates:

```python
gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=slippery,
    render_mode=render_mode,
)
```

The standard 4×4 environment provides:

```text
16 discrete states
4 discrete actions
```

The driver obtains these dynamically:

```python
n_states = env.observation_space.n
n_actions = env.action_space.n
```

Therefore `dqn_vqc.py` receives the environment dimensions instead of hard-coding them inside the model class.

---

# 7. State representation

FrozenLake returns an integer state.

For a 4×4 map:

```text
state ∈ {0, 1, ..., 15}
```

The QDRL model converts that integer into a binary representation.

For example:

```text
11 decimal
    ↓
1011 binary
```

For four qubits:

```text
1 0 1 1
```

is mapped to angles:

```text
π 0 π π
```

So the quantum input is:

\[
x_i = \pi b_i
\]

where:

\[
b_i \in \{0,1\}.
\]

The reusable VQC then applies the configured encoding rotation using these angles.

---

# 8. `config_quantum_circuit.yaml`

This file contains only quantum-circuit architecture.

The current configuration structure is:

```yaml
quantum_circuit:
  n_qubits: 4
  n_layers: 2
  encoding_gate: RY
  trainable_rotations: [RX, RY, RZ]
  entanglement: ring
  entangling_gate: CNOT
  measurement_pauli: Z
```

Training parameters such as learning rate, episodes, gamma, replay size, and epsilon are intentionally **not** stored here.

They are runtime arguments handled by `util.py`.

---

# 9. Configuration parameter: `n_qubits`

```yaml
n_qubits: 4
```

Defines the number of wires/qubits in the VQC.

The generic VQC encoder expects exactly one input angle per qubit.

For FrozenLake:

```text
16 states = 2^4
```

so four binary digits are sufficient.

The `VQCDQN` class validates whether the VQC has enough qubits:

```python
if self.n_states > 2 ** self.n_qubits:
    raise ValueError(...)
```

The minimum qubit requirement for `N` discrete states is:

\[
q_{\min}
=
\lceil \log_2 N \rceil.
\]

---

# 10. Configuration parameter: `n_layers`

```yaml
n_layers: 2
```

Defines how many trainable quantum layers are repeated.

The QNode structure is:

```text
Encoding
   ↓
Variational layer 0
   ↓
Entanglement 0
   ↓
Variational layer 1
   ↓
Entanglement 1
   ↓
Measurement
```

For general `L`:

```text
Encoding
   ↓
[Trainable rotations → Entanglement] × L
   ↓
Measurement
```

---

# 11. Configuration parameter: `encoding_gate`

```yaml
encoding_gate: RY
```

Allowed by the framework:

```text
RX
RY
RZ
```

The FrozenLake wrapper already converts state bits to angles `0` and `π`.

For the default setting, the VQC receives:

```text
[x0, x1, x2, x3]
```

and applies:

```text
RY(x0) on wire 0
RY(x1) on wire 1
RY(x2) on wire 2
RY(x3) on wire 3
```

---

# 12. Configuration parameter: `trainable_rotations`

```yaml
trainable_rotations: [RX, RY, RZ]
```

Every variational layer applies these gates to every qubit.

For one qubit:

```text
RX(theta_0)
RY(theta_1)
RZ(theta_2)
```

The ordering in the YAML list is preserved.

The weight tensor has shape:

```text
[n_layers, n_qubits, number_of_trainable_rotations]
```

For:

```text
2 layers
4 qubits
3 rotations
```

the tensor shape is:

```text
[2, 4, 3]
```

and contains:

\[
2\times4\times3=24
\]

trainable VQC weights.

This count refers only to the circuit weights inside `VQC`.

The hybrid `VQCDQN` also contains the trainable classical Q-head.

---

# 13. Configuration parameter: `entanglement`

Allowed:

```text
none
chain
ring
```

## `none`

No two-qubit gates are added.

## `chain`

For four qubits:

```text
0 → 1
1 → 2
2 → 3
```

## `ring`

The chain is created first and then the final qubit is connected back to the first:

```text
0 → 1
1 → 2
2 → 3
3 → 0
```

The ring closure is added only when:

```python
n_qubits > 2
```

---

# 14. Configuration parameter: `entangling_gate`

Allowed:

```text
CNOT
CZ
```

The default is:

```yaml
entangling_gate: CNOT
```

The topology and the gate type are independent settings.

For example:

```yaml
entanglement: ring
entangling_gate: CZ
```

would preserve ring connectivity but replace the CNOT operations by controlled-Z gates.

---

# 15. Configuration parameter: `measurement_pauli`

Allowed:

```text
X
Y
Z
```

Default:

```yaml
measurement_pauli: Z
```

The VQC returns one expectation value for every qubit:

\[
z_i = \langle Z_i\rangle.
\]

For four qubits, the VQC output has four values:

```text
[z0, z1, z2, z3]
```

These values are passed to the classical Q-head.

---

# Part I — Shared Utility Module

# 16. `util.py`

Purpose:

```text
Shared runtime utilities for FrozenLake VQC-DQN drivers
```

The file contains:

```python
DEFAULT_CONFIG
parse_args()
resolve_training_device()
load_quantum_circuit_config()
set_seed()
```

It removes CLI and environment-independent setup code from the main driver.

---

# 17. `DEFAULT_CONFIG`

```python
DEFAULT_CONFIG = "config_quantum_circuit.yaml"
```

Used as the default path for:

```bash
--config
```

This keeps the YAML filename centralized.

---

# 18. `parse_args()`

Signature:

```python
def parse_args():
```

Creates and returns an `argparse.Namespace`.

The function contains all primary QDRL runtime parameters.

---

# 19. CLI argument: `--episodes`

```python
--episodes
```

Type:

```text
int
```

Default:

```text
500
```

Controls the outer training loop:

```python
for episode in range(1, args.episodes + 1):
```

---

# 20. CLI argument: `--lr`

```python
--lr
```

Type:

```text
float
```

Default:

```text
1e-3
```

Passed to:

```python
torch.optim.Adam(
    policy_net.parameters(),
    lr=args.lr,
)
```

It therefore controls updates for both:

- the trainable VQC parameters inside `policy_net.vqc`;
- the classical `q_head` parameters.

---

# 21. CLI argument: `--gamma`

```python
--gamma
```

Default:

```text
0.99
```

Discount factor in the DQN target:

\[
y =
r+\gamma(1-d)\max_{a'}Q_{\text{target}}(s',a').
\]

---

# 22. CLI argument: `--batch-size`

Default:

```text
16
```

Number of transitions sampled from replay memory for one optimization call.

Used by:

```python
replay_buffer.sample(batch_size, device)
```

---

# 23. CLI argument: `--replay-size`

Default:

```text
1000
```

Passed to:

```python
ReplayBuffer(args.replay_size)
```

which internally uses:

```python
deque(maxlen=capacity)
```

When full, the oldest transitions are automatically removed.

---

# 24. CLI argument: `--min-replay-size`

Default:

```text
32
```

Training does not begin immediately.

The driver waits until:

```python
len(replay_buffer) >= args.min_replay_size
```

before calling `optimize_dqn()`.

This startup threshold is separate from the batch size.

---

# 25. CLI argument: `--target-update`

Default:

```text
20
```

Expressed in episodes.

The driver synchronizes:

```text
policy network → target network
```

when:

```python
episode % args.target_update == 0
```

using:

```python
target_net.load_state_dict(
    policy_net.state_dict()
)
```

---

# 26. CLI epsilon parameters

The utility exposes:

```text
--epsilon-start
--epsilon-end
--epsilon-decay
```

Defaults:

```text
epsilon_start = 1.0
epsilon_end   = 0.05
epsilon_decay = 150.0
```

The driver uses exponential decay:

\[
\epsilon(e)
=
\epsilon_{\text{end}}
+
(\epsilon_{\text{start}}-\epsilon_{\text{end}})
e^{-e/\tau}.
\]

where:

```text
e   = zero-based episode index
τ   = epsilon_decay
```

---

# 27. CLI argument: `--max-steps`

Default:

```text
100
```

Upper bound for environment interactions within one training or evaluation episode.

Training loop:

```python
for step in range(
    1,
    args.max_steps + 1,
):
```

---

# 28. CLI argument: `--eval-episodes`

Default:

```text
20
```

Number of greedy evaluation episodes executed after training.

---

# 29. CLI argument: `--device`

Choices:

```text
auto
cpu
cuda
mps
```

It controls requested training placement.

The final resolved choice is produced by:

```python
resolve_training_device()
```

---

# 30. CLI argument: `--slippery`

Boolean flag:

```bash
--slippery
```

If absent:

```text
is_slippery = False
```

If present:

```text
is_slippery = True
```

This flag is passed directly to `FrozenLake-v1`.

---

# 31. CLI argument: `--render`

Boolean flag.

It does not render during training.

It is passed to the separate evaluation environment after training.

---

# 32. CLI argument: `--config`

Default:

```text
config_quantum_circuit.yaml
```

Specifies which YAML circuit definition is loaded.

---

# 33. CLI argument: `--seed`

Default:

```text
42
```

Passed to:

```python
set_seed(args.seed)
```

and also used for:

- episode environment resets;
- the dedicated Python RNG used by epsilon-greedy action selection;
- evaluation reset seeds.

---

# 34. Output-related CLI arguments

The utility defines:

```text
--results-output
--model-output
--plots-dir
--plot-window
--circuit-output
```

Defaults:

```text
results/frozenlake_vqc_dqn.csv
results/frozenlake_vqc_dqn.pt
results/plots
100
results/frozenlake_vqc_circuit.png
```

These control file locations but do not alter the DQN mathematical update.

---

# 35. `resolve_training_device(requested)`

Signature:

```python
def resolve_training_device(requested):
```

Purpose:

> Choose a stable device for hybrid training with PennyLane `default.qubit`.

Behavior:

## CPU request

```text
cpu → cpu
```

## CUDA request

If:

```python
torch.cuda.is_available()
```

then:

```text
cuda → cuda
```

otherwise:

```text
cuda request → warning → cpu
```

## MPS request

The utility intentionally returns CPU:

```text
mps request → warning → cpu
```

The reason encoded by the code is that the chosen VQC simulator is PennyLane `default.qubit`, so the QDRL driver uses CPU for stable forward/backward execution on Apple systems.

## Auto request

If CUDA exists:

```text
auto → cuda
```

Otherwise:

```text
auto → cpu
```

If MPS exists, the function prints an informational message but still returns CPU.

This QDRL-level device policy is deliberately more conservative than the generic `_resolve_device()` method inside `VQC`.

---

# 36. `load_quantum_circuit_config(path)`

Reads YAML using:

```python
yaml.safe_load(file)
```

The file must contain:

```yaml
quantum_circuit:
```

as a top-level key.

The function returns only:

```python
config["quantum_circuit"]
```

If the YAML structure is invalid, it raises `KeyError`.

This means the `VQC` class receives a clean circuit-only dictionary rather than the full YAML document.

---

# 37. `set_seed(seed)`

Sets:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

and, when CUDA is available:

```python
torch.cuda.manual_seed_all(seed)
```

This centralizes reproducibility setup.

---

# Part II — QDRL/DQN Module

# 38. `dqn_vqc.py`

This module contains the reusable RL-level components:

```text
ReplayBuffer
VQCDQN
select_action()
optimize_dqn()
```

It intentionally does not contain:

- CLI parsing;
- Gym environment creation;
- the episode loop;
- result saving;
- evaluation loop;
- rendering;
- plotting.

Those remain in the driver or plotting module.

---

# 39. `ReplayBuffer`

Class:

```python
class ReplayBuffer:
```

Purpose:

> Maintain a bounded experience-replay memory for DQN transitions.

Each transition is stored as:

```text
(state, action, reward, next_state, done)
```

---

# 40. `ReplayBuffer.__init__(capacity)`

Creates:

```python
self.memory = deque(maxlen=capacity)
```

Properties of this design:

- fixed maximum capacity;
- appending is simple;
- old transitions are removed automatically after capacity is reached.

---

# 41. `ReplayBuffer.push(...)`

Signature:

```python
push(
    state,
    action,
    reward,
    next_state,
    done
)
```

Stores one environment transition:

\[
(s_t,a_t,r_t,s_{t+1},d_t).
\]

Unlike the earlier notebook implementation, this version stores the values directly as a Python tuple.

---

# 42. `ReplayBuffer.sample(batch_size, device)`

Uses:

```python
random.sample(
    self.memory,
    batch_size
)
```

to randomly choose transitions.

The method unpacks:

```python
states,
actions,
rewards,
next_states,
dones
```

and creates Torch tensors.

Tensor dtypes:

```text
states      → torch.long
actions     → torch.long
rewards     → torch.float32
next_states → torch.long
dones       → torch.float32
```

Tensor locations:

```text
device
```

passed by the training driver.

Actions are reshaped:

```python
actions.unsqueeze(1)
```

so the action tensor is compatible with:

```python
gather(1, actions)
```

later.

For batch size `B`:

```text
states      [B]
actions     [B, 1]
rewards     [B]
next_states [B]
dones       [B]
```

---

# 43. `ReplayBuffer.__len__()`

Returns:

```python
len(self.memory)
```

Used in two places:

1. driver startup threshold:
   ```python
   len(replay_buffer) >= args.min_replay_size
   ```

2. internal optimizer protection:
   ```python
   len(replay_buffer) < batch_size
   ```

---

# 44. `VQCDQN`

Class:

```python
class VQCDQN(nn.Module):
```

Purpose:

> Wrap the reusable VQC inside a Deep Q-Network interface.

The class contains:

```text
state encoder
    ↓
VQC
    ↓
classical linear Q-head
```

It is itself a `torch.nn.Module`.

Therefore:

```python
policy_net.parameters()
```

includes both:

- VQC parameters;
- Q-head parameters.

---

# 45. `VQCDQN.__init__(vqc, n_states, n_actions)`

Stores:

```python
self.vqc
self.n_states
self.n_actions
self.n_qubits
```

where:

```python
self.n_qubits = int(vqc.n_qubits)
```

---

# 46. State-capacity validation

The class checks:

```python
if self.n_states > 2 ** self.n_qubits:
    raise ValueError(...)
```

This verifies that the binary state representation fits the available qubits.

For `Q` qubits, the encoder can represent integer IDs:

\[
0,\ldots,2^Q-1.
\]

---

# 47. Classical Q-head

Created as:

```python
self.q_head = nn.Linear(
    self.n_qubits,
    self.n_actions,
)
```

For the standard FrozenLake setup:

```text
4 quantum measurements
        ↓
Linear(4,4)
        ↓
4 Q-values
```

The VQC itself is therefore not forced to directly represent one Q-value per measured wire.

Instead, measurements are treated as quantum features.

---

# 48. `state_to_quantum_angles(states)`

Purpose:

> Convert integer environment state IDs into binary-derived VQC angles.

The method accepts either:

```text
scalar state
```

or:

```text
batch of states
```

It converts them to:

```python
torch.long
```

on the same device as:

```python
self.q_head.weight
```

---

# 49. Single-state detection

The method uses:

```python
single_state = states.ndim == 0
```

A scalar is temporarily converted to a length-one batch:

```python
states = states.unsqueeze(0)
```

This allows the bit-conversion code to work for both scalar and batch inputs.

---

# 50. Bit-position tensor

The code builds:

```python
shifts = torch.arange(
    self.n_qubits - 1,
    -1,
    -1,
    ...
)
```

For four qubits:

```text
[3, 2, 1, 0]
```

This produces **most-significant-bit first** encoding.

---

# 51. Binary conversion

The core expression is:

```python
bits = (
    (states.unsqueeze(1) >> shifts) & 1
).to(torch.float32)
```

For state 11:

```text
11 >> [3,2,1,0]
      ↓
[1,0,1,1]
```

after masking with:

```text
& 1
```

---

# 52. Bit-to-angle mapping

The method computes:

```python
angles = torch.pi * bits
```

Therefore:

```text
0 → 0
1 → π
```

For state 11:

```text
[1,0,1,1]
    ↓
[π,0,π,π]
```

For a batch of `B` states and `Q` qubits:

```text
angles.shape = [B, Q]
```

For one state:

```text
angles.shape = [Q]
```

after `squeeze(0)`.

---

# 53. `VQCDQN.forward(states)`

Purpose:

> Return action Q-values for one state or a batch.

Sequence:

```text
states
  ↓
state_to_quantum_angles()
  ↓
quantum_inputs
  ↓
self.vqc(quantum_inputs)
  ↓
quantum_features
  ↓
self.q_head(quantum_features)
  ↓
Q-values
```

The VQC output is explicitly converted to:

```python
torch.float32
```

before entering the linear layer.

---

# 54. Forward tensor shapes

For one state:

```text
state                       scalar
quantum_inputs              [Q]
quantum_features            [Q]
q_values                    [A]
```

For a batch:

```text
states                      [B]
quantum_inputs              [B,Q]
quantum_features            [B,Q]
q_values                    [B,A]
```

where:

```text
B = batch size
Q = number of qubits
A = number of actions
```

---

# 55. `select_action(...)`

Signature:

```python
select_action(
    policy_net,
    state,
    epsilon,
    n_actions,
    rng,
)
```

Implements epsilon-greedy exploration.

---

# 56. Exploration branch

The condition:

```python
rng.random() < epsilon
```

selects a random action:

```python
rng.randrange(n_actions)
```

The driver provides a dedicated seeded RNG:

```python
rng = random.Random(args.seed)
```

---

# 57. Exploitation branch

The state is converted to:

```python
torch.long
```

and evaluated under:

```python
with torch.no_grad():
```

Then:

```python
q_values.argmax()
```

selects the action with maximum predicted Q-value.

No gradient graph is required for action selection.

---

# 58. `optimize_dqn(...)`

Signature:

```python
optimize_dqn(
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    batch_size,
    gamma,
    device,
)
```

Purpose:

> Perform one DQN minibatch gradient update.

Return:

```text
float loss
```

or:

```text
None
```

if replay memory contains fewer than `batch_size` entries.

---

# 59. Replay minibatch

The optimizer receives:

```text
states
actions
rewards
next_states
dones
```

from:

```python
replay_buffer.sample(...)
```

---

# 60. Current Q-value computation

The online network predicts Q-values for every action:

```python
policy_net(states)
```

Shape:

```text
[B,A]
```

The code extracts Q-values corresponding only to the actions actually stored in replay:

```python
current_q = policy_net(states).gather(
    1,
    actions,
).squeeze(1)
```

Result shape:

```text
[B]
```

Mathematically:

\[
Q_{\theta}(s_i,a_i).
\]

---

# 61. Target-network computation

Inside:

```python
with torch.no_grad():
```

the target network evaluates next states:

```python
target_net(next_states)
```

and takes:

```python
.max(dim=1).values
```

giving:

\[
\max_{a'}Q_{\theta^-}(s'_i,a').
\]

No gradient is stored for target calculation.

---

# 62. Bellman target

The code computes:

```python
target_q = rewards + (
    gamma
    * (1.0 - dones)
    * next_q
)
```

Equivalent to:

\[
y_i =
r_i
+
\gamma(1-d_i)
\max_{a'}Q_{\theta^-}(s'_i,a').
\]

For terminal transitions:

```text
done = 1
```

so:

```text
(1 - done) = 0
```

and the target becomes:

\[
y_i=r_i.
\]

---

# 63. Loss function

The implementation uses:

```python
F.smooth_l1_loss(
    current_q,
    target_q,
)
```

This is PyTorch's Smooth L1 / Huber-style regression loss.

It compares:

```text
online Q(s,a)
```

against:

```text
target Bellman value
```

---

# 64. Gradient update

The sequence is:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

This updates all trainable parameters registered in `policy_net`.

---

# 65. Gradient clipping

The code applies:

```python
torch.nn.utils.clip_grad_norm_(
    policy_net.parameters(),
    max_norm=10.0,
)
```

This limits the total gradient norm before the optimizer step.

---

# Part III — VQC Framework

# 66. `VQC_framework/__init__.py`

The file contains:

```python
from .vqc import VQC

__all__ = ["VQC"]
```

Purpose:

> Keep the public package API minimal.

User/QDRL code imports:

```python
from VQC_framework import VQC
```

rather than importing internal helpers directly.

The package internals can therefore evolve while preserving the public import.

---

# 67. `VQC_framework/vqc.py`

Contains:

```python
class VQC(nn.Module)
```

This is the main reusable quantum model.

The class is independent of FrozenLake and DQN.

It knows how to:

- read circuit configuration;
- validate it;
- allocate trainable parameters;
- construct PennyLane `default.qubit`;
- create a Torch-compatible QNode;
- execute one sample or a batch;
- count parameters;
- report architecture;
- draw the circuit;
- save/display a graphical circuit.

It does not know about:

- epsilon;
- replay memory;
- Bellman targets;
- DQN;
- FrozenLake;
- rewards;
- episodes.

---

# 68. VQC internal constants

```python
_DIFF_METHOD = "backprop"
_WEIGHT_INIT_SCALE = 0.1
_RANDOM_SEED = 42
```

These are framework implementation values, not YAML architecture parameters.

---

# 69. `_DIFF_METHOD`

Used when constructing:

```python
@qml.qnode(
    ...,
    interface="torch",
    diff_method=self._DIFF_METHOD,
)
```

Current value:

```text
backprop
```

---

# 70. `_WEIGHT_INIT_SCALE`

VQC weights are initialized as:

```python
0.1 * torch.randn(...)
```

The scale is therefore:

```text
0.1
```

---

# 71. `_RANDOM_SEED`

Before VQC weights are initialized:

```python
torch.manual_seed(42)
```

is called inside the VQC constructor.

---

# 72. `VQC.__init__(config, device="auto")`

The constructor:

1. copies the config;
2. extracts circuit values;
3. normalizes strings;
4. validates architecture;
5. resolves PyTorch device;
6. initializes trainable weights;
7. creates PennyLane `default.qubit`;
8. builds the QNode;
9. stores the QNode as `self.qnode`.

---

# 73. VQC configuration defaults

If keys are omitted, the class defaults to:

```text
n_qubits             4
n_layers             2
encoding_gate        RY
trainable_rotations  RX, RY, RZ
entanglement         ring
entangling_gate      CNOT
measurement_pauli    Z
```

The YAML normally provides these explicitly.

---

# 74. Configuration normalization

Gate names are normalized with:

```python
.upper()
```

Topology uses:

```python
.lower()
```

This allows differently cased configuration text to be interpreted consistently.

---

# 75. VQC trainable weight shape

Created as:

```python
weight_shape = (
    self.n_layers,
    self.n_qubits,
    len(self.trainable_rotations),
)
```

Then:

```python
initial_weights = (
    self._WEIGHT_INIT_SCALE
    * torch.randn(...)
)
```

and:

```python
self.weights = nn.Parameter(...)
```

Because the weights are `nn.Parameter`, they are automatically included in:

```python
policy_net.parameters()
```

through module nesting:

```text
VQCDQN
 ├── vqc
 │    └── weights
 └── q_head
      ├── weight
      └── bias
```

---

# 76. PennyLane device

The VQC always creates:

```python
qml.device(
    "default.qubit",
    wires=self.n_qubits,
)
```

No backend name is exposed through YAML.

The current framework intentionally uses only PennyLane's default simulator.

---

# 77. QNode construction

The QNode is defined inside the constructor:

```python
@qml.qnode(
    self.quantum_device,
    interface="torch",
    diff_method="backprop",
)
def circuit(inputs, weights):
    ...
```

It has two logical inputs:

```text
inputs  = classical encoded angles
weights = trainable VQC parameters
```

---

# 78. QNode stage 1: classical-to-quantum encoding

Calls:

```python
encode_inputs(
    inputs,
    self.n_qubits,
    self.encoding_gate,
)
```

This applies one configured rotation to every qubit.

---

# 79. QNode stage 2: variational blocks

For each layer:

```python
apply_variational_layer(...)
apply_entanglement(...)
```

So the structure is:

```text
single-qubit trainable rotations
             ↓
       entanglement
```

repeated `n_layers` times.

---

# 80. QNode stage 3: measurement

The QNode returns:

```python
[
    qml.expval(
        make_observable(
            self.measurement_pauli,
            wire
        )
    )
    for wire in range(self.n_qubits)
]
```

Thus the VQC's natural output dimension equals:

```text
n_qubits
```

---

# 81. `VQC._resolve_device(requested)`

This generic framework helper supports:

```text
auto
cpu
cuda
mps
```

Its behavior differs slightly from QDRL-level `util.resolve_training_device()`.

Generic VQC behavior:

```text
auto:
  CUDA if available
  else MPS if available
  else CPU

cuda:
  CUDA if available
  else warning + CPU

mps:
  MPS if available
  else warning + CPU

cpu:
  CPU
```

The FrozenLake QDRL driver resolves the training device first and normally passes the resolved string to the VQC.

---

# 82. `VQC.forward(inputs)`

Accepted shapes:

```text
[n_qubits]
```

or:

```text
[batch_size, n_qubits]
```

The method converts input with:

```python
torch.as_tensor(
    inputs,
    dtype=torch.float32,
    device=self.device,
)
```

---

# 83. Single-sample execution

If:

```python
x.ndim == 1
```

the code calls:

```python
self._forward_single(x)
```

---

# 84. Batch execution

If:

```python
x.ndim == 2
```

the code executes:

```python
torch.stack(
    [
        self._forward_single(row)
        for row in x
    ]
)
```

Therefore batching is currently implemented as repeated single-circuit simulation rather than a vectorized PennyLane batch transform.

---

# 85. Invalid input rank

The VQC raises:

```python
ValueError
```

unless input shape is:

```text
[Q]
```

or:

```text
[B,Q]
```

---

# 86. `VQC._forward_single(x)`

This is the internal one-sample quantum execution function.

It first validates:

```python
x.numel() == self.n_qubits
```

Because the current encoder uses one angle per qubit.

---

# 87. CPU staging inside VQC

The method explicitly performs:

```python
x_cpu = x.reshape(self.n_qubits).to("cpu")
weights_cpu = self.weights.to("cpu")
```

Then:

```python
result = self.qnode(
    x_cpu,
    weights_cpu,
)
```

After quantum execution:

```python
return result.to(self.device)
```

This creates the following flow:

```text
QDRL/PyTorch device
      │
      ▼
VQC input / VQC parameters
      │
      ▼
CPU staging
      │
      ▼
PennyLane default.qubit
      │
      ▼
measurement tensor
      │
      ▼
move back to VQC device
```

The `.to("cpu")` operation remains part of the Torch computational graph, allowing gradients to propagate back to the original trainable parameters.

---

# 88. QNode output normalization

PennyLane may return expectation values as a tuple/list.

The framework handles this with:

```python
if isinstance(result, (tuple, list)):
    result = torch.stack(list(result))
```

So public callers receive a Tensor.

---

# 89. `VQC.parameter_count()`

Returns:

```python
sum(
    parameter.numel()
    for parameter in self.parameters()
    if parameter.requires_grad
)
```

This is the count of trainable parameters registered directly within the `VQC` module.

It does not include the surrounding `VQCDQN.q_head` unless parameter counting is performed on the entire `VQCDQN`.

---

# 90. `VQC.summary()`

Returns a formatted text description containing:

- qubit count;
- layer count;
- encoding gate;
- trainable rotation names;
- entanglement topology;
- entangling gate;
- measurement basis;
- simulator name;
- resolved PyTorch device;
- trainable VQC parameter count.

It is used by the driver for architecture inspection.

---

# 91. `VQC.draw(inputs=None)`

Delegates to:

```python
draw_circuit(...)
```

and returns a text diagram of the actual QNode.

The circuit is not hand-coded separately for drawing; it is generated from the same `self.qnode` used for forward execution.

---

# 92. `VQC.visualize(...)`

Signature:

```python
visualize(
    inputs=None,
    save_path=None,
    show=True,
)
```

Delegates to:

```python
visualize_circuit(...)
```

and returns:

```text
(fig, ax)
```

from Matplotlib.

---

# 93. `VQC_framework/circuit.py`

This module contains only reusable gate-level logic.

Functions:

```text
validate_circuit_config()
apply_rotation()
encode_inputs()
apply_variational_layer()
apply_entanglement()
make_observable()
```

It has no RL-specific dependency.

---

# 94. `validate_circuit_config(...)`

Checks:

```text
n_qubits >= 1
n_layers >= 1
encoding gate ∈ {RX,RY,RZ}
trainable rotation list is nonempty
all trainable rotations ∈ {RX,RY,RZ}
entanglement ∈ {chain,ring,none}
entangling gate ∈ {CNOT,CZ}
measurement ∈ {X,Y,Z}
```

Invalid values produce `ValueError`.

Validation occurs before QNode construction.

---

# 95. `apply_rotation(name, angle, wire)`

Maps strings to PennyLane operations:

```text
RX → qml.RX
RY → qml.RY
RZ → qml.RZ
```

Example:

```python
apply_rotation(
    "RY",
    angle,
    wire=2,
)
```

creates:

```python
qml.RY(
    angle,
    wires=2,
)
```

---

# 96. `encode_inputs(inputs, n_qubits, encoding_gate)`

Loops through every wire:

```python
for wire in range(n_qubits):
```

and applies:

```python
apply_rotation(
    encoding_gate,
    inputs[wire],
    wire,
)
```

The function assumes:

```text
number of input values = number of qubits
```

---

# 97. `apply_variational_layer(...)`

For every qubit:

```python
for wire in range(n_qubits):
```

and every configured trainable gate:

```python
for gate_index, gate_name
```

it applies:

```python
layer_weights[
    wire,
    gate_index
]
```

as the angle.

One layer therefore consumes a parameter matrix:

```text
[n_qubits, number_of_rotations]
```

---

# 98. `apply_entanglement(...)`

Returns immediately when:

```text
entanglement == "none"
```

or:

```text
n_qubits == 1
```

Otherwise it defines a small internal helper that selects either:

```python
qml.CNOT(...)
```

or:

```python
qml.CZ(...)
```

The code first creates adjacent connections.

For ring topology and more than two qubits it additionally creates:

```text
last qubit → qubit 0
```

---

# 99. `make_observable(measurement_pauli, wire)`

Returns:

```text
X → qml.PauliX(wire)
Y → qml.PauliY(wire)
Z → qml.PauliZ(wire)
```

The VQC then wraps the returned observable with:

```python
qml.expval(...)
```

---

# 100. `VQC_framework/visualization.py`

Contains:

```text
prepare_drawing_input()
draw_circuit()
visualize_circuit()
```

The drawing code is kept separate from model execution and gate construction.

---

# 101. `prepare_drawing_input(inputs, n_qubits)`

If no input is supplied:

```python
torch.zeros(
    n_qubits,
    dtype=torch.float32,
)
```

is used.

Otherwise:

```python
torch.as_tensor(
    inputs,
    dtype=torch.float32
).reshape(-1)
```

converts the supplied values.

The function validates that exactly `n_qubits` values are present.

It returns a CPU tensor for drawing.

---

# 102. `draw_circuit(...)`

Uses:

```python
qml.draw(qnode)
```

with:

```python
weights.detach().to("cpu")
```

Detaching is appropriate because text visualization does not require gradients.

It returns a string.

---

# 103. `visualize_circuit(...)`

Uses:

```python
qml.draw_mpl(qnode)
```

and returns:

```python
fig, ax
```

The generated figure title is:

```text
Variational Quantum Circuit
```

If `save_path` is supplied:

```python
fig.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
)
```

is called.

If `show=True`:

```python
plt.show()
```

is executed.

---

# Part IV — Main Driver

# 104. `driver_frozenlake_dqn_fixed.py`

This is the only QDRL file containing:

```python
if __name__ == "__main__":
    main()
```

It is responsible for orchestration rather than low-level model implementation.

Its local functions are:

```text
epsilon_by_episode()
create_environment()
save_training_history()
save_model()
evaluate_policy()
save_circuit_figure()
main()
```

Runtime helpers are imported from `util.py`.

DQN helpers are imported from `dqn_vqc.py`.

VQC implementation is imported through the public package interface:

```python
from VQC_framework import VQC
```

---

# 105. Driver imports

Important imports:

```python
import csv
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
```

Project imports:

```python
from VQC_framework import VQC
```

```python
from dqn_vqc import (
    ReplayBuffer,
    VQCDQN,
    optimize_dqn,
    select_action,
)
```

```python
from plot_frozenlake_results import plot_results
```

```python
from util import (
    parse_args,
    resolve_training_device,
    load_quantum_circuit_config,
    set_seed,
)
```

---

# 106. `epsilon_by_episode(args, episode)`

Implements the exponential schedule:

```python
args.epsilon_end
+ (
    args.epsilon_start
    - args.epsilon_end
)
* math.exp(
    -episode
    / args.epsilon_decay
)
```

Input:

```text
episode = zero-based episode number
```

Output:

```text
current epsilon float
```

---

# 107. `create_environment(slippery, render=False)`

If:

```python
render=False
```

then:

```python
render_mode = None
```

If:

```python
render=True
```

then:

```python
render_mode = "human"
```

The function constructs `FrozenLake-v1`.

The driver deliberately creates the training environment with:

```python
render=False
```

and uses rendering only for optional evaluation.

---

# 108. `save_training_history(history, output_path)`

Creates parent directories if needed.

Writes CSV fields:

```text
episode
reward
steps
loss
epsilon
success
```

The function receives the already-created list of per-episode dictionaries.

It does not calculate these metrics itself.

---

# 109. CSV schema

Each history row has:

```python
{
    "episode": episode,
    "reward": total_reward,
    "steps": steps_taken,
    "loss": mean_loss,
    "epsilon": epsilon,
    "success": success,
}
```

Definitions in code:

- `episode` — one-based training episode number;
- `reward` — total environment reward accumulated in the episode;
- `steps` — number of environment steps executed;
- `loss` — arithmetic mean of minibatch loss values generated during that episode, or `0.0` if no update occurred;
- `epsilon` — epsilon used for the episode;
- `success` — integer computed by `total_reward > 0.0`.

These fields are later consumed by the plotting module.

---

# 110. `save_model(...)`

Saves a PyTorch checkpoint with:

```python
torch.save(...)
```

Checkpoint keys:

```text
model_state_dict
quantum_circuit_config
episodes
learning_rate
gamma
slippery
```

`model_state_dict` contains the trained nested module state.

Because `VQCDQN` owns both:

```text
vqc
q_head
```

the state dictionary includes both quantum-model parameters and classical-head parameters.

---

# 111. `evaluate_policy(...)`

Creates a separate FrozenLake environment.

Evaluation uses purely greedy selection:

```python
q_values.argmax()
```

No epsilon-greedy random branch is used.

The method:

1. resets environment with an evaluation-specific seed;
2. evaluates the policy under `torch.no_grad()`;
3. executes actions;
4. stops on termination/truncation or `max_steps`;
5. closes the environment;
6. returns two aggregate values.

The returned values are part of the evaluation interface; this documentation does not interpret them.

---

# 112. Evaluation seed offset

The reset seed is:

```python
seed + 10000 + episode
```

This separates evaluation reset sequences from training reset sequences.

---

# 113. `save_circuit_figure(...)`

Purpose:

> Visualize the actual VQC used inside the DQN.

Default example:

```text
example_state = 11
```

The function:

1. creates a state tensor;
2. calls:
   ```python
   policy_net.state_to_quantum_angles(state)
   ```
3. obtains the actual encoded VQC input;
4. calls:
   ```python
   policy_net.vqc.visualize(...)
   ```
5. saves the circuit figure.

Thus the saved circuit diagram is generated from the same reusable VQC instance used by the policy network.

---

# 114. `main()` — initialization phase

The main function begins:

```python
args = parse_args()
set_seed(args.seed)
```

Then resolves the training device:

```python
training_device = resolve_training_device(
    args.device
)
```

Then reads circuit architecture:

```python
circuit_config = (
    load_quantum_circuit_config(
        args.config
    )
)
```

---

# 115. `main()` — environment phase

Creates the training environment:

```python
env = create_environment(
    slippery=args.slippery,
    render=False,
)
```

and reads:

```python
n_states = env.observation_space.n
n_actions = env.action_space.n
```

This makes the DQN wrapper environment-aware without making the VQC framework environment-specific.

---

# 116. `main()` — two independent VQC instances

The driver creates:

```python
policy_vqc = VQC(...)
```

and:

```python
target_vqc = VQC(...)
```

The target VQC is a separate module instance.

This is important because the target network must not be the same object as the online policy network.

---

# 117. `main()` — two VQCDQN networks

The driver wraps the VQCs:

```python
policy_net = VQCDQN(
    policy_vqc,
    n_states=n_states,
    n_actions=n_actions,
)
```

and:

```python
target_net = VQCDQN(
    target_vqc,
    n_states=n_states,
    n_actions=n_actions,
)
```

Both are moved to:

```python
device
```

resolved from the policy VQC.

---

# 118. Initial target synchronization

Before training:

```python
target_net.load_state_dict(
    policy_net.state_dict()
)
```

This makes policy and target parameters initially identical.

Then:

```python
target_net.eval()
```

marks the target model as evaluation mode.

The current model does not contain dropout or batch normalization, but the target-network role is kept explicit.

---

# 119. Optimizer construction

The driver creates:

```python
optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=args.lr,
)
```

Since `policy_net` nests the VQC and Q-head, the optimizer sees both.

---

# 120. Replay buffer construction

```python
replay_buffer = ReplayBuffer(
    args.replay_size
)
```

The buffer exists for the entire training run.

It is not reset at each episode.

---

# 121. Dedicated action RNG

The driver creates:

```python
rng = random.Random(
    args.seed
)
```

This object is passed to:

```python
select_action(...)
```

for epsilon-greedy randomness.

---

# 122. Training history container

The driver initializes:

```python
history = []
```

One dictionary is appended after every episode.

---

# 123. Episode loop

```python
for episode in range(
    1,
    args.episodes + 1,
):
```

At the beginning of every episode:

```python
state, _ = env.reset(
    seed=args.seed + episode
)
```

The per-episode epsilon is calculated once:

```python
epsilon = epsilon_by_episode(
    args,
    episode - 1,
)
```

So epsilon remains constant throughout that episode.

---

# 124. Per-episode tracking variables

Initialized as:

```text
total_reward = 0.0
episode_losses = []
steps_taken = 0
```

These are reset every episode.

---

# 125. Step loop

The step loop runs:

```python
for step in range(
    1,
    args.max_steps + 1,
):
```

At every step the sequence is:

```text
select action
   ↓
environment step
   ↓
compute done
   ↓
store transition
   ↓
advance state
   ↓
accumulate reward
   ↓
optional optimization
   ↓
break if terminal
```

---

# 126. Action selection in training

The driver calls:

```python
action = select_action(
    policy_net,
    state,
    epsilon,
    n_actions,
    rng,
)
```

This isolates epsilon-greedy policy logic from the episode loop.

---

# 127. Gymnasium step return

The driver unpacks:

```text
next_state
reward
terminated
truncated
info
```

The `info` dictionary is ignored.

The driver defines:

```python
done = (
    terminated
    or truncated
)
```

This combined flag is stored in replay memory.

---

# 128. Replay insertion order

The transition is stored before the current state is overwritten:

```python
replay_buffer.push(
    state,
    action,
    reward,
    next_state,
    done,
)
```

Then:

```python
state = next_state
```

---

# 129. Learning-start condition

The driver calls the optimizer only after:

```python
len(replay_buffer)
>= args.min_replay_size
```

This is explicitly commented as delaying learning until replay contains more diversity.

---

# 130. Per-step optimization

Once enabled, the driver calls:

```python
loss = optimize_dqn(
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    args.batch_size,
    args.gamma,
    device,
)
```

One minibatch update can therefore occur after each environment interaction.

Returned scalar losses are appended to:

```python
episode_losses
```

---

# 131. Episode termination

If:

```python
done
```

the step loop breaks immediately.

Otherwise it continues until:

```text
max_steps
```

is reached.

---

# 132. Periodic target synchronization

After each episode:

```python
if (
    episode
    % args.target_update
    == 0
):
    target_net.load_state_dict(
        policy_net.state_dict()
    )
```

Thus target synchronization is episode-based, not step-based, in this modular FrozenLake implementation.

---

# 133. Mean episode loss

If optimization occurred:

```python
mean_loss = float(
    np.mean(episode_losses)
)
```

otherwise:

```text
0.0
```

is stored.

---

# 134. Success flag

The code defines:

```python
success = int(
    total_reward > 0.0
)
```

This is a software-level binary field used for logging and plotting.

---

# 135. Progress printing

The driver prints status when:

```text
episode == 1
episode % 25 == 0
episode == final episode
```

It computes a recent window:

```python
recent = history[-100:]
```

and calculates recent aggregates for console monitoring.

These console values do not affect training updates.

---

# 136. Post-training sequence

After the training environment is closed, the driver executes:

```text
save_training_history()
save_model()
plot_results()
save_circuit_figure()
evaluate_policy()
```

This keeps model optimization separate from file export and evaluation.

---

# Part V — Plotting Module

# 137. `plot_frozenlake_results.py`

The plotting module is separate from model training.

Its purpose is to read the saved CSV and generate figures.

It does not train the policy.

Core functions in the modular plotting implementation are:

```text
parse_args()
moving_average()
load_results()
save_reward_plot()
save_success_plot()
save_loss_plot()
save_epsilon_plot()
plot_results()
main()
```

The current driver imports:

```python
from plot_frozenlake_results import plot_results
```

so plotting can be invoked programmatically after training.

The script can also be used independently through its own `main()`.

---

# 138. Plotting `parse_args()`

Defines:

```text
--input
--output-dir
--window
```

with defaults corresponding to the standard results CSV and plots folder.

---

# 139. `moving_average(values, window)`

Converts input to:

```python
np.asarray(
    values,
    dtype=float,
)
```

Then computes a moving mean manually.

For early indices where fewer than `window` samples exist, it uses the shorter prefix.

Thus no leading values are discarded.

---

# 140. `load_results(path)`

Reads CSV with:

```python
csv.DictReader
```

and converts fields back to numeric types.

The expected CSV schema is exactly:

```text
episode
reward
steps
loss
epsilon
success
```

If no rows exist, the function raises:

```python
ValueError(
    "The result CSV is empty."
)
```

---

# 141. `save_reward_plot(...)`

Uses:

```text
episode
reward
moving average of reward
```

and writes:

```text
reward.png
```

The raw reward trace and moving average are represented separately in the same figure.

---

# 142. `save_success_plot(...)`

Computes:

```python
100.0 * moving_average(
    successes,
    window,
)
```

and writes:

```text
success_rate.png
```

The y-axis is constrained to:

```text
0 ... 100
```

because the plotted quantity is percentage-form binary moving average.

---

# 143. `save_loss_plot(...)`

Plots episode-level mean DQN loss from the CSV and writes:

```text
loss.png
```

---

# 144. `save_epsilon_plot(...)`

Plots the stored per-episode epsilon values and writes:

```text
epsilon.png
```

---

# 145. `plot_results(...)`

Programmatic entry point used by the QDRL driver.

Typical signature:

```python
plot_results(
    input_csv,
    output_dir="results/plots",
    window=100,
)
```

Sequence:

1. read CSV;
2. create output directory;
3. extract arrays;
4. save reward figure;
5. save success figure;
6. save loss figure;
7. save epsilon figure;
8. return output directory.

---

# Part VI — Requirements and Dependencies

# 146. `requirements.txt`

The project relies on the following library categories:

```text
PennyLane
PyTorch
Gymnasium
PyYAML
NumPy
Matplotlib
```

The exact version pins in the project's `requirements.txt` should be treated as the installation source of truth.

Roles:

| Package | Role |
|---|---|
| `pennylane` | Quantum gates, QNode, `default.qubit`, expectation values, quantum differentiation, circuit drawing |
| `torch` | `nn.Module`, trainable parameters, optimizer, tensors, gradients, linear Q-head |
| `gymnasium` | FrozenLake environment |
| `PyYAML` | Quantum-circuit YAML configuration parsing |
| `numpy` | Episode statistics and plotting-array preparation |
| `matplotlib` | Training figures and graphical quantum-circuit rendering |

---

# Part VII — Complete Model Data Flow

# 147. One environment interaction

For one state:

```text
FrozenLake state ID
      │
      │ e.g. 11
      ▼
VQCDQN.state_to_quantum_angles
      │
      │ 11 → 1011
      │ 1011 → [π,0,π,π]
      ▼
VQC.forward
      │
      ▼
PennyLane QNode
      │
      ├── encoding rotation per qubit
      ├── trainable layer 1
      ├── entanglement
      ├── trainable layer 2
      ├── entanglement
      └── Pauli expectation values
      │
      ▼
quantum feature vector
      │
      │ shape [4]
      ▼
nn.Linear(4,4)
      │
      ▼
Q(s,0), Q(s,1), Q(s,2), Q(s,3)
      │
      ▼
epsilon-greedy action
      │
      ▼
FrozenLake env.step(action)
```

---

# 148. One replay optimization

```text
ReplayBuffer
    │
    ├── states       [B]
    ├── actions      [B,1]
    ├── rewards      [B]
    ├── next_states  [B]
    └── dones        [B]
    │
    ▼
policy_net(states)
    │
    ▼
Q_online(s, all actions)
    │
    ▼
gather stored action
    │
    ▼
current_q [B]

next_states
    │
    ▼
target_net(next_states)
    │
    ▼
max over actions
    │
    ▼
next_q [B]
    │
    ▼
Bellman target
    │
    ▼
target_q [B]

current_q, target_q
    │
    ▼
Smooth L1 loss
    │
    ▼
backpropagation
    │
    ▼
gradient clipping
    │
    ▼
Adam step
```

---

# 149. Module nesting and gradient path

The trainable policy model has this structure:

```text
VQCDQN
│
├── VQC
│   └── weights
│
└── Linear Q-head
    ├── weight
    └── bias
```

During:

```python
loss.backward()
```

the computational path is:

```text
loss
 ↓
Q-head
 ↓
quantum features
 ↓
PennyLane Torch QNode
 ↓
VQC trainable weights
```

Therefore both quantum and classical parts participate in the optimizer update.

---

# 150. Target network structure

The target network is another complete:

```text
VQCDQN
```

containing its own:

```text
VQC
Linear Q-head
```

It is not optimized directly.

Its parameters are refreshed through:

```python
target_net.load_state_dict(
    policy_net.state_dict()
)
```

at the configured episode interval.

---

# 151. Distinction between circuit config and RL config

The project intentionally separates them.

## Circuit architecture

Stored in:

```text
config_quantum_circuit.yaml
```

Examples:

```text
number of qubits
number of VQC layers
encoding gate
trainable gates
entanglement topology
entangling gate
measurement Pauli
```

## QDRL/training configuration

Provided by CLI:

```text
episodes
learning rate
gamma
batch size
replay capacity
minimum replay size
target update period
epsilon schedule
max steps
evaluation episodes
device
seed
output paths
```

This makes the VQC reusable in other algorithms.

---

# 152. Distinction between `util.resolve_training_device()` and `VQC._resolve_device()`

There are two device layers.

## QDRL-level decision

```python
resolve_training_device()
```

chooses a conservative training device appropriate to the current `default.qubit` QDRL workflow.

## Generic VQC-level decision

```python
VQC._resolve_device()
```

supports CPU, CUDA, and MPS for the reusable package interface.

In the FrozenLake driver, the first function resolves the requested device and passes its decision into `VQC`.

---

# 153. Visualization data flow

The driver uses an actual environment-state example:

```text
integer state
   ↓
state_to_quantum_angles()
   ↓
actual VQC input angles
   ↓
VQC.visualize()
   ↓
visualization.py
   ↓
qml.draw_mpl(self.qnode)
```

The displayed/saved circuit therefore matches the QNode architecture used in training.

---

# 154. Error and validation behavior

The project explicitly checks several invalid situations.

## Circuit configuration

Raises `ValueError` for invalid:

```text
qubit count
layer count
encoding gate
trainable gate list
entanglement mode
entangling gate
measurement Pauli
```

## YAML structure

Raises `KeyError` when:

```text
quantum_circuit:
```

is missing.

## State capacity

Raises `ValueError` when:

```text
n_states > 2^n_qubits
```

## VQC input dimension

Raises `ValueError` when the VQC does not receive exactly one input value per qubit.

## VQC input rank

Only:

```text
[Q]
[B,Q]
```

are accepted.

## Plot CSV

Raises `ValueError` when the file contains no result rows.

## Invalid device name

Produces `ValueError`.

Unavailable CUDA or MPS requests are handled by controlled fallback/warning logic.

---

# 155. Source-file dependency graph

```text
driver_frozenlake_dqn_fixed.py
│
├── util.py
│   ├── argparse
│   ├── random
│   ├── numpy
│   ├── torch
│   └── yaml
│
├── dqn_vqc.py
│   ├── collections.deque
│   ├── random
│   └── torch
│
├── plot_frozenlake_results.py
│   ├── csv
│   ├── pathlib
│   ├── numpy
│   └── matplotlib
│
├── gymnasium
│
└── VQC_framework
    │
    ├── __init__.py
    │    └── VQC
    │
    ├── vqc.py
    │    ├── torch
    │    ├── pennylane
    │    ├── circuit.py
    │    └── visualization.py
    │
    ├── circuit.py
    │    ├── torch
    │    └── pennylane
    │
    └── visualization.py
         ├── torch
         ├── pennylane
         └── matplotlib
```

---

# 156. Public interfaces

Normal application code should primarily use:

```python
from VQC_framework import VQC
```

QDRL driver additionally uses:

```python
from dqn_vqc import (
    ReplayBuffer,
    VQCDQN,
    select_action,
    optimize_dqn,
)
```

and:

```python
from util import (
    parse_args,
    resolve_training_device,
    load_quantum_circuit_config,
    set_seed,
)
```

The low-level functions in:

```text
VQC_framework/circuit.py
VQC_framework/visualization.py
```

are framework internals and normally do not need to be imported by the QDRL driver.

---

# 157. Generated checkpoint structure

The checkpoint saved by the driver is a dictionary.

Conceptually:

```python
{
    "model_state_dict": ...,
    "quantum_circuit_config": ...,
    "episodes": ...,
    "learning_rate": ...,
    "gamma": ...,
    "slippery": ...,
}
```

This combines:

- trainable model state;
- quantum architecture metadata;
- selected training metadata.

The code currently documents saving; a separate checkpoint-loading pipeline is not implemented in the shown driver.

---

# 158. Generated file interfaces

The code may generate:

```text
CSV training history
PyTorch checkpoint
PNG VQC circuit diagram
PNG training figures
```

These are program outputs.

Their presence is part of the software interface, but this documentation does not evaluate their numerical content.

---

# 159. Current architectural boundaries

The design currently keeps responsibilities clean:

## Environment belongs to driver

```text
FrozenLake creation
reset
step
render
```

## RL belongs to `dqn_vqc.py` + driver

```text
replay
epsilon-greedy
Bellman target
online network
target network
loss
optimizer
episode loop
```

## Quantum model belongs to `VQC_framework`

```text
gates
weights
entanglement
measurement
QNode
simulator
circuit visualization
```

## Quantum architecture belongs to YAML

```text
qubits
layers
gates
entanglement
measurement
```

## Runtime hyperparameters belong to CLI

```text
episodes
learning rate
gamma
epsilon
replay
target updates
device
```

---

# 160. Current code assumptions

The implementation assumes:

1. the environment state can be represented as a non-negative integer ID;
2. the integer state fits into the available number of qubits;
3. one binary digit is mapped to one qubit input angle;
4. the VQC returns one expectation value per qubit;
5. a classical linear head converts VQC outputs into action Q-values;
6. the action space is discrete;
7. DQN uses replay memory and a hard-updated target network;
8. PennyLane `default.qubit` is the only quantum simulator used by the framework;
9. VQC input/parameter simulation is staged on CPU internally;
10. current batch quantum execution is implemented as repeated single-sample QNode calls.

---

# 161. Current extensibility points

The project structure allows later modification without rewriting all modules.

Examples of future code changes that can be isolated:

## New environment

Primarily modify:

```text
driver
state preprocessing
```

while preserving the VQC framework.

## Different state encoding

Modify:

```text
VQCDQN.state_to_quantum_angles()
```

or the VQC encoder.

## Different circuit architecture

Change:

```text
config_quantum_circuit.yaml
```

when supported by `circuit.py`.

## Different quantum layer design

Modify:

```text
VQC_framework/circuit.py
```

while keeping the QDRL API stable.

## Different RL algorithm

Reuse:

```text
VQC_framework
```

and replace:

```text
dqn_vqc.py
driver logic
```

as required.

---

# 162. Minimal execution command

The driver can be launched with its defaults.

Conceptually:

```bash
python driver_frozenlake_dqn_fixed.py
```

Runtime values can be overridden through CLI.

For example:

```bash
python driver_frozenlake_dqn_fixed.py \
    --episodes 500 \
    --lr 0.001 \
    --gamma 0.99 \
    --device cpu
```

The YAML circuit architecture is loaded separately.

---

# 163. Complete training call relationship

A useful code-reading summary is:

```text
main()
│
├── parse_args()
├── set_seed()
├── resolve_training_device()
├── load_quantum_circuit_config()
├── create_environment()
│
├── VQC()                    policy VQC
├── VQC()                    target VQC
│
├── VQCDQN()                 policy Q-network
├── VQCDQN()                 target Q-network
│
├── ReplayBuffer()
├── Adam()
│
└── episode loop
    │
    ├── env.reset()
    ├── epsilon_by_episode()
    │
    └── step loop
        │
        ├── select_action()
        ├── env.step()
        ├── ReplayBuffer.push()
        │
        └── optimize_dqn()
             │
             ├── ReplayBuffer.sample()
             ├── policy_net(states)
             │    ├── state_to_quantum_angles()
             │    ├── VQC.forward()
             │    │    └── PennyLane QNode
             │    └── q_head()
             │
             ├── target_net(next_states)
             ├── Bellman target
             ├── Smooth L1 loss
             ├── backward()
             ├── gradient clipping
             └── optimizer.step()
```

---

# 164. VQC call relationship

Inside either online or target Q-network:

```text
VQCDQN.forward(states)
│
├── state_to_quantum_angles(states)
│
├── VQC.forward(angles)
│   │
│   ├── _forward_single()
│   │
│   └── QNode
│       │
│       ├── encode_inputs()
│       │    └── apply_rotation()
│       │
│       ├── apply_variational_layer()
│       │    └── apply_rotation()
│       │
│       ├── apply_entanglement()
│       │
│       └── make_observable()
│
└── q_head(quantum_features)
```

This is the central integration point between QDRL and the generic VQC framework.

---

# 165. Summary of classes and functions

## `driver_frozenlake_dqn_fixed.py`

```text
epsilon_by_episode
create_environment
save_training_history
save_model
evaluate_policy
save_circuit_figure
main
```

## `util.py`

```text
parse_args
resolve_training_device
load_quantum_circuit_config
set_seed
```

## `dqn_vqc.py`

```text
ReplayBuffer
  __init__
  push
  sample
  __len__

VQCDQN
  __init__
  state_to_quantum_angles
  forward

select_action
optimize_dqn
```

## `plot_frozenlake_results.py`

```text
parse_args
moving_average
load_results
save_reward_plot
save_success_plot
save_loss_plot
save_epsilon_plot
plot_results
main
```

## `VQC_framework/vqc.py`

```text
VQC
  __init__
  _resolve_device
  forward
  _forward_single
  parameter_count
  summary
  draw
  visualize
```

## `VQC_framework/circuit.py`

```text
validate_circuit_config
apply_rotation
encode_inputs
apply_variational_layer
apply_entanglement
make_observable
```

## `VQC_framework/visualization.py`

```text
prepare_drawing_input
draw_circuit
visualize_circuit
```

---

# 166. Final software view

The current project is best understood as four layers:

```text
┌──────────────────────────────────────┐
│ User / Experiment Layer              │
│ driver_frozenlake_dqn_fixed.py       │
│ util.py                              │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ QDRL Layer                           │
│ dqn_vqc.py                           │
│ ReplayBuffer + VQCDQN + DQN update   │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ Reusable Quantum Model Layer         │
│ VQC_framework                        │
│ VQC + circuit + visualization        │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│ Quantum Execution Layer              │
│ PennyLane default.qubit              │
└──────────────────────────────────────┘
```

The YAML configuration defines the quantum architecture, the CLI defines QDRL runtime/training settings, the DQN module defines the reinforcement-learning update, and the VQC package provides the reusable differentiable quantum circuit.

This separation is the central design principle of the current implementation.
