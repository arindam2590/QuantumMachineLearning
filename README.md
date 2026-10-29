# QuantumMachineLearning

## 1. Scope of this document

This document explains the complete modular **single-agent Quantum Deep Reinforcement Learning (QDRL)** implementation based on a **Variational Quantum Circuit Deep Q-Network (VQC-DQN)** for the Gymnasium FrozenLake environment, together with the reusable `VQC_framework`.

## 2. Current project architecture

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
