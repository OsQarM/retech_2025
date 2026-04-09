# Task 4 — MPS/MPO-Based Hamiltonian Learning from Quantum States

## Overview

This task implements a **neural network approach** to learning a quantum Hamiltonian whose time-evolved state reproduces a target bitstring probability distribution. Unlike Task 3 (which uses time-series trajectories), here the target is a **single snapshot** of measurement outcomes.

Two main workflows are provided:

1. **MPS/MPO learning** (`mps_learning.ipynb`): the target distribution comes from a synthetic quantum state with Matrix Product State (MPS) structure. A neural network — optionally with Matrix Product Operator (MPO) layers for compression — maps per-qubit measurement probabilities to Hamiltonian parameters.

2. **MLP-to-Hamiltonian** (`MLP_to_H.ipynb`): the target distribution comes from the output activations of an external pre-trained MLP (e.g. an image classifier). The learned Hamiltonian encodes the MLP's computation as a quantum circuit.

Both workflows use PyTorch and backpropagate through an RK4 time-evolution step to train the network end-to-end.

---

## Directory Structure

```
task_4/
├── config/
│   ├── MPS_data_generation.yaml                    # Config for MPS/state data generation
│   ├── MPS_learning_configuration_MPO_optimized.yaml  # Config for MPS/MPO training
│   └── MLP_to_H_configuration.yaml                 # Config for MLP-to-Hamiltonian training
├── data/                    # Generated datasets
├── notebooks/
│   ├── generate_state_data.py    # Generates simple quantum states (zeros/plus/GHZ)
│   ├── generate_MPS_data.py      # Generates MPS-structured quantum states
│   ├── mps_learning.ipynb        # Main notebook: MPS/MPO Hamiltonian learning
│   └── MLP_to_H.ipynb            # Main notebook: MLP output → Hamiltonian
├── plots/                   # Output figures
├── results/                 # Saved Hamiltonians and learned parameters
├── saved_models/            # Saved PyTorch model checkpoints
└── src/
    ├── physics.py           # Quantum operators, dynamics, and RK4 evolution (PyTorch)
    ├── NeuralNetworks.py    # MPS/MPO network architectures (tensorkrowch-based)
    ├── model.py             # Legacy simple NN with optional MPO layer
    ├── utils.py             # Gradient/NaN checks and parameter counting
    ├── mpo_tenpy.py         # MPO utilities via TeNPy
    ├── run_inference.py     # Inference helpers for saved models
    └── plots.py             # Plotting utilities
```

---

## Workflow A — MPS/MPO Learning

### Step 1 — Generate data

Run one of the two data generation scripts to produce a target bitstring distribution:

```bash
# Option 1: simple quantum state (zeros / plus / GHZ)
python notebooks/generate_state_data.py

# Option 2: random MPS with controlled bond dimension
python notebooks/generate_MPS_data.py
```

Both read `config/MPS_data_generation.yaml` and save bitstring counts to `data/`.

### Step 2 — Train the model

Open and run `notebooks/mps_learning.ipynb`. The notebook:
1. Loads the bitstring data and converts it to per-qubit marginal probabilities (shape `L × 2`).
2. Builds a neural network (MPS or MPO type) controlled by `MPS_learning_configuration_MPO_optimized.yaml`.
3. Trains via gradient descent: the network outputs Hamiltonian parameters → RK4 evolves an initial state → the loss compares the evolved distribution to the target.
4. Optionally compresses the trained dense network into an MPO representation.
5. Saves the learned Hamiltonian to `results/` and plots the probability distribution comparison.

---

## Workflow B — MLP to Hamiltonian

Open and run `notebooks/MLP_to_H.ipynb`. The notebook:
1. Loads pre-trained MLP weights from the path specified in `MLP_to_H_configuration.yaml`.
2. Depending on `learning_mode`:
   - **`output_only`**: the target is the MLP's output probability vector (e.g. softmax scores for 10 classes). `L` is set automatically from `MLP_output_size`.
   - **`output_from_input`**: the target is the MLP's output, but the quantum initial state is encoded from the MLP's *input* (e.g. a flattened image). `L` is derived from `MLP_input_size`.
3. Trains a Hamiltonian to reproduce the target distribution via time evolution.
4. Saves the result to `results/`.

---

## Configuration Files

### `config/MPS_data_generation.yaml`

Controls synthetic data generation.

| Parameter | Description |
|-----------|-------------|
| `L` | **Number of qubits.** |
| `bond_dimension` | Bond dimension of the generated MPS (controls entanglement complexity). |
| `min_val` / `max_val` | Range for random MPS tensor entries. |
| `N_shots` | Number of bitstring samples to draw. |
| `state_topology` | **`"linear"`** (chain MPS) or **`"star"`** (central tensor with `L-1` legs). |
| `state_type` | **`"zeros"`**, **`"plus"`**, or **`"ghz"`** (used by `generate_state_data.py` only). |
| `dynamics_type` | `"schrodinger"` or `"lindblad"` (for noisy state generation). |
| `noise_model` | `"global"` or `"local"`. |
| `T1_global` / `T2_global` | Relaxation / dephasing times (global noise model). |
| `T1_list` / `T2_list` | Per-qubit times (local noise model, length `L`). |
| `seed_data` | Random seed. |

### `config/MPS_learning_configuration_MPO_optimized.yaml`

Controls the training in `mps_learning.ipynb`.

| Parameter | Description |
|-----------|-------------|
| `L` | **Number of qubits** (must match the dataset). |
| `bond_dimension_data` | Bond dimension of the target data (informational). |
| `bond_dimension_learning` | **Bond dimension of the MPS/MPO network** used for learning. |
| `t_max` | Total time for the Hamiltonian evolution used in the loss. |
| `dt` | RK4 integration time step. |
| `initial_state_kind` | Starting state for evolution: `"all_zeros"` or `"all_plus"`. |
| `data_kind` | Source of target data: `"mps"`, `"zeros"`, `"plus"`, or `"ghz"`. |
| `NN_TYPE` | **`"mps"`**: uses `tensorkrowch` MPSLayer. **`"mpo"`**: uses TT-SVD MPO decomposition. |
| `MPO_SIZE` | Size of the MPO layer (must be a **perfect cube**: 27, 64, 125, …). |
| `MAX_MPO_CHI` | **Maximum bond dimension** for TT-SVD compression. Higher = more expressive. |
| `SIMPLE_MPO` | `True` for a minimal architecture (useful when convergence is poor). |
| `TRAINABLE_MPO` | `True` to allow MPO core tensors to be trained via gradient descent. |
| `MPO_ON` | `True` to actually apply the MPO decomposition during the forward pass. |
| `N_epochs` | Number of training epochs. |
| `learning_rate` | Adam optimizer learning rate. |
| `lambda_reg` | **L2 regularisation** on network parameters. |
| `GRAD_CLIP_NORM` | Gradient clipping norm (set to `0` to disable). |
| `USE_LR_SCHEDULER` | Enable learning-rate decay on plateau. |
| `LR_PATIENCE` / `LR_FACTOR` | Patience (epochs) and decay factor for the LR scheduler. |
| `USE_EARLY_STOPPING` | Stop training if loss does not improve. |
| `EARLY_STOP_PATIENCE` | Number of epochs to wait before early stopping triggers. |

> **MPO convergence tips** (from the config comments): if MPO training is unstable, try increasing `MAX_MPO_CHI`, enabling `SIMPLE_MPO`, lowering `learning_rate`, or starting with `NN_TYPE: "mps"` to verify the physics pipeline first.

### `config/MLP_to_H_configuration.yaml`

Controls the training in `MLP_to_H.ipynb`.

| Parameter | Description |
|-----------|-------------|
| `learning_mode` | **`"output_only"`**: learn from MLP output vector only. **`"output_from_input"`**: encode MLP input into the quantum initial state. |
| `MLP_input_folder` | Path to the folder containing MLP weight files. |
| `MLP_input_file` | Path to a `.npy` file with a sample MLP input (e.g. a flattened image). |
| `MLP_input_size` | Dimension of the MLP input (e.g. `784` for 28×28 images). |
| `MLP_output_size` | Dimension of the MLP output (e.g. `10` for 10-class classification). Determines `L` in `output_only` mode. |
| `bond_dimension_learning` | Bond dimension for the quantum state encoding of MLP inputs. |
| `t_max` | Total evolution time for the Hamiltonian. |
| `dt` | RK4 integration time step. |
| `N_epochs` | Training epochs. |
| `learning_rate` | Adam optimizer learning rate. |
| `MPO_ON` | Enable MPO decomposition during inference. |
| `MAX_MPO_CHI` | MPO bond dimension cap. |
| `lambda_reg` | L2 regularisation weight. |
| `GRAD_CLIP_NORM` | Gradient clipping norm. |

---

## Key Source Files

| File | Role |
|------|------|
| `src/physics.py` | Pauli operators, Hamiltonian construction, RK4 time evolution (PyTorch autograd-compatible). |
| `src/NeuralNetworks.py` | All network architectures: `MPS_MLP` (tensorkrowch), `NeuralNetworkMPO` / `NeuralNetworkTrainableMPO` (TT-SVD MPO), `SimpleMPONetwork`. |
| `src/model.py` | Legacy simple dense network with optional MPO middle layer. |
| `src/plots.py` | Bar-plot comparison of learned vs. target distributions; training loss curves. |
| `src/utils.py` | NaN checks during training; parameter count logging. |
