#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:27:51 2025

@author: marcin
"""
import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import copy 
import pandas as pd
import os
import glob

# Define the Array type alias for JAX
Array = jnp.ndarray

# ============================================================
# 0. CONFIG BLOCK: Analysis Settings
# ============================================================
CONFIG = {
    # ====================================================================
    # 1. EXPERIMENT TARGET SELECTION & DEFINITION
    # ====================================================================
    "L": 6,                       # System Size 
                                   

    "t_max": 1.0,                 # Experiment Duration

    # INITIAL STATE (The state prepared at t=0)
    # Options: 
    #   - "all_plus": The uniform superposition state |+...+> (eigenstate of X).
    #   - "all_zeros": The computational basis ground state |0...0> (eigenstate of Z).
    "initial_state_kind": "all_plus", 

    # ====================================================================
    # 2. ANALYSIS HYPERPARAMETERS 
    # Control the Neural ODE solver (RK4) used during the learning process.
    # These are independent of the experimental sampling rate.
    # ====================================================================
    "dt": 0.01,                   # Integration Time Step: The internal step size for the Runge-Kutta solver.
        

    "N_epochs": 500,              # Total Training Steps: The total number of optimization loops 
                                  

    "learning_rate": 1e-2,        # Optimizer Step Size: The learning rate for the ADAM optimizer.
 

    # ====================================================================
    # 3. INITIAL GUESS 
    # ====================================================================
    "Jx_init": 0.5,               # Initial guess for Nearest-Neighbor XX coupling strength.
    "Jy_init": 0.5,               # Initial guess for Nearest-Neighbor YY coupling strength.
    "Jz_init": 0.5,               # Initial guess for Nearest-Neighbor ZZ coupling strength.
    "hx_init": 0.0,               # Initial guess for local X magnetic field strength.
    "hy_init": 0.0,               # Initial guess for local Y magnetic field strength.
    "hz_init": 0.5,               # Initial guess for local Z magnetic field strength.

    "INIT_PERTURB_SCALE": 0.0,    # Noise Scale: A value > 0.0 adds random Gaussian noise to the 
                                  # initial guess above. Useful for testing if the optimizer can 
                                  # recover the true parameters from a "bad" starting point.

    # ====================================================================
    # 4. NDE ARCHITECTURE PARAMETERS ( The Model Structure)
    #    The hybrid model: d|ψ>/dt = -i [ H_Ansatz(θ) + H_Neural(φ) ] |ψ>
    # ====================================================================
    "MODEL_TYPE": "white",        # Objective Mode:
                                  # - "white": INTERPRETABILITY. Learns static parameters θ 
                                  #   that best describe the system (requires Distillation phase).
                                  # - "black": PREDICTION. Learns a complex neural function 
                                  #   to simulate dynamics (theta is ignored).

    "NN_MODEL_TYPE": "time_dependent", 
                                  # Neural Correction Type:
                                  # - "time_dependent": The NN learns time-varying coefficients c_k(t). 
                                  # - "state_dependent": The NN learns a state-dependent map NN(|ψ>). 

    "NN_hidden_sizes": [64],      # Neural Network Depth/Width: List of hidden layer sizes for the MLP.
                                   

    "learn_theta": True,          # Optimization Flag:
                                  # - True: The optimizer updates the Ansatz parameters (θ). Required for White-Box.
                                  # - False: θ is frozen at init. Used for pure black-box error correction.

    # ====================================================================
    # 5. CURRICULUM & REGULARIZATION (The Learning Strategy)
    # Controls the three-phase training schedule designed to "distill" physics into θ.
    # ====================================================================
    "print_every": 10,            # Logging frequency (in epochs).

    # Curriculum Phases (Fractions must sum to 1.0):
    "PHASE1_SPLIT": 0.4,         # Phase 1 (Warm-up): Train BOTH (θ + NN). Finds the general solution region.
    "PHASE2_SPLIT": 0.4,         # Phase 2 (Distillation): Freeze NN, Train θ. Forces the interpretable 
                                  # Ansatz to absorb the dynamics learned by the NN.
    "PHASE3_SPLIT": 0.1,          # Phase 3 (Refinement): Freeze θ, Train NN. Cleans up residual numerical 
                                  # errors or non-physical noise.

    "lambda_reg": 1e-1,           # L2 Regularization Strength on NN parameters (φ).
                                  # CRITICAL PARAMETER for White-Box:
                                  # - High value : Penalizes the NN heavily, forcing the 
                                  #   physics to be explained by the simple Ansatz (θ).
                                  # - Low value : Allows the NN to dominate, leading to 
                                  #   high fidelity but wrong physical parameters.

    "seed_init": 4321,            # Random seed for Neural Network initialization (reproducibility).

    "T_extrapolate_factor": 5.0,  # Extrapolation Horizon:
                                  
}

# ============================================================
# 1. UTILITIES
# ============================================================
def paulis(dtype=jnp.complex64):
    sx = jnp.array([[0., 1.],[1., 0.]], dtype=dtype)
    sy = jnp.array([[0., -1j],[1j, 0.]], dtype=dtype)
    sz = jnp.array([[1., 0.],[0., -1.]], dtype=dtype)
    id2 = jnp.eye(2, dtype=dtype)
    return sx, sy, sz, id2

def kron_n(ops):
    out = ops[0]
    for A in ops[1:]: out = jnp.kron(out, A)
    return out

def build_xyz_basis(L: int, dtype=jnp.complex64):
    sx, sy, sz, id2 = paulis(dtype)
    dim = 2**L
    ops_out = []
    for pauli in [sx, sy, sz]:
        H_term = jnp.zeros((dim, dim), dtype=dtype)
        for i in range(L - 1):
            ops = [id2] * L; ops[i] = pauli; ops[i+1] = pauli
            H_term = H_term + kron_n(ops)
        ops_out.append(H_term)
    for pauli in [sx, sy, sz]:
        H_term = jnp.zeros((dim, dim), dtype=dtype)
        for i in range(L):
            ops = [id2] * L; ops[i] = pauli
            H_term = H_term + kron_n(ops)
        ops_out.append(H_term)
    return ops_out

def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list):
    return sum(theta[k] * OPS_XYZ[k] for k in range(6))

def prepare_initial_state(L: int, kind: str, loaded_vector: np.ndarray = None) -> Array:
    """
    Prepares the initial quantum state.
    - If 'loaded_vector' is provided (from simulation metadata), it uses that.
    - Otherwise, it constructs the state based on 'kind' (from CONFIG).
    """
    if loaded_vector is not None:
        print(f"   -> Using loaded initial state vector (from metadata).")
        return jnp.array(loaded_vector, dtype=jnp.complex64)
    
    print(f"   -> Constructing initial state from config: '{kind}'")
    dim = 2**L
    if kind == "all_zeros":
        # |00...0>
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        # |++...+> (Uniform Superposition)
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        # Fallback to all_zeros if unknown
        print(f"WARNING: Unknown state '{kind}'. Defaulting to |00...0>.")
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
        
    return psi

def fidelity(psi_a, psi_b):
    ov = jnp.vdot(psi_a, psi_b)
    return jnp.abs(ov)**2

def relative_absolute_error(theta_true: Array, theta_learned: Array) -> float:
    error_abs = jnp.sum(jnp.abs(theta_true - theta_learned))
    true_norm = jnp.sum(jnp.abs(theta_true))
    return error_abs / jnp.where(true_norm > 1e-12, true_norm, 1.0)

def make_observables(L):
    sx, sy, sz, id2 = paulis()
    obs = {}
    for name, op in zip(['X_0', 'Y_0', 'Z_0'], [sx, sy, sz]):
        ops = [op] + [id2]*(L-1)
        obs[name] = kron_n(ops) 
    if L >= 2:
        obs['X_0 X_1'] = kron_n([sx, sx] + [id2]*(L-2))
        obs['Y_0 Y_1'] = kron_n([sy, sy] + [id2]*(L-2))
        obs['Z_0 Z_1'] = kron_n([sz, sz] + [id2]*(L-2))

    def calculate_observables(psi_traj):
        results = {}
        for name, op in obs.items():
            exp_val = jnp.array([jnp.vdot(psi, op @ psi).real for psi in psi_traj])
            results[name] = np.array(exp_val)
        return results
    return obs, calculate_observables

# ============================================================
# 2. NDE LOGIC
# ============================================================
def init_mlp_params(layer_sizes, key, scale=0.1):
    params = []; keys = random.split(key, len(layer_sizes)-1)
    for k, (m,n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = scale * random.normal(k, (m,n)); b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params

def mlp_forward(params, x):
    h = x
    for layer in params[:-1]: h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

def get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN):
    t_input = jnp.array([[t]]); return NN_MAP_FUN(nn_params, t_input)[0] 

def get_nn_state_dependent_correction(nn_params, psi, NN_MAP_FUN, dim):
    psi_vec_in = jnp.concatenate([psi.real, psi.imag])
    NN_out_2D = NN_MAP_FUN(nn_params, psi_vec_in)
    return NN_out_2D[:dim] + 1j * NN_out_2D[dim:]

def make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE):
    dim = 2**L
    def H_phys(params):
        if MODEL_TYPE == "white": return xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ)
        else: return jnp.zeros((dim, dim), dtype=jnp.complex64) 
        
    def H_NN_time_dependent(nn_params, t):
        coeffs = get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN)
        return sum(coeffs[k] * OPS_XYZ[k] for k in range(6))
        
    def rhs_ode(t: float, psi: Array, params: dict):
        H_A = H_phys(params); phys_term = -1j * (H_A @ psi)
        if NN_MODEL_TYPE == "time_dependent":
            H_corr = H_NN_time_dependent(params["nn"], t)
            return phys_term - 1j * (H_corr @ psi)
        elif NN_MODEL_TYPE == "state_dependent":
            corr_term = get_nn_state_dependent_correction(params["nn"], psi, NN_MAP_FUN, dim)
            return phys_term + corr_term
        return phys_term
    return rhs_ode

def rk4_step(psi, t, dt, rhs_fun, params):
    dt_c = jnp.asarray(dt, dtype=psi.dtype)
    k1 = rhs_fun(t, psi, params)
    k2 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c,     psi + dt_c*k3, params)
    psi_next = psi + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return psi_next / jnp.linalg.norm(psi_next) 

def evolve_trajectory(psi0, t_grid, rhs_fun, params):
    dt_grid = t_grid[1:] - t_grid[:-1]; t_prev_grid = t_grid[:-1]
    @jax.jit
    def scan_fn(psi_prev, t_dt):
        t_prev, dt = t_dt
        psi_next = rk4_step(psi_prev, t_prev, dt, rhs_fun, params)
        return psi_next, psi_next
    _, psi_traj_scan = jax.lax.scan(scan_fn, psi0, (t_prev_grid, dt_grid))
    return jnp.concatenate([psi0[None, :], psi_traj_scan], axis=0)

# ============================================================
# 3. LOSS AND DATA LOADING
# ============================================================
def nde_loss(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, lambda_reg, t_grid_shots, psi0, counts_shots):
    rhs_fun = make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE)
    psi_traj_shots = evolve_trajectory(psi0, t_grid_shots, rhs_fun, params)
    ll = log_likelihood_trajectory(psi_traj_shots, counts_shots)
    loss_nll = -ll
    reg = 0.0
    for p in jtu.tree_leaves(params["nn"]): reg = reg + jnp.sum(p**2)
    return loss_nll + lambda_reg * reg, (loss_nll, reg, psi_traj_shots)

def log_likelihood_trajectory(psi_traj, counts, eps=1e-9):
    probs = jnp.abs(psi_traj)**2
    probs = probs / probs.sum(axis=1, keepdims=True)
    logp = jnp.log(probs + eps)
    ll = jnp.sum(counts * logp)
    N_tot = jnp.sum(counts)
    return ll / N_tot

def load_experimental_data(config):
    """
    Finds a data file matching the configured L and t_max.
    """
    L = config["L"]; T_max = config["t_max"]
    search_pattern = f"experimental_data_L{L}_T{T_max:.2f}_*_config_df.csv"
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No data found for L={L}, T={T_max:.2f}. Run generate_data.py first.")
    
    config_file = files[0] 
    file_core = config_file.replace("_config_df.csv", "").replace("experimental_data_", "")
    
    print(f"✅ Found Data Set: {file_core}")
    
    try:
        df_counts = pd.read_csv(f"experimental_data_{file_core}_counts.csv", index_col='Time')
        df_config = pd.read_csv(config_file).iloc[0]
        
        # Load metadata if it exists, but handle case where it might not (real experiment)
        try:
            metadata = np.load(f"experimental_data_{file_core}_metadata.npz")
            initial_state_vector = metadata["initial_state"]
            theta_true_array = metadata["theta_true"]
        except:
            initial_state_vector = None
            theta_true_array = None
        
        # --- UPDATE CONFIG FROM DATA ---
        config["N_time_shots"] = int(df_config["N_time_shots"])
        config["N_shots_per_time"] = int(df_config["N_shots_per_time"])
        
        # Try to load true parameters if they exist (simulated data)
        if "Jx_true" in df_config:
            config["Jx_true"] = float(df_config["Jx_true"])
            config["Jy_true"] = float(df_config["Jy_true"])
            config["Jz_true"] = float(df_config["Jz_true"])
            config["hx_true"] = float(df_config["hx_true"])
            config["hy_true"] = float(df_config["hy_true"])
            config["hz_true"] = float(df_config["hz_true"])
        else:
             # If real data, set placeholders so code doesn't crash, but they won't mean anything
             theta_true_array = None 
        
        print(f"   -> Loaded Data Structure: R={config['N_shots_per_time']}, J={config['N_time_shots']}")
        print(f"   -> Using Solver dt: {config['dt']} (User defined)")
        
        t_grid_shots = df_counts.index.values.astype(np.float32)
        counts_shots = df_counts.values.astype(np.int32)
        
        return t_grid_shots, counts_shots, initial_state_vector, theta_true_array
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# ============================================================
# 4. TRAINING HELPERS
# ============================================================
def adam_init(params):
    m = jtu.tree_map(jnp.zeros_like, params); v = jtu.tree_map(jnp.zeros_like, params)
    return {"step": 0, "m": m, "v": v}

def adam_update(params, grads, opt_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step = opt_state["step"] + 1
    m = jtu.tree_map(lambda m, g: beta1*m + (1-beta1)*g, opt_state["m"], grads)
    v = jtu.tree_map(lambda v, g: beta2*v + (1-beta2)*(g*g), opt_state["v"], grads)
    m_hat = jtu.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jtu.tree_map(lambda x: x / (1 - beta2**step), v)
    params_new = jtu.tree_map(lambda p, mh, vh: p - lr*mh/(jnp.sqrt(vh)+eps), params, m_hat, v_hat)
    return params_new, {"step": step, "m": m, "v": v}

def make_step_fn(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, lambda_reg, learning_rate):
    @jax.jit
    def jit_step_fn(params, opt_state, t_grid_shots, psi0, counts_shots, trainable_mask):
        (loss_val, aux), grads = jax.value_and_grad(
            lambda p: nde_loss(p, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, lambda_reg, t_grid_shots, psi0, counts_shots), 
            has_aux=True
        )(params)
        masked_grads = jtu.tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), grads, trainable_mask)
        params_new, opt_state_new = adam_update(params, masked_grads, opt_state, lr=learning_rate)
        return params_new, opt_state_new, loss_val, aux
    return jit_step_fn

def get_trainable_mask(params, train_theta, train_nn):
    mask = {}; mask["theta"] = jtu.tree_map(lambda x: jnp.array(train_theta), params["theta"], is_leaf=lambda x: x is params["theta"])
    mask["nn"] = jtu.tree_map(lambda x: jnp.array(train_nn), params["nn"])
    return mask

def train_phase(params_init, N_epochs, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, t_grid_shots, psi0, counts_shots, train_theta, train_nn, phase_name, step_fn):
    if N_epochs <= 0: return params_init, []
    train_theta_current = train_theta and (MODEL_TYPE == "white")
    print(f"\n--- Phase: {phase_name} ({'θ TRAINED' if train_theta_current else 'θ Frozen'}, {'φ TRAINED' if train_nn else 'φ Frozen'}) ---")
    params = params_init; opt_state = adam_init(params_init); losses = []
    trainable_mask = get_trainable_mask(params, train_theta_current, train_nn)
    
    # Prepare true params for logging if available
    if "Jx_true" in config:
        theta_true_diag = np.array([config["Jx_true"], config["Jy_true"], config["Jz_true"], config["hx_true"], config["hy_true"], config["hz_true"]])
    else:
        theta_true_diag = np.zeros(6)

    for epoch in range(1, N_epochs + 1):
        params, opt_state, loss_val, (loss_nll, reg, psi_traj) = step_fn(params, opt_state, t_grid_shots, psi0, counts_shots, trainable_mask)
        loss_val, loss_nll, reg = jax.device_get((loss_val, loss_nll, reg))
        losses.append(float(loss_val))
        if epoch % config["print_every"] == 0 or epoch == N_epochs:
            print(f"[{phase_name} {epoch:03d}/{N_epochs}] total_loss = {loss_val:.4e} | nll = {loss_nll:.4e} | reg = {reg:.4e}")
            if MODEL_TYPE == "white":
                print("  theta_true :", np.round(theta_true_diag, 3))
                print("  theta_curr :", np.round(np.array(params["theta"]), 3))
    return params, losses

#%%
# ============================================================
# 5. MAIN EXECUTION
# ============================================================

config = copy.deepcopy(CONFIG)

# 1. LOAD DATA (Overwrites config params)
t_grid_shots, counts_shots, initial_state_vector, theta_true_data = load_experimental_data(config)
L = config["L"]; dim = 2**L

# Construct state from file OR config description
psi0 = prepare_initial_state(L, config["initial_state_kind"], initial_state_vector)

if theta_true_data is not None:
    theta_true = jnp.array(theta_true_data, dtype=jnp.float32)
else:
    theta_true = None # Real experiment case

t_grid_fine = jnp.arange(0.0, config["t_max"] + config["dt"]/2, config["dt"])

# 2. SETUP MODEL
OPS_XYZ = build_xyz_basis(L); NUM_COEFFICIENTS = 6
NN_MODEL_TYPE = config["NN_MODEL_TYPE"]; NN_MAP_FUN = mlp_forward
if NN_MODEL_TYPE == "time_dependent": NN_INPUT_DIM = 1; NN_OUTPUT_DIM = NUM_COEFFICIENTS
elif NN_MODEL_TYPE == "state_dependent": NN_INPUT_DIM = 2 * dim; NN_OUTPUT_DIM = 2 * dim 
else: raise ValueError(f"Unknown NN_MODEL_TYPE")

layer_sizes = [NN_INPUT_DIM] + config["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
key = random.PRNGKey(config["seed_init"]); key, k_nn, k_th = random.split(key, 3)
nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)

theta_init_list = [config["Jx_init"], config["Jy_init"], config["Jz_init"], config["hx_init"], config["hy_init"], config["hz_init"]]
theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
if config["INIT_PERTURB_SCALE"] > 0: theta_init += config["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_COEFFICIENTS,))

params = {"theta": theta_init, "nn": nn_params}

# 3. TRAIN
step_fn = make_step_fn(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], config["lambda_reg"], config["learning_rate"])
N_total = config["N_epochs"]
P1 = int(N_total * config["PHASE1_SPLIT"]); P2 = int(N_total * config["PHASE2_SPLIT"]); P3 = N_total - P1 - P2

params, l1 = train_phase(params, P1, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], t_grid_shots, psi0, counts_shots, train_theta=config["learn_theta"], train_nn=True, phase_name="P1 Warm-up", step_fn=step_fn)
params, l2 = train_phase(params, P2, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], t_grid_shots, psi0, counts_shots, train_theta=config["learn_theta"], train_nn=False, phase_name="P2 Distill", step_fn=step_fn)
params, l3 = train_phase(params, P3, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], t_grid_shots, psi0, counts_shots, train_theta=False, train_nn=True, phase_name="P3 Refine", step_fn=step_fn)
losses = l1 + l2 + l3

# 4. DIAGNOSTICS
T_extrap = config["t_max"] * config["T_extrapolate_factor"]
t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, config["dt"])

rhs_fun_model = make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"])

params_final_theta_only = {"theta": params["theta"], "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
psi_model = evolve_trajectory(psi0, t_grid_long, rhs_fun_model, params)
psi_vanilla = evolve_trajectory(psi0, t_grid_long, rhs_fun_model, params_final_theta_only)

obs_dict, calc_obs = make_observables(L)
obs_model = calc_obs(psi_model)

theta_final = params["theta"]
nn_l2_norm = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))

print("\n==================================================================")
print("           FINAL LEARNED HAMILTONIAN PARAMETERS")
print("==================================================================")
labels = ["Jx","Jy","Jz","hx","hy","hz"]
for i in range(6):
    print(f"{labels[i]:<10}: {theta_final[i]:.5f}")
    
print(f"\nNN Parameter L2 Norm: {float(nn_l2_norm):.4e}")
#%%
# If we have ground truth (simulation), compare and plot fidelity
if theta_true is not None:
    rel_error = relative_absolute_error(theta_true, theta_final)
    print(f"True Params : {np.round(np.array(theta_true), 5)}")
    print(f"Relative Abs Error vs True: {rel_error:.4f}")
    
    # True dynamics (no NN)
    rhs_fun_true = lambda t, psi, p: -1j * xyz_hamiltonian_from_theta(L, theta_true, OPS_XYZ) @ psi
    psi_true = evolve_trajectory(psi0, t_grid_long, rhs_fun_true, {"theta": theta_true, "nn": None})
    obs_true = calc_obs(psi_true)
    
    fid_nde = 1 - np.array([fidelity(psi_true[k], psi_model[k]) for k in range(len(psi_true))])
    fid_van = 1 - np.array([fidelity(psi_true[k], psi_vanilla[k]) for k in range(len(psi_true))])
   
    # Plots with True comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("L = {:d} |".format(CONFIG["L"]) + CONFIG["initial_state_kind"])
    ax1.plot(t_grid_long, 1-fid_nde, 'r', label='NDE Fidelity')
    ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax1.legend(); ax1.set_title("Fidelity")
    ax2.loglog(t_grid_long, fid_nde, 'r', label='NDE Infidelity')
    ax2.loglog(t_grid_long, fid_van, 'b--', label='Vanilla Infidelity')
    ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax2.legend(); ax2.set_title("Infidelity")
    plt.tight_layout(); plt.show()
    
    fig, ax = plt.subplots(2,3, figsize=(15,8))
    fig.suptitle("L = {:d} |".format(CONFIG["L"]) + CONFIG["initial_state_kind"])
    singles = ['X_0', 'Y_0', 'Z_0']; doubles = ['X_0 X_1', 'Y_0 Y_1', 'Z_0 Z_1']
    for i, ob in enumerate(singles):
        ax[0, i].plot(t_grid_long, obs_true[ob], 'k', label='True')
        ax[0, i].plot(t_grid_long, obs_model[ob], 'r--', label='NDE')
        ax[0, i].set_title(ob); ax[0, i].legend(); ax[0, i].set_ylim(-1.1, 1.1)
    if L>=2:
        for i, ob in enumerate(doubles):
            if ob in obs_true:
                ax[1, i].plot(t_grid_long, obs_true[ob], 'k', label='True')
                ax[1, i].plot(t_grid_long, obs_model[ob], 'r--', label='NDE')
                ax[1, i].set_title(ob); ax[1, i].legend(); ax[1, i].set_ylim(-1.1, 1.1)
    plt.tight_layout(); plt.show()
    
else:
    # Real Experiment (No True data to compare)
    print("\n(No Ground Truth available for comparison plots)")
    # Plot only the model's predictions
    fig, ax = plt.subplots(2,3, figsize=(15,8))
    singles = ['X_0', 'Y_0', 'Z_0']; doubles = ['X_0 X_1', 'Y_0 Y_1', 'Z_0 Z_1']
    for i, ob in enumerate(singles):
        ax[0, i].plot(t_grid_long, obs_model[ob], 'r--', label='NDE')
        ax[0, i].set_title(ob); ax[0, i].legend(); ax[0, i].set_ylim(-1.1, 1.1)
    if L>=2:
        for i, ob in enumerate(doubles):
            ax[1, i].plot(t_grid_long, obs_model[ob], 'r--', label='NDE')
            ax[1, i].set_title(ob); ax[1, i].legend(); ax[1, i].set_ylim(-1.1, 1.1)
    plt.tight_layout(); plt.show()
#%%
# Plots
# plt.figure(figsize=(5,4)); plt.plot(losses); plt.title("Training Loss"); plt.show()

# if NN_MODEL_TYPE == "time_dependent":
#     coeffs_list = [get_nn_coeffs_from_params(params["nn"], t, mlp_forward) for t in t_grid_long]
#     coeffs = np.array(coeffs_list)
#     plt.figure(figsize=(10, 5))
#     lbls = ["XX","YY","ZZ","X","Y","Z"]
#     for i in range(6): plt.plot(t_grid_long, coeffs[:, i], label=lbls[i])
#     plt.axvspan(0, config["t_max"], color='gray', alpha=0.1); plt.legend(); plt.title("Learned Coefficients c_k(t)"); plt.show()

 