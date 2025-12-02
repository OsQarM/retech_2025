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

    # SYSTEM TYPE
    "system_type": "open",        # "closed" or "open"
    "learn_noise": True,          # Whether to learn noise parameters
    "noise_types": ["dephasing"], # ["thermal", "dephasing"] or []

    # INITIAL STATE
    "initial_state_kind": "all_plus", 

    # ====================================================================
    # 2. ANALYSIS HYPERPARAMETERS 
    # ====================================================================
    "dt": 0.01,                   # Integration Time Step
    "N_epochs": 500,              # Total Training Steps
    "learning_rate": 1e-2,        # Optimizer Step Size

    # ====================================================================
    # 3. INITIAL GUESS (Non-Homogeneous)
    # ====================================================================
    # For L=6: Initialize with lists of length L-1 for J, L for h
    "Jx_init": [0.5] * 5,         # L-1 values
    "Jy_init": [0.5] * 5,         # L-1 values
    "Jz_init": [0.5] * 5,         # L-1 values
    "hx_init": [0.0] * 6,         # L values
    "hy_init": [0.0] * 6,         # L values
    "hz_init": [0.5] * 6,         # L values
    
    # Noise parameters initial guess
    "gamma_thermal_init": [0.01] * 6,    # L values
    "gamma_dephasing_init": [0.01] * 6,  # L values

    "INIT_PERTURB_SCALE": 0.0,    # Noise Scale for initialization

    # ====================================================================
    # 4. NDE ARCHITECTURE PARAMETERS
    # ====================================================================
    "MODEL_TYPE": "white",        # "white" or "black"
    "NN_MODEL_TYPE": "time_dependent", 
    "NN_hidden_sizes": [64],      
    "learn_theta": True,          
    "learn_gamma": True,          # Learn noise parameters

    # ====================================================================
    # 5. CURRICULUM & REGULARIZATION
    # ====================================================================
    "print_every": 10,            
    "PHASE1_SPLIT": 0.4,          # Phase 1: Train θ + φ + γ
    "PHASE2_SPLIT": 0.4,          # Phase 2: Freeze φ, Train θ + γ
    "PHASE3_SPLIT": 0.1,          # Phase 3: Freeze θ, Train φ

    "lambda_reg": 1e-1,           # L2 Regularization on NN
    "lambda_gamma": 1e-2,         # L2 Regularization on noise parameters
    "seed_init": 4321,            
    "T_extrapolate_factor": 5.0,  
}

# ============================================================
# 1. UTILITIES (Modified for Non-Homogeneous)
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

def build_xyz_individual_ops(L: int, dtype=jnp.complex64):
    '''
    Build individual Pauli operators for non-homogeneous Hamiltonian
    '''
    sx, sy, sz, id2 = paulis(dtype)
    ops_out = []
    
    # Nearest-neighbor couplings (3 types × (L-1) bonds)
    for pauli in [sx, sy, sz]:
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = pauli
            ops[i+1] = pauli
            ops_out.append(kron_n(ops))
    
    # Local fields (3 types × L sites)
    for pauli in [sx, sy, sz]:
        for i in range(L):
            ops = [id2] * L
            ops[i] = pauli
            ops_out.append(kron_n(ops))
    
    return ops_out

def build_noise_operators(L: int, dtype=jnp.complex64):
    '''
    Build jump operators for Lindblad dynamics
    '''
    sx, sy, sz, id2 = paulis(dtype)
    
    sigma_minus_list = []
    sigma_plus_list = []
    sigma_z_list = []
    
    for i in range(L):
        # σ⁻ = (σ^x - iσ^y)/2
        sigma_minus = 0.5 * (sx - 1j * sy)
        ops_minus = [id2] * L
        ops_minus[i] = sigma_minus
        sigma_minus_list.append(kron_n(ops_minus))
        
        # σ⁺ = (σ^x + iσ^y)/2
        sigma_plus = 0.5 * (sx + 1j * sy)
        ops_plus = [id2] * L
        ops_plus[i] = sigma_plus
        sigma_plus_list.append(kron_n(ops_plus))
        
        # σ^z
        ops_z = [id2] * L
        ops_z[i] = sz
        sigma_z_list.append(kron_n(ops_z))
    
    return sigma_minus_list, sigma_plus_list, sigma_z_list

def xyz_hamiltonian_nonhomogeneous(theta: Array, OPS_INDIVIDUAL: list):
    '''
    Non-homogeneous Hamiltonian from individual operators
    '''
    return sum(theta[k] * OPS_INDIVIDUAL[k] for k in range(len(theta)))

def vectorize_density_matrix(rho):
    return rho.reshape(-1, 1)

def devectorize_density_matrix(rho_vec, dim):
    return rho_vec.reshape(dim, dim)

def liouvillian_superoperator(H, jump_operators, gamma_list):
    '''
    Construct Lindblad superoperator for open system
    '''
    dim = H.shape[0]
    I = jnp.eye(dim)
    
    # Hamiltonian part
    H_super = -1j * (jnp.kron(H, I) - jnp.kron(I, H.T))
    
    L_super = H_super
    
    # Add dissipative terms
    for L_k, gamma in zip(jump_operators, gamma_list):
        if gamma > 0:
            Lk_dag = L_k.conj().T
            term1 = gamma * jnp.kron(L_k, L_k.conj())
            Lk_dag_Lk = Lk_dag @ L_k
            term2 = -0.5 * gamma * (jnp.kron(Lk_dag_Lk, I) + jnp.kron(I, Lk_dag_Lk.conj()))
            L_super += term1 + term2
    
    return L_super

def prepare_initial_state(L: int, kind: str, loaded_vector: np.ndarray = None) -> Array:
    if loaded_vector is not None:
        return jnp.array(loaded_vector, dtype=jnp.complex64)
    
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
        
    return psi

def prepare_initial_density_matrix(psi0):
    '''Convert state vector to density matrix'''
    rho0 = jnp.outer(psi0, psi0.conj())
    return vectorize_density_matrix(rho0)

def fidelity_density_matrix(rho1_vec, rho2_vec):
    '''Fidelity between two density matrices'''
    dim = int(jnp.sqrt(rho1_vec.shape[0]))
    rho1 = devectorize_density_matrix(rho1_vec, dim)
    rho2 = devectorize_density_matrix(rho2_vec, dim)
    
    # Compute fidelity F(ρ1, ρ2) = (tr√(√ρ1 ρ2 √ρ1))^2
    sqrt_rho1 = jax.scipy.linalg.sqrtm(rho1)
    middle = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_middle = jax.scipy.linalg.sqrtm(middle)
    fid = jnp.real(jnp.trace(sqrt_middle))**2
    return fid

def relative_absolute_error(theta_true: Array, theta_learned: Array) -> float:
    error_abs = jnp.sum(jnp.abs(theta_true - theta_learned))
    true_norm = jnp.sum(jnp.abs(theta_true))
    return error_abs / jnp.where(true_norm > 1e-12, true_norm, 1.0)

def make_comprehensive_observables(L):
    '''
    Build observables for all sites and pairs
    '''
    sx, sy, sz, id2 = paulis()
    obs = {}
    
    # Single qubit observables for ALL qubits
    for i in range(L):
        for name_base, op in zip(['X', 'Y', 'Z'], [sx, sy, sz]):
            ops = [id2] * L
            ops[i] = op
            obs[f'{name_base}_{i}'] = kron_n(ops)
    
    # Nearest-neighbor correlations for ALL pairs
    for i in range(L - 1):
        for name_base, op in zip(['XX', 'YY', 'ZZ'], [sx, sy, sz]):
            ops = [id2] * L
            ops[i] = op
            ops[i+1] = op
            obs[f'{name_base}_{i}{i+1}'] = kron_n(ops)
    
    def calculate_observables_from_density_matrix(rho_traj_vec):
        '''
        Calculate observables from density matrix trajectory
        rho_traj_vec: shape [N_times, dim^2]
        '''
        results = {}
        N_times = rho_traj_vec.shape[0]
        dim = int(jnp.sqrt(rho_traj_vec.shape[1]))
        
        for name, op in obs.items():
            exp_vals = []
            for k in range(N_times):
                rho_vec = rho_traj_vec[k]
                rho = devectorize_density_matrix(rho_vec, dim)
                exp_val = jnp.trace(rho @ op).real
                exp_vals.append(exp_val)
            results[name] = np.array(exp_vals)
        return results
    
    return obs, calculate_observables_from_density_matrix

# ============================================================
# 2. NDE LOGIC (Modified for Open Systems)
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

def get_nn_state_dependent_correction(nn_params, rho_vec, NN_MAP_FUN, dim_squared):
    # For state-dependent correction in open system
    rho_real_imag = jnp.concatenate([rho_vec.real, rho_vec.imag])
    NN_out = NN_MAP_FUN(nn_params, rho_real_imag)
    return NN_out[:dim_squared] + 1j * NN_out[dim_squared:]

def make_rhs_fun_open_system(L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, system_type, noise_types):
    '''
    Create RHS function for open or closed system
    '''
    dim = 2**L
    dim_squared = dim * dim
    
    def H_phys(params):
        if MODEL_TYPE == "white": 
            return xyz_hamiltonian_nonhomogeneous(params["theta"], OPS_INDIVIDUAL)
        else: 
            return jnp.zeros((dim, dim), dtype=jnp.complex64)
    
    def H_NN_time_dependent(nn_params, t):
        coeffs = get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN)
        # coeffs has length NUM_PARAMS (6L-3)
        return sum(coeffs[k] * OPS_INDIVIDUAL[k] for k in range(len(OPS_INDIVIDUAL)))
    
    def build_jump_operators_and_gammas(params):
        '''
        Build jump operators and gamma lists from parameters
        '''
        jump_operators = []
        gamma_list = []
        
        if system_type == "open" and noise_types:
            sigma_minus_list, sigma_plus_list, sigma_z_list = build_noise_operators(L)
            
            gamma_idx = 0
            if "thermal" in noise_types:
                # Add σ⁻ operators for relaxation
                jump_operators.extend(sigma_minus_list)
                if "gamma_thermal" in params:
                    gamma_list.extend(params["gamma_thermal"])
                gamma_idx += L
            
            if "dephasing" in noise_types:
                # Add σ^z operators for dephasing
                jump_operators.extend(sigma_z_list)
                if "gamma_dephasing" in params:
                    gamma_list.extend(params["gamma_dephasing"])
        
        return jump_operators, gamma_list
    
    def rhs_ode_open(t: float, rho_vec: Array, params: dict):
        '''
        Right-hand side for open system Lindblad equation
        '''
        # Physical Hamiltonian
        H_A = H_phys(params)
        
        # Neural network correction to Hamiltonian
        H_corr = jnp.zeros((dim, dim), dtype=jnp.complex64)
        if NN_MODEL_TYPE == "time_dependent":
            H_corr = H_NN_time_dependent(params["nn"], t)
        
        H_total = H_A + H_corr
        
        # Build jump operators and gamma list
        jump_operators, gamma_list = build_jump_operators_and_gammas(params)
        
        if jump_operators and gamma_list:
            # Open system: Lindblad dynamics
            L_super = liouvillian_superoperator(H_total, jump_operators, gamma_list)
            drho_dt_vec = L_super @ rho_vec
        else:
            # Closed system: von Neumann equation
            rho = devectorize_density_matrix(rho_vec, dim)
            drho_dt = -1j * (H_total @ rho - rho @ H_total)
            drho_dt_vec = vectorize_density_matrix(drho_dt)
        
        # Add state-dependent neural correction if needed
        if NN_MODEL_TYPE == "state_dependent":
            corr_term = get_nn_state_dependent_correction(params["nn"], rho_vec, NN_MAP_FUN, dim_squared)
            drho_dt_vec = drho_dt_vec + corr_term
        
        return drho_dt_vec
    
    return rhs_ode_open

def rk4_step_density_matrix(rho_vec, t, dt, rhs_fun, params):
    dt_c = jnp.asarray(dt, dtype=rho_vec.dtype)
    k1 = rhs_fun(t, rho_vec, params)
    k2 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, rho_vec + dt_c*k3, params)
    rho_next_vec = rho_vec + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Renormalize to preserve trace
    dim = int(jnp.sqrt(rho_next_vec.shape[0]))
    rho_next = devectorize_density_matrix(rho_next_vec, dim)
    trace = jnp.trace(rho_next).real
    if jnp.abs(trace - 1.0) > 1e-8:
        rho_next = rho_next / trace
        rho_next_vec = vectorize_density_matrix(rho_next)
    
    return rho_next_vec

def evolve_trajectory_density_matrix(rho0_vec, t_grid, rhs_fun, params):
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    @jax.jit
    def scan_fn(rho_prev_vec, t_dt):
        t_prev, dt = t_dt
        rho_next_vec = rk4_step_density_matrix(rho_prev_vec, t_prev, dt, rhs_fun, params)
        return rho_next_vec, rho_next_vec
    
    _, rho_traj_scan = jax.lax.scan(scan_fn, rho0_vec, (t_prev_grid, dt_grid))
    return jnp.concatenate([rho0_vec[None, :], rho_traj_scan], axis=0)

# ============================================================
# 3. LOSS AND DATA LOADING (Modified)
# ============================================================
def nde_loss_open(params, L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, 
                  system_type, noise_types, lambda_reg, lambda_gamma,
                  t_grid_shots, rho0_vec, counts_shots):
    '''
    Loss function for open system
    '''
    rhs_fun = make_rhs_fun_open_system(L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, 
                                       MODEL_TYPE, system_type, noise_types)
    
    # Evolve density matrix
    rho_traj_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_shots, rhs_fun, params)
    
    # Compute log-likelihood from diagonal elements
    N_times = len(t_grid_shots)
    dim = 2**L
    ll = 0.0
    eps = 1e-9
    
    for k in range(N_times):
        rho_vec = rho_traj_vec[k]
        rho = devectorize_density_matrix(rho_vec, dim)
        probs = jnp.diag(rho).real
        probs = probs / jnp.sum(probs)
        logp = jnp.log(probs + eps)
        ll += jnp.sum(counts_shots[k] * logp)
    
    N_tot = jnp.sum(counts_shots)
    loss_nll = -ll / N_tot
    
    # Regularization terms
    reg_nn = 0.0
    for p in jtu.tree_leaves(params["nn"]):
        reg_nn += jnp.sum(p**2)
    
    reg_gamma = 0.0
    if system_type == "open" and noise_types:
        if "gamma_thermal" in params:
            reg_gamma += jnp.sum(jnp.array(params["gamma_thermal"])**2)
        if "gamma_dephasing" in params:
            reg_gamma += jnp.sum(jnp.array(params["gamma_dephasing"])**2)
    
    total_loss = loss_nll + lambda_reg * reg_nn + lambda_gamma * reg_gamma
    
    return total_loss, (loss_nll, reg_nn, reg_gamma, rho_traj_vec)

def log_likelihood_trajectory_density_matrix(rho_traj_vec, counts, eps=1e-9):
    '''
    Compute log-likelihood from density matrix trajectory
    '''
    N_times = rho_traj_vec.shape[0]
    dim = int(jnp.sqrt(rho_traj_vec.shape[1]))
    ll = 0.0
    
    for k in range(N_times):
        rho_vec = rho_traj_vec[k]
        rho = devectorize_density_matrix(rho_vec, dim)
        probs = jnp.diag(rho).real
        probs = probs / jnp.sum(probs)
        logp = jnp.log(probs + eps)
        ll += jnp.sum(counts[k] * logp)
    
    N_tot = jnp.sum(counts)
    return ll / N_tot

def load_experimental_data_nonhomogeneous(config):
    """
    Load data with non-homogeneous parameters
    """
    L = config["L"]
    T_max = config["t_max"]
    
    # Search for matching files
    search_pattern = f"experimental_data_L{L}_T{T_max:.2f}_*_config_df.csv"
    files = glob.glob(search_pattern)
    
    if not files:
        # Try non-homogeneous pattern
        search_pattern = f"experimental_data_L{L}_T{T_max:.2f}_*_nonhomogeneous*_config_df.csv"
        files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No data found for L={L}, T={T_max:.2f}")
    
    config_file = files[0] 
    file_core = config_file.replace("_config_df.csv", "").replace("experimental_data_", "")
    
    print(f"✅ Found Data Set: {file_core}")
    
    try:
        # Load counts
        df_counts = pd.read_csv(f"experimental_data_{file_core}_counts.csv", index_col='Time')
        
        # Load config
        df_config = pd.read_csv(config_file).iloc[0]
        
        # Load metadata
        metadata = np.load(f"experimental_data_{file_core}_metadata.npz", allow_pickle=True)
        
        # Extract data
        initial_state_vector = metadata["initial_state"]
        theta_true_array = metadata["theta_true"]
        system_type = metadata.get("system_type", "closed")
        noise_types = metadata.get("noise_types", [])
        
        # Update config from data
        config["N_time_shots"] = int(df_config["N_time_shots"])
        config["N_shots_per_time"] = int(df_config["N_shots_per_time"])
        config["system_type"] = system_type
        config["noise_types"] = noise_types
        
        # Parse non-homogeneous parameters if stored as strings
        def parse_list_param(value_str, default_length):
            if isinstance(value_str, str):
                try:
                    # Parse string like "[0.9, 0.85, 0.8, 0.75, 0.7]"
                    return eval(value_str)
                except:
                    return [0.5] * default_length
            return [0.5] * default_length
        
        L = config["L"]
        if "Jx_true" in df_config:
            config["Jx_true"] = parse_list_param(df_config["Jx_true"], L-1)
            config["Jy_true"] = parse_list_param(df_config["Jy_true"], L-1)
            config["Jz_true"] = parse_list_param(df_config["Jz_true"], L-1)
            config["hx_true"] = parse_list_param(df_config["hx_true"], L)
            config["hy_true"] = parse_list_param(df_config["hy_true"], L)
            config["hz_true"] = parse_list_param(df_config["hz_true"], L)
        
        print(f"   -> Loaded {system_type} system data")
        if system_type == "open":
            print(f"   -> Noise types: {noise_types}")
        
        t_grid_shots = df_counts.index.values.astype(np.float32)
        counts_shots = df_counts.values.astype(np.int32)
        
        return t_grid_shots, counts_shots, initial_state_vector, theta_true_array, system_type, noise_types
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# ============================================================
# 4. TRAINING HELPERS (Modified for Noise Parameters)
# ============================================================
def adam_init(params):
    m = jtu.tree_map(jnp.zeros_like, params)
    v = jtu.tree_map(jnp.zeros_like, params)
    return {"step": 0, "m": m, "v": v}

def adam_update(params, grads, opt_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step = opt_state["step"] + 1
    m = jtu.tree_map(lambda m, g: beta1*m + (1-beta1)*g, opt_state["m"], grads)
    v = jtu.tree_map(lambda v, g: beta2*v + (1-beta2)*(g*g), opt_state["v"], grads)
    m_hat = jtu.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jtu.tree_map(lambda x: x / (1 - beta2**step), v)
    params_new = jtu.tree_map(lambda p, mh, vh: p - lr*mh/(jnp.sqrt(vh)+eps), params, m_hat, v_hat)
    return params_new, {"step": step, "m": m, "v": v}

def make_step_fn_open(L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, 
                     system_type, noise_types, lambda_reg, lambda_gamma, learning_rate):
    @jax.jit
    def jit_step_fn(params, opt_state, t_grid_shots, rho0_vec, counts_shots, trainable_mask):
        (loss_val, aux), grads = jax.value_and_grad(
            lambda p: nde_loss_open(p, L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, 
                                   MODEL_TYPE, system_type, noise_types, 
                                   lambda_reg, lambda_gamma, t_grid_shots, rho0_vec, counts_shots),
            has_aux=True
        )(params)
        
        # Apply trainable mask
        masked_grads = jtu.tree_map(
            lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), 
            grads, trainable_mask
        )
        
        params_new, opt_state_new = adam_update(params, masked_grads, opt_state, lr=learning_rate)
        return params_new, opt_state_new, loss_val, aux
    
    return jit_step_fn

def get_trainable_mask_open(params, train_theta, train_nn, train_gamma, system_type, noise_types):
    '''
    Create mask for trainable parameters
    '''
    mask = {}
    mask["theta"] = jtu.tree_map(lambda x: jnp.array(train_theta), params["theta"])
    mask["nn"] = jtu.tree_map(lambda x: jnp.array(train_nn), params["nn"])
    
    if system_type == "open":
        if "gamma_thermal" in params and "thermal" in noise_types:
            mask["gamma_thermal"] = jtu.tree_map(lambda x: jnp.array(train_gamma), params["gamma_thermal"])
        if "gamma_dephasing" in params and "dephasing" in noise_types:
            mask["gamma_dephasing"] = jtu.tree_map(lambda x: jnp.array(train_gamma), params["gamma_dephasing"])
    
    return mask

def train_phase_open(params_init, N_epochs, config, OPS_INDIVIDUAL, NN_MAP_FUN, 
                    NN_MODEL_TYPE, MODEL_TYPE, t_grid_shots, rho0_vec, counts_shots,
                    train_theta, train_nn, train_gamma, phase_name, step_fn):
    '''
    Training phase for open system
    '''
    if N_epochs <= 0:
        return params_init, []
    
    system_type = config["system_type"]
    noise_types = config.get("noise_types", [])
    
    train_theta_current = train_theta and (MODEL_TYPE == "white")
    train_gamma_current = train_gamma and (system_type == "open")
    
    print(f"\n--- Phase: {phase_name} ---")
    print(f"   θ: {'TRAINED' if train_theta_current else 'FROZEN'}")
    print(f"   φ: {'TRAINED' if train_nn else 'FROZEN'}")
    print(f"   γ: {'TRAINED' if train_gamma_current else 'FROZEN'}")
    
    params = params_init
    opt_state = adam_init(params_init)
    losses = []
    
    trainable_mask = get_trainable_mask_open(params, train_theta_current, train_nn, 
                                           train_gamma_current, system_type, noise_types)
    
    # Prepare true parameters for logging if available
    L = config["L"]
    if "Jx_true" in config:
        # Flatten true parameters
        theta_true_flat = np.array(
            config["Jx_true"] + config["Jy_true"] + config["Jz_true"] +
            config["hx_true"] + config["hy_true"] + config["hz_true"]
        )
    else:
        theta_true_flat = np.zeros(6 * L - 3)
    
    for epoch in range(1, N_epochs + 1):
        params, opt_state, loss_val, (loss_nll, reg_nn, reg_gamma, rho_traj) = step_fn(
            params, opt_state, t_grid_shots, rho0_vec, counts_shots, trainable_mask
        )
        
        loss_val, loss_nll, reg_nn, reg_gamma = jax.device_get((loss_val, loss_nll, reg_nn, reg_gamma))
        losses.append(float(loss_val))
        
        if epoch % config["print_every"] == 0 or epoch == N_epochs:
            print(f"[{phase_name} {epoch:03d}/{N_epochs}] loss = {loss_val:.4e} | "
                  f"nll = {loss_nll:.4e} | reg_nn = {reg_nn:.4e} | reg_γ = {reg_gamma:.4e}")
            
            if MODEL_TYPE == "white" and epoch % (config["print_every"] * 5) == 0:
                print("  θ (first 6):", np.round(np.array(params["theta"][:6]), 3))
                if system_type == "open" and "gamma_thermal" in params:
                    print("  γ_thermal (first 3):", np.round(np.array(params["gamma_thermal"][:3]), 4))
    
    return params, losses

# ============================================================
# 5. MAIN EXECUTION (Modified)
# ============================================================

if __name__ == "__main__":
    config = copy.deepcopy(CONFIG)
    
    # 1. LOAD DATA
    t_grid_shots, counts_shots, initial_state_vector, theta_true_data, system_type, noise_types = load_experimental_data_nonhomogeneous(config)
    
    L = config["L"]
    dim = 2**L
    
    # Update config with loaded system type
    config["system_type"] = system_type
    config["noise_types"] = noise_types
    
    # Prepare initial state and density matrix
    psi0 = prepare_initial_state(L, config["initial_state_kind"], initial_state_vector)
    rho0_vec = prepare_initial_density_matrix(psi0)
    
    if theta_true_data is not None:
        theta_true = jnp.array(theta_true_data, dtype=jnp.float32)
    else:
        theta_true = None
    
    t_grid_fine = jnp.arange(0.0, config["t_max"] + config["dt"]/2, config["dt"])
    
    # 2. SETUP MODEL (Non-Homogeneous)
    OPS_INDIVIDUAL = build_xyz_individual_ops(L)
    NUM_PARAMS = 6 * L - 3  # Non-homogeneous parameters
    
    NN_MODEL_TYPE = config["NN_MODEL_TYPE"]
    NN_MAP_FUN = mlp_forward
    
    if NN_MODEL_TYPE == "time_dependent":
        NN_INPUT_DIM = 1
        NN_OUTPUT_DIM = NUM_PARAMS  # Output coefficients for all operators
    elif NN_MODEL_TYPE == "state_dependent":
        NN_INPUT_DIM = 2 * dim * dim  # Real and imag parts of vectorized density matrix
        NN_OUTPUT_DIM = 2 * dim * dim
    else:
        raise ValueError(f"Unknown NN_MODEL_TYPE")
    
    # Initialize neural network
    layer_sizes = [NN_INPUT_DIM] + config["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
    key = random.PRNGKey(config["seed_init"])
    key, k_nn, k_th, k_gamma = random.split(key, 4)
    nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)
    
    # Initialize Hamiltonian parameters (non-homogeneous)
    theta_init_list = (
        config["Jx_init"] + config["Jy_init"] + config["Jz_init"] +
        config["hx_init"] + config["hy_init"] + config["hz_init"]
    )
    theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
    
    if config["INIT_PERTURB_SCALE"] > 0:
        theta_init += config["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_PARAMS,))
    
    # Initialize noise parameters if open system
    params = {"theta": theta_init, "nn": nn_params}
    
    if system_type == "open":
        if "thermal" in noise_types:
            params["gamma_thermal"] = jnp.array(config["gamma_thermal_init"], dtype=jnp.float32)
        if "dephasing" in noise_types:
            params["gamma_dephasing"] = jnp.array(config["gamma_dephasing_init"], dtype=jnp.float32)
    
    # 3. TRAINING
    step_fn = make_step_fn_open(
        L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"],
        system_type, noise_types, config["lambda_reg"], config["lambda_gamma"], 
        config["learning_rate"]
    )
    
    N_total = config["N_epochs"]
    P1 = int(N_total * config["PHASE1_SPLIT"])
    P2 = int(N_total * config["PHASE2_SPLIT"])
    P3 = N_total - P1 - P2
    
    # Phase 1: Train θ + φ + γ
    params, l1 = train_phase_open(
        params, P1, config, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, 
        config["MODEL_TYPE"], t_grid_shots, rho0_vec, counts_shots,
        train_theta=config["learn_theta"], train_nn=True, 
        train_gamma=config.get("learn_gamma", True),
        phase_name="P1 Warm-up", step_fn=step_fn
    )
    
    # Phase 2: Freeze φ, Train θ + γ
    params, l2 = train_phase_open(
        params, P2, config, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE,
        config["MODEL_TYPE"], t_grid_shots, rho0_vec, counts_shots,
        train_theta=config["learn_theta"], train_nn=False,
        train_gamma=config.get("learn_gamma", True),
        phase_name="P2 Distill", step_fn=step_fn
    )
    
    # Phase 3: Freeze θ, Train φ
    params, l3 = train_phase_open(
        params, P3, config, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE,
        config["MODEL_TYPE"], t_grid_shots, rho0_vec, counts_shots,
        train_theta=False, train_nn=True,
        train_gamma=False,  # Freeze gamma in refinement
        phase_name="P3 Refine", step_fn=step_fn
    )
    
    losses = l1 + l2 + l3
    
    # 4. DIAGNOSTICS
    T_extrap = config["t_max"] * config["T_extrapolate_factor"]
    t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, config["dt"])
    
    # Evolve with full model
    rhs_fun_model = make_rhs_fun_open_system(
        L, OPS_INDIVIDUAL, NN_MAP_FUN, NN_MODEL_TYPE, 
        config["MODEL_TYPE"], system_type, noise_types
    )
    
    rho_model_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_long, rhs_fun_model, params)
    
    # Evolve with physics only (zero NN)
    params_physics_only = params.copy()
    params_physics_only["nn"] = jtu.tree_map(jnp.zeros_like, params["nn"])
    rho_physics_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_long, rhs_fun_model, params_physics_only)
    
    # Get observables
    obs_dict, calc_obs = make_comprehensive_observables(L)
    obs_model = calc_obs(rho_model_vec)
    obs_physics = calc_obs(rho_physics_vec)
    
    # Compute NN norm
    nn_l2_norm = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))
    
    # Print results
    print("\n" + "="*70)
    print("                 FINAL LEARNED PARAMETERS")
    print("="*70)
    
    # Hamiltonian parameters
    labels = get_parameter_labels(L)
    theta_final = params["theta"]
    
    print(f"\nHamiltonian Parameters (first 10 of {len(labels)}):")
    for i in range(min(10, len(labels))):
        print(f"  {labels[i]:<10}: {theta_final[i]:.5f}")
    
    if system_type == "open":
        print(f"\nNoise Parameters:")
        if "gamma_thermal" in params:
            print(f"  γ_thermal: {np.array(params['gamma_thermal']).round(4)}")
        if "gamma_dephasing" in params:
            print(f"  γ_dephasing: {np.array(params['gamma_dephasing']).round(4)}")
    
    print(f"\nNN Parameter L2 Norm: {float(nn_l2_norm):.4e}")
    
    # Compare with ground truth if available
    if theta_true is not None:
        rel_error = relative_absolute_error(theta_true, theta_final)
        print(f"\nRelative Abs Error vs True: {rel_error:.4f}")
        
        # True dynamics
        H_true = xyz_hamiltonian_nonhomogeneous(theta_true, OPS_INDIVIDUAL)
        
        if system_type == "open":
            # Build true jump operators
            jump_ops_true = []
            gamma_list_true = []
            
            if "thermal" in noise_types:
                sigma_minus_list, _, _ = build_noise_operators(L)
                jump_ops_true.extend(sigma_minus_list)
                gamma_list_true.extend(config.get("gamma_thermal_true", [0.01]*L))
            
            if "dephasing" in noise_types:
                _, _, sigma_z_list = build_noise_operators(L)
                jump_ops_true.extend(sigma_z_list)
                gamma_list_true.extend(config.get("gamma_dephasing_true", [0.01]*L))
            
            # True open system evolution
            def rhs_true_open(t, rho_vec, params):
                L_super = liouvillian_superoperator(H_true, jump_ops_true, gamma_list_true)
                return L_super @ rho_vec
            
            rho_true_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_long, rhs_true_open, {})
        else:
            # True closed system evolution
            def rhs_true_closed(t, rho_vec, params):
                dim = int(jnp.sqrt(rho_vec.shape[0]))
                rho = devectorize_density_matrix(rho_vec, dim)
                drho_dt = -1j * (H_true @ rho - rho @ H_true)
                return vectorize_density_matrix(drho_dt)
            
            rho_true_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_long, rhs_true_closed, {})
        
        # Compute fidelities
        fid_model = np.array([fidelity_density_matrix(rho_true_vec[k], rho_model_vec[k]) 
                            for k in range(len(rho_true_vec))])
        fid_physics = np.array([fidelity_density_matrix(rho_true_vec[k], rho_physics_vec[k]) 
                              for k in range(len(rho_true_vec))])
        
        infid_model = 1 - fid_model
        infid_physics = 1 - fid_physics
        
        # True observables
        obs_true = calc_obs(rho_true_vec)
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"L={L} | {system_type} system | Noise: {noise_types}")
        
        ax1.plot(t_grid_long, fid_model, 'r', label='NDE Fidelity')
        ax1.plot(t_grid_long, fid_physics, 'b--', label='Physics-only Fidelity')
        ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Fidelity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.loglog(t_grid_long, infid_model, 'r', label='NDE Infidelity')
        ax2.loglog(t_grid_long, infid_physics, 'b--', label='Physics-only Infidelity')
        ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Infidelity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot observables
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Observables - L={L} | {system_type} system")
        
        # Select observables to plot
        observables_to_plot = ['X_0', 'Y_0', 'Z_0', 'XX_01', 'YY_01', 'ZZ_01']
        
        for i, obs_name in enumerate(observables_to_plot):
            ax = axes[i // 3, i % 3]
            if obs_name in obs_true:
                ax.plot(t_grid_long, obs_true[obs_name], 'k-', label='True', linewidth=2)
                ax.plot(t_grid_long, obs_model[obs_name], 'r--', label='NDE')
                ax.plot(t_grid_long, obs_physics[obs_name], 'b:', label='Physics-only')
                ax.set_title(obs_name)
                ax.set_xlabel('Time')
                ax.set_ylabel('Expectation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        plt.tight_layout()
        plt.show()
    
    else:
        # No ground truth - just plot model predictions
        print("\n(No ground truth available for comparison)")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Model Predictions - L={L} | {system_type} system")
        
        observables_to_plot = ['X_0', 'Y_0', 'Z_0', 'XX_01', 'YY_01', 'ZZ_01']
        
        for i, obs_name in enumerate(observables_to_plot):
            ax = axes[i // 3, i % 3]
            if obs_name in obs_model:
                ax.plot(t_grid_long, obs_model[obs_name], 'r-', label='NDE', linewidth=2)
                ax.plot(t_grid_long, obs_physics[obs_name], 'b--', label='Physics-only')
                ax.set_title(obs_name)
                ax.set_xlabel('Time')
                ax.set_ylabel('Expectation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        plt.tight_layout()
        plt.show()
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {system_type} system')
    plt.grid(True, alpha=0.3)
    plt.show()