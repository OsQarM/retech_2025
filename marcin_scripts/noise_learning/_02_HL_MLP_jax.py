#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:27:51 2025
Modified on Tue Dec 09 11:25 2025 by oscar
Extended for Noisy Systems with Dephasing and Damping

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
    "L": 4,                       # System Size 
    "t_max": 1.0,                 # Experiment Duration

    # INITIAL STATE (The state prepared at t=0)
    "initial_state_kind": "all_plus", 

    # ====================================================================
    # 2. NOISE SETTINGS (NEW)
    # ====================================================================
    "include_noise": False,         # Whether to include noise in the model
    "noise_channels": ["dephasing", "damping"],  # Which noise channels to learn
    "initial_dephasing_rate": 1,  # Initial guess for dephasing rate (per qubit)
    "initial_damping_rate": 1,    # Initial guess for damping rate (per qubit)
    "learn_noise_rates": False,       # Whether noise rates are learnable
    "noise_regularization": 0.1,     # Regularization for noise rates (prevent overfitting noise)

    # ====================================================================
    # 3. ANALYSIS HYPERPARAMETERS 
    # ====================================================================
    "dt": 0.01,                   # Integration Time Step
    "N_epochs": 500,              # Total Training Steps
    "learning_rate": 1e-2,        # Optimizer Step Size

    # ====================================================================
    # 4. INITIAL GUESS 
    # ====================================================================
    "hamiltonian_type": "uniform_xyz",  # Options: "uniform_xyz" or "general_local_zz"
    
    # For "uniform_xyz" initialization:
    "Jx_init": 0.5,               # Initial guess for Nearest-Neighbor XX coupling strength.
    "Jy_init": 0.5,               # Initial guess for Nearest-Neighbor YY coupling strength.
    "Jz_init": 0.5,               # Initial guess for Nearest-Neighbor ZZ coupling strength.
    "hx_init": 0.0,               # Initial guess for local X magnetic field strength.
    "hy_init": 0.0,               # Initial guess for local Y magnetic field strength.
    "hz_init": 0.5,               # Initial guess for local Z magnetic field strength.
    
    # For "general_local_zz" initialization:
    "hx_list_init": [0.5]*5,      # X field initial guess for each qubit
    "hz_list_init": [0.5]*5,      # Z field initial guess for each qubit
    "Jzz_list_init": [0.5]*4,     # ZZ coupling initial guess for each bond

    "INIT_PERTURB_SCALE": 0.0,    # Noise Scale for initial parameter perturbation

    # ====================================================================
    # 5. NDE ARCHITECTURE PARAMETERS
    # ====================================================================
    "MODEL_TYPE": "white",        # "white" for interpretability, "black" for prediction
    "NN_MODEL_TYPE": "time_dependent",  # "time_dependent" or "state_dependent"
    "NN_hidden_sizes": [64],      # Neural Network hidden layer sizes
    "learn_theta": True,          # Whether to learn theta parameters

    # ====================================================================
    # 6. CURRICULUM & REGULARIZATION
    # ====================================================================
    "print_every": 20,            # Logging frequency
    "PHASE1_SPLIT": 0.4,          # Warm-up phase
    "PHASE2_SPLIT": 0.4,          # Distillation phase
    "PHASE3_SPLIT": 0.2,          # Refinement phase
    "lambda_reg": 1e-1,           # L2 Regularization on NN parameters
    "seed_init": 4321,            # Random seed
    "T_extrapolate_factor": 5.0,  # Extrapolation horizon
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


def make_observables(L):
    '''
    Build Single and 2-qubit observables for ALL qubits and ALL connected pairs.
    Returns: (obs_dict, calculate_observables) tuple
    '''
    sx, sy, sz, id2 = paulis()
    obs = {}
    
    # Single qubit observables for ALL qubits (X, Y, Z for each qubit)
    for qubit in range(L):
        # X observable
        ops = [id2] * L
        ops[qubit] = sx
        obs[f'X_{qubit}'] = kron_n(ops)
        
        # Y observable  
        ops = [id2] * L
        ops[qubit] = sy
        obs[f'Y_{qubit}'] = kron_n(ops)
        
        # Z observable
        ops = [id2] * L
        ops[qubit] = sz
        obs[f'Z_{qubit}'] = kron_n(ops)
    
    # Two-qubit observables for ALL pairs (not just adjacent)
    # For Hamiltonian learning, we typically care about connected pairs
    # This creates ZZ for all adjacent pairs (i,i+1)
    for i in range(L-1):
        # ZZ correlation for adjacent pair i, i+1
        ops = [id2] * L
        ops[i] = sz
        ops[i+1] = sz
        obs[f'Z_{i} Z_{i+1}'] = kron_n(ops)
        
        # Optional: Also create XX and YY if needed
        ops = [id2] * L
        ops[i] = sx
        ops[i+1] = sx
        obs[f'X_{i} X_{i+1}'] = kron_n(ops)
        
        ops = [id2] * L
        ops[i] = sy
        ops[i+1] = sy
        obs[f'Y_{i} Y_{i+1}'] = kron_n(ops)

    def calculate_observables(psi_traj):
        results = {}
        for name, op in obs.items():
            exp_val = jnp.array([jnp.vdot(psi, op @ psi).real for psi in psi_traj])
            results[name] = np.array(exp_val)
        return results
    
    return obs, calculate_observables

def get_theta_shape(L: int, hamiltonian_type: str) -> int:
    if hamiltonian_type == "uniform_xyz":
        return 6  # Jx, Jy, Jz, hx, hy, hz
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)  # hx_i (L), hz_i (L), Jzz_i (L-1)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def build_xyz_basis(L: int, hamiltonian_type: str = "uniform_xyz", dtype=jnp.complex64):
    sx, sy, sz, id2 = paulis(dtype)
    dim = 2**L
    
    if hamiltonian_type == "uniform_xyz":
        ops_out = []
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L - 1):
                ops = [id2] * L
                ops[i] = pauli
                ops[i+1] = pauli
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L):
                ops = [id2] * L
                ops[i] = pauli
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        return ops_out
    
    elif hamiltonian_type == "general_local_zz":
        ops_out = []
        # Local X fields
        for i in range(L):
            ops = [id2] * L
            ops[i] = sx
            ops_out.append(kron_n(ops))
        # Local Z fields
        for i in range(L):
            ops = [id2] * L
            ops[i] = sz
            ops_out.append(kron_n(ops))
        # ZZ interactions
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = sz
            ops[i+1] = sz
            ops_out.append(kron_n(ops))
        return ops_out
    
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list, hamiltonian_type: str = "uniform_xyz") -> Array:
    expected_shape = get_theta_shape(L, hamiltonian_type)
    
    if len(theta) != expected_shape:
        raise ValueError(
            f"For L={L} and hamiltonian_type='{hamiltonian_type}', "
            f"expected {expected_shape} parameters, got {len(theta)}"
        )
    
    H = jnp.zeros((2**L, 2**L), dtype=jnp.complex64)
    for i in range(expected_shape):
        H += theta[i] * OPS_XYZ[i]
    
    return H

def prepare_initial_density_matrix(L: int, kind: str, loaded_vector: np.ndarray = None) -> Array:
    """
    Prepare initial density matrix (NEW: for mixed state evolution)
    """
    if loaded_vector is not None:
        print(f"   -> Using loaded initial state vector (from metadata).")
        psi = jnp.array(loaded_vector, dtype=jnp.complex64)
    else:
        print(f"   -> Constructing initial state from config: '{kind}'")
        dim = 2**L
        if kind == "all_zeros":
            psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
        elif kind == "all_plus":
            amp = 1.0 / jnp.sqrt(dim)
            psi = jnp.full((dim,), amp, dtype=jnp.complex64)
        else:
            print(f"WARNING: Unknown state '{kind}'. Defaulting to |00...0>.")
            psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    
    # Convert to density matrix: ρ = |ψ⟩⟨ψ|
    rho = jnp.outer(psi, psi.conj())
    return rho

def get_noise_operators(L: int):
    """
    Build noise operators for dephasing and damping
    Returns: (dephasing_ops, damping_ops)
    """
    sx, sy, sz, id2 = paulis()
    dim = 2**L
    
    dephasing_ops = []
    damping_ops = []
    
    for i in range(L):
        # Dephasing: Z_i operator
        ops = [id2] * L
        ops[i] = sz
        dephasing_ops.append(kron_n(ops))
        
        # Damping: σ_- = (X - iY)/2
        sigma_minus = (sx - 1j * sy) / 2
        ops = [id2] * L
        ops[i] = sigma_minus
        damping_ops.append(kron_n(ops))
    
    return dephasing_ops, damping_ops

def vectorize_density_matrix(rho):
    """Convert density matrix to vector (column-major order)"""
    return rho.flatten()

def unvectorize_density_matrix(rho_vec, dim):
    """Convert vector back to density matrix"""
    return rho_vec.reshape(dim, dim)

# ============================================================
# 2. NDE LOGIC FOR NOISY SYSTEMS
# ============================================================

def mlp_forward(params, x):
    '''Neural network forward pass'''
    h = x
    for layer in params[:-1]: 
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

def init_mlp_params(layer_sizes, key, scale=0.1):
    '''Initialize NN parameters'''
    params = []
    keys = random.split(key, len(layer_sizes)-1)
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = scale * random.normal(k, (m, n))
        b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params


def split_nn_output(nn_out, OPS_XYZ, L, include_noise):
    #Include noise rates as learnable parameters
    num_ham = len(OPS_XYZ)
    h_coeffs = nn_out[:num_ham]

    if not include_noise:
        return h_coeffs, None, None

    gamma_z = jnp.abs(nn_out[num_ham:num_ham + L])
    gamma_m = jnp.abs(nn_out[num_ham + L:num_ham + 2*L])
    return h_coeffs, gamma_z, gamma_m


def get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN, include_noise=False, L=1):
    """
    Get coefficients from NN, including noise rates if needed
    """
    t_input = jnp.array([[t]])
    full_output = NN_MAP_FUN(nn_params, t_input)[0]
    
    if not include_noise:
        return full_output
    
    # Split output into Hamiltonian coefficients and noise rates
    num_ham_coeffs = get_theta_shape(L, "uniform_xyz") if L <= 6 else get_theta_shape(L, "general_local_zz")
    h_coeffs = full_output[:num_ham_coeffs]
    noise_rates = full_output[num_ham_coeffs:]
    
    return h_coeffs, noise_rates

def make_rhs_fun_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, include_noise=True):
    """
    Create right-hand side function for Lindblad equation
    """
    dim = 2**L
    dephasing_ops, damping_ops = get_noise_operators(L)
    
    def H_phys(params):
        if MODEL_TYPE == "white": 
            return xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ, hamiltonian_type)
        else: 
            return jnp.zeros((dim, dim), dtype=jnp.complex64)
    
    def get_noise_rates(nn_params, t):
        """Extract noise rates from NN output"""
        if not include_noise:
            return jnp.zeros(L), jnp.zeros(L)
        
        t_input = jnp.array([[t]])
        full_output = NN_MAP_FUN(nn_params, t_input)[0]
        
        # Split: first num_ham_params for H, then L for dephasing, then L for damping
        num_ham_params = len(OPS_XYZ)
        dephasing_rates = full_output[num_ham_params:num_ham_params + L]
        damping_rates = full_output[num_ham_params + L:num_ham_params + 2*L]
        
        # Ensure rates are positive
        dephasing_rates = jnp.abs(dephasing_rates)
        damping_rates = jnp.abs(damping_rates)
        
        return dephasing_rates, damping_rates
    
    def rhs_ode_lindblad(t: float, rho_vec: Array, params: dict):
        """Lindblad equation for density matrix"""
        # Convert vectorized density matrix back to matrix form
        rho = unvectorize_density_matrix(rho_vec, dim)
            
        # NN evaluation (ONCE)
        t_input = jnp.array([[t]])
        nn_out = NN_MAP_FUN(params["nn"], t_input)[0]
        h_coeffs, gamma_z, gamma_m = split_nn_output(nn_out, OPS_XYZ, L, include_noise)

        # Hamiltonian
        H_total = H_phys(params)
        
        if MODEL_TYPE != "black":
            H_nn = sum(h_coeffs[k] * OPS_XYZ[k] for k in range(len(OPS_XYZ)))
            H_total = H_total + H_nn

        comm_term = -1j * (H_total @ rho - rho @ H_total)

        if not include_noise:
            return vectorize_density_matrix(comm_term)

        # === VECTORISED DISSIPATORS ===
        Z = jnp.stack(dephasing_ops)                 # (L, dim, dim)
        Sm = jnp.stack(damping_ops)
        Sp = jnp.conj(jnp.swapaxes(Sm, -1, -2))

        # Dephasing: γ (ZρZ − ρ)
        deph = gamma_z[:, None, None] * (Z @ rho @ Z - rho)

        # Damping: γ (σ₋ ρ σ₊ − ½{σ₊σ₋, ρ})
        jump = Sm @ rho @ Sp
        anticomm = 0.5 * (Sp @ Sm @ rho + rho @ Sp @ Sm)
        damp = gamma_m[:, None, None] * (jump - anticomm)

        diss_term = jnp.sum(deph + damp, axis=0)

        return vectorize_density_matrix(comm_term + diss_term)
        
        # Vectorize for ODE solver
        return vectorize_density_matrix(total)
    
    return rhs_ode_lindblad

def rk4_step_lindblad(rho_vec, t, dt, rhs_fun, params, dim):
    """RK4 step for Lindblad equation"""
    dt_c = jnp.asarray(dt, dtype=rho_vec.dtype)
    
    k1 = rhs_fun(t, rho_vec, params)
    k2 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, rho_vec + dt_c*k3, params)
    
    rho_next_vec = rho_vec + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Enforce trace preservation (renormalize)
    rho_next = unvectorize_density_matrix(rho_next_vec, dim)
    trace = jnp.trace(rho_next).real
    rho_next = rho_next / trace
    
    return vectorize_density_matrix(rho_next)

def evolve_density_matrix(rho0, t_grid, rhs_fun, params):
    """
    Evolve density matrix according to Lindblad equation
    """
    dim = rho0.shape[0]
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    # Vectorize initial density matrix
    rho0_vec = vectorize_density_matrix(rho0)
    
    @jax.jit
    def scan_fn(rho_prev_vec, t_dt):
        t_prev, dt = t_dt
        rho_next_vec = rk4_step_lindblad(
            rho_prev_vec, t_prev, dt, rhs_fun, params, dim
        )
        return rho_next_vec, rho_next_vec

    _, rho_traj_vec = jax.lax.scan(scan_fn, rho0_vec, (t_prev_grid, dt_grid))
    
    # Convert back to density matrices
    rho_traj = jnp.concatenate([rho0[None, ...], 
                               jnp.array([unvectorize_density_matrix(rv, dim) for rv in rho_traj_vec])])
    
    return rho_traj

# ============================================================
# 3. LOSS AND DATA LOADING
# ============================================================
def nde_loss_noisy(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, 
                   hamiltonian_type, lambda_reg, noise_reg, t_grid_shots, 
                   rho0, counts_shots, include_noise=True):
    """
    Loss function for noisy system
    """
    rhs_fun = make_rhs_fun_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, 
                                 MODEL_TYPE, hamiltonian_type, include_noise)
    
    # Evolve density matrix
    rho_traj = evolve_density_matrix(rho0, t_grid_shots, rhs_fun, params)
    
    # Extract probabilities from density matrix diagonal
    dim = 2**L
    probs = jnp.diagonal(rho_traj, axis1=1, axis2=2).real
    probs = jnp.clip(probs, 1e-9, 1.0)  # Clip for numerical stability
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # Negative log-likelihood
    logp = jnp.log(probs)
    ll = jnp.sum(counts_shots * logp)
    loss_nll = -ll / jnp.sum(counts_shots)
    
    # Regularization terms
    reg_nn = 0.0
    for p in jtu.tree_leaves(params["nn"]): 
        reg_nn = reg_nn + jnp.sum(p**2)
    
    # Additional regularization for noise rates (prevent overfitting noise)
    reg_noise = 0.0
    if include_noise:
        # Penalize large noise rates
        num_ham_params = len(OPS_XYZ)
        if include_noise:
            t_inputs = t_grid_shots[:, None]
            nn_outs = jax.vmap(lambda t: NN_MAP_FUN(params["nn"], t[None, :])[0])(t_inputs)
            _, gamma_z, gamma_m = jax.vmap(
                lambda o: split_nn_output(o, OPS_XYZ, L, True)
            )(nn_outs)

            reg_noise = jnp.mean(gamma_z) + jnp.mean(gamma_m)
        else:
            reg_noise = 0.0
    
    total_loss = loss_nll + lambda_reg * reg_nn + noise_reg * reg_noise
    
    return total_loss, (loss_nll, reg_nn, reg_noise, rho_traj)

def load_experimental_data(config):
    """
    Load experimental data (modified to handle noisy data)
    """
    L = config["L"]
    T_max = config["t_max"]
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
        
        try:
            metadata = np.load(f"experimental_data_{file_core}_metadata.npz")
            initial_state_vector = metadata["initial_state"]
            theta_true_array = metadata["theta_true"]
            
            # Check if noisy data is available
            if "noise_params" in metadata:
                noise_params = metadata["noise_params"]
                print(f"   -> Loaded noise parameters: {noise_params}")
                config["noise_params_true"] = noise_params
        except:
            initial_state_vector = None
            theta_true_array = None
        
        # Update config from data
        config["N_time_shots"] = int(df_config["N_time_shots"])
        config["N_shots_per_time"] = int(df_config["N_shots_per_time"])
        
        if "hamiltonian_type" in df_config:
            config["hamiltonian_type"] = df_config["hamiltonian_type"]
        
        # Check if data indicates noisy simulation
        if "include_noise" in df_config:
            config["include_noise_data"] = bool(df_config["include_noise"])
            if config["include_noise_data"] and not config["include_noise"]:
                print("⚠️  WARNING: Data appears noisy but model is configured for unitary evolution.")
                print("   Consider setting 'include_noise': True in config")
        
        hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
        
        if hamiltonian_type == "general_local_zz" and theta_true_array is not None:
            L = config["L"]
            hx_true = list(theta_true_array[:L])
            hz_true = list(theta_true_array[L:2*L])
            Jzz_true = list(theta_true_array[2*L:])
            
            config["hx_list_true"] = hx_true
            config["hz_list_true"] = hz_true
            config["Jzz_list_true"] = Jzz_true
            
            print(f"   -> Loaded true general_local_zz parameters:")
            print(f"      hx_i: {hx_true}")
            print(f"      hz_i: {hz_true}")
            print(f"      Jzz_i: {Jzz_true}")
        
        elif hamiltonian_type == "uniform_xyz":
            if "Jx_true" in df_config:
                config["Jx_true"] = float(df_config["Jx_true"])
                config["Jy_true"] = float(df_config["Jy_true"])
                config["Jz_true"] = float(df_config["Jz_true"])
                config["hx_true"] = float(df_config["hx_true"])
                config["hy_true"] = float(df_config["hy_true"])
                config["hz_true"] = float(df_config["hz_true"])
        
        print(f"   -> Loaded Data Structure: R={config['N_shots_per_time']}, J={config['N_time_shots']}")
        print(f"   -> Hamiltonian Type: {config.get('hamiltonian_type', 'uniform_xyz')}")
        print(f"   -> Include Noise in Model: {config.get('include_noise', False)}")
        
        t_grid_shots = df_counts.index.values.astype(np.float32)
        counts_shots = df_counts.values.astype(np.int32)
        
        return t_grid_shots, counts_shots, initial_state_vector, theta_true_array
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# ============================================================
# 4. TRAINING HELPERS (Modified for noisy systems)
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

def get_trainable_mask(params, train_theta, train_nn):
    mask = {}
    mask["theta"] = jnp.ones_like(params["theta"], dtype=bool) if train_theta else jnp.zeros_like(params["theta"], dtype=bool)
    mask_nn = jtu.tree_map(lambda p: jnp.ones_like(p, dtype=bool) if train_nn else jnp.zeros_like(p, dtype=bool), params["nn"]) 
    mask["nn"] = mask_nn
    return mask

def make_step_fn_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, 
                       lambda_reg, noise_reg, learning_rate, include_noise=True):
    """
    Create step function for noisy system training
    """
    grad_fn = jax.value_and_grad(nde_loss_noisy, has_aux=True)

    @jax.jit
    def step_fn(params, opt_state, t_grid_shots, rho0, counts_shots, trainable_mask):
        (loss_val, aux), grads = grad_fn(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, 
                                         MODEL_TYPE, hamiltonian_type, lambda_reg, noise_reg,
                                         t_grid_shots, rho0, counts_shots, include_noise)
        masked_grads = jtu.tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), grads, trainable_mask)
        params_new, opt_state_new = adam_update(params, masked_grads, opt_state, lr=learning_rate)
        return params_new, opt_state_new, loss_val, aux

    return step_fn

def train_phase_noisy(params_init, N_epochs, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, 
                      MODEL_TYPE, hamiltonian_type, t_grid_shots, rho0, counts_shots, 
                      train_theta, train_nn, phase_name, step_fn):
    if N_epochs <= 0:
        return params_init, []
    
    train_theta_current = train_theta and (MODEL_TYPE == "white")
    
    print(f"\n--- Phase: {phase_name} ({'θ TRAINED' if train_theta_current else 'θ Frozen'}, "
          f"{'φ TRAINED' if train_nn else 'φ Frozen'}) ---")
    
    params = params_init
    opt_state = adam_init(params_init)
    losses = []
    trainable_mask = get_trainable_mask(params, train_theta_current, train_nn)
    
    num_params = len(params["theta"])
    print(f"  Training {num_params} parameters")
    
    for epoch in range(1, N_epochs + 1):
        params, opt_state, loss_val, (loss_nll, reg_nn, reg_noise, rho_traj) = step_fn(
            params, opt_state, t_grid_shots, rho0, counts_shots, trainable_mask)
        
        loss_val, loss_nll, reg_nn, reg_noise = jax.device_get(
            (loss_val, loss_nll, reg_nn, reg_noise))
        losses.append(float(loss_val))
        
        if epoch % config["print_every"] == 0 or epoch == N_epochs:
            print(f"[{phase_name} {epoch:03d}/{N_epochs}] total_loss = {loss_val:.4e} | "
                  f"nll = {loss_nll:.4e} | reg_nn = {reg_nn:.4e} | reg_noise = {reg_noise:.4e}")
            if MODEL_TYPE == "white" and num_params <= 20:
                print(f"  theta_curr : {np.round(np.array(params['theta']), 3)}")
    
    return params, losses

# ============================================================
# 5. VISUALIZATION AND ANALYSIS
# ============================================================
def plot_noise_rates(t_grid, params, OPS_XYZ, NN_MAP_FUN, L):
    """Plot learned noise rates over time"""
    times = np.linspace(0, np.max(t_grid), 100)
    dephasing_rates = []
    damping_rates = []
    
    for t in times:
        output = NN_MAP_FUN(params["nn"], jnp.array([[t]]))[0]
        num_ham_params = len(OPS_XYZ)
        gamma_z = output[num_ham_params:num_ham_params + L]
        gamma_m = output[num_ham_params + L:num_ham_params + 2*L]
        dephasing_rates.append(np.array(gamma_z))
        damping_rates.append(np.array(gamma_m))
    
    dephasing_rates = np.array(dephasing_rates)
    damping_rates = np.array(damping_rates)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    for i in range(L):
        ax1.plot(times, dephasing_rates[:, i], label=f'Qubit {i}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dephasing rate γ_z(t)')
    ax1.set_title('Learned Dephasing Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i in range(L):
        ax2.plot(times, damping_rates[:, i], label=f'Qubit {i}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Damping rate γ_m(t)')
    ax2.set_title('Learned Damping Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_purity(rho_traj):
    """Calculate purity Tr(ρ²) for each timestep"""
    purities = []
    for rho in rho_traj:
        purity = jnp.trace(rho @ rho).real
        purities.append(purity)
    return np.array(purities)

# ============================================================
# 6. MAIN EXECUTION
# ============================================================

config = copy.deepcopy(CONFIG)

# 1. LOAD DATA
t_grid_shots, counts_shots, initial_state_vector, theta_true_data = load_experimental_data(config)
L = config["L"]
dim = 2**L

# 2. PREPARE INITIAL DENSITY MATRIX
rho0 = prepare_initial_density_matrix(L, config["initial_state_kind"], initial_state_vector)

if theta_true_data is not None:
    theta_true = jnp.array(theta_true_data, dtype=jnp.float32)
else:
    theta_true = None

t_grid_fine = jnp.arange(0.0, config["t_max"] + config["dt"]/2, config["dt"])

# 3. SETUP MODEL
hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
NUM_COEFFICIENTS = get_theta_shape(L, hamiltonian_type)

print(f"Number of Hamiltonian parameters: {NUM_COEFFICIENTS}")
print(f"Include noise in model: {config['include_noise']}")

# 4. SETUP NEURAL NETWORK
NN_MODEL_TYPE = config["NN_MODEL_TYPE"]
NN_MAP_FUN = mlp_forward

# Adjust output dimension based on noise inclusion
if config["include_noise"]:
    # Output: H coefficients + dephasing rates + damping rates
    NN_OUTPUT_DIM = NUM_COEFFICIENTS + 2*L
else:
    NN_OUTPUT_DIM = NUM_COEFFICIENTS

if NN_MODEL_TYPE == "time_dependent": 
    NN_INPUT_DIM = 1
elif NN_MODEL_TYPE == "state_dependent": 
    NN_INPUT_DIM = 2 * dim
else: 
    raise ValueError(f"Unknown NN_MODEL_TYPE")

layer_sizes = [NN_INPUT_DIM] + config["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
key = random.PRNGKey(config["seed_init"])
key, k_nn, k_th = random.split(key, 3)
nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)

# 5. INITIALIZE PARAMETERS
if hamiltonian_type == "uniform_xyz":
    theta_init_list = [
        config["Jx_init"], config["Jy_init"], config["Jz_init"],
        config["hx_init"], config["hy_init"], config["hz_init"]
    ]
elif hamiltonian_type == "general_local_zz":
    L = config["L"]
    hx_init_list = config.get("hx_list_init", [0.0] * L)
    hz_init_list = config.get("hz_list_init", [0.5] * L)
    Jzz_init_list = config.get("Jzz_list_init", [0.5] * (L-1))
    
    if len(hx_init_list) != L:
        raise ValueError(f"hx_list_init must have length L={L}, got {len(hx_init_list)}")
    if len(hz_init_list) != L:
        raise ValueError(f"hz_list_init must have length L={L}, got {len(hz_init_list)}")
    if len(Jzz_init_list) != L-1:
        raise ValueError(f"Jzz_list_init must have length L-1={L-1}, got {len(Jzz_init_list)}")
    
    theta_init_list = list(hx_init_list) + list(hz_init_list) + list(Jzz_init_list)
else:
    raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
if config["INIT_PERTURB_SCALE"] > 0: 
    theta_init += config["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_COEFFICIENTS,))

params = {"theta": theta_init, "nn": nn_params}

# 6. TRAIN
include_noise = config.get("include_noise", False)
noise_reg = config.get("noise_regularization", 0.1)

step_fn = make_step_fn_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], 
                             hamiltonian_type, config["lambda_reg"], noise_reg, 
                             config["learning_rate"], include_noise)

N_total = config["N_epochs"]
P1 = int(N_total * config["PHASE1_SPLIT"])
P2 = int(N_total * config["PHASE2_SPLIT"])
P3 = N_total - P1 - P2

print("\n" + "="*60)
print("STARTING TRAINING (Noisy NDE Model)")
print("="*60)

# Phase 1: Warm-up (train both θ and NN)
params, l1 = train_phase_noisy(params, P1, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE,
                               config["MODEL_TYPE"], hamiltonian_type, t_grid_shots,
                               rho0, counts_shots, train_theta=config["learn_theta"],
                               train_nn=True, phase_name="P1 Warm-up", step_fn=step_fn)

# Phase 2: Distillation (freeze NN, train θ only)
params, l2 = train_phase_noisy(params, P2, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE,
                               config["MODEL_TYPE"], hamiltonian_type, t_grid_shots,
                               rho0, counts_shots, train_theta=config["learn_theta"],
                               train_nn=False, phase_name="P2 Distill", step_fn=step_fn)

# Phase 3: Refinement (freeze θ, train NN only)
params, l3 = train_phase_noisy(params, P3, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE,
                               config["MODEL_TYPE"], hamiltonian_type, t_grid_shots,
                               rho0, counts_shots, train_theta=False,
                               train_nn=True, phase_name="P3 Refine", step_fn=step_fn)

losses = l1 + l2 + l3

# 7. DIAGNOSTICS

T_extrap = config["t_max"] * config["T_extrapolate_factor"]
t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, config["dt"])

# Create RHS function for final evaluation with and without noise
rhs_fun_noisy = make_rhs_fun_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE,
                                   config["MODEL_TYPE"], hamiltonian_type, include_noise)

rhs_fun_unitary = make_rhs_fun_noisy(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE,
                                   config["MODEL_TYPE"], hamiltonian_type, include_noise=False)

# Evolve with final parameters
rho_traj_model = evolve_density_matrix(
    rho0, t_grid_long, rhs_fun_noisy, params
)

rho_traj_vanilla = evolve_density_matrix(
    rho0, t_grid_long, rhs_fun_unitary,
    {"theta": params["theta"], "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
)

# Extract final parameters
theta_final = params["theta"]
nn_l2_norm = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))



###############################################
#DIAGNOSTICS
###############################################


def extract_noise_rates(params, t_grid, OPS_XYZ, L):
    t_inputs = jnp.array(t_grid)[:, None]
    nn_outs = jax.vmap(lambda t: NN_MAP_FUN(params["nn"], t[None, :])[0])(t_inputs)

    _, gamma_z, gamma_m = jax.vmap(
        lambda o: split_nn_output(o, OPS_XYZ, L, True)
    )(nn_outs)

    return np.array(gamma_z), np.array(gamma_m)

def relative_absolute_error(theta_true: Array, theta_learned: Array) -> float:
    """
    Calculate relative absolute error between true and learned parameters.
    
    Args:
        theta_true: True parameter values
        theta_learned: Learned parameter values
        
    Returns:
        Relative absolute error: sum|θ_true - θ_learned| / sum|θ_true|
    """
    # Convert to numpy arrays if they're JAX arrays
    if hasattr(theta_true, 'device_buffer'):  # JAX array
        theta_true_np = np.array(theta_true)
        theta_learned_np = np.array(theta_learned)
    else:
        theta_true_np = theta_true
        theta_learned_np = theta_learned
    
    error_abs = np.sum(np.abs(theta_true_np - theta_learned_np))
    true_norm = np.sum(np.abs(theta_true_np))
    
    # Avoid division by zero
    if true_norm < 1e-12:
        return error_abs
    
    return error_abs / true_norm


def plot_hamiltonian_histogram(theta_true, theta_init, theta_final, hamiltonian_type, L):
    """
    Plot histogram comparing true, initial, and learned Hamiltonian parameters.
    
    Args:
        theta_true: True parameter values (can be None)
        theta_init: Initial parameter guesses
        theta_final: Learned parameter values
        hamiltonian_type: "uniform_xyz" or "general_local_zz"
        L: System size (number of qubits)
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy arrays if needed
    if hasattr(theta_init, 'device_buffer'):  # JAX array
        theta_init = np.array(theta_init)
        theta_final = np.array(theta_final)
        if theta_true is not None and hasattr(theta_true, 'device_buffer'):
            theta_true = np.array(theta_true)
    
    # Prepare labels based on hamiltonian_type
    if hamiltonian_type == "uniform_xyz":
        labels = ["Jx", "Jy", "Jz", "hx", "hy", "hz"]
    elif hamiltonian_type == "general_local_zz":
        labels = [f"hx_{i}" for i in range(L)] + \
                 [f"hz_{i}" for i in range(L)] + \
                 [f"Jzz_{i}" for i in range(L-1)]
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
    n_params = len(theta_final)
    x = np.arange(n_params)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    if theta_true is not None:
        # Check if true parameters have the right length
        if len(theta_true) == n_params:
            ax.bar(x - width, theta_true, width, label='True', alpha=0.8, color='green')
        else:
            print(f"Warning: True parameters length {len(theta_true)} doesn't match expected {n_params}")
    
    ax.bar(x, theta_init, width, label='Initial Guess', alpha=0.8, color='blue')
    ax.bar(x + width, theta_final, width, label='Learned', alpha=0.8, color='red')
    
    # Customize plot
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.set_title(f'Hamiltonian Parameters Comparison ({hamiltonian_type}, L={L})')
    ax.set_xticks(x)
    
    # Use labels if we have them, otherwise use indices
    if len(labels) >= n_params:
        ax.set_xticklabels(labels[:n_params], rotation=45, ha='right', fontsize=10)
    else:
        ax.set_xticklabels([str(i) for i in range(n_params)], rotation=45, ha='right')
    
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add value annotations on top of bars
    for i in range(n_params):
        # Find the maximum y-value among the bars at this position
        y_values = []
        if theta_true is not None and len(theta_true) == n_params:
            y_values.append(theta_true[i])
        y_values.append(theta_init[i])
        y_values.append(theta_final[i])
        
        max_y = max(y_values)
        min_y = min(y_values)
        
        # Add value text for learned parameters
        ax.text(i + width, theta_final[i] + 0.05 * (max_y - min_y + 0.1), 
                f'{theta_final[i]:.3f}', 
                ha='center', va='bottom', fontsize=8, color='red')
        
        # Add value text for initial guess (if significantly different)
        if abs(theta_init[i] - theta_final[i]) > 0.05:
            ax.text(i, theta_init[i] + 0.05 * (max_y - min_y + 0.1), 
                    f'{theta_init[i]:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='blue')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Adjust y-limits to accommodate annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.15)  # Add 15% margin at top
    
    plt.tight_layout()
    return fig

print("\n" + "="*60)
print("FINAL LEARNED PARAMETERS")
print("="*60)

if hamiltonian_type == "uniform_xyz":
    labels = ["Jx", "Jy", "Jz", "hx", "hy", "hz"]
    for i, label in enumerate(labels):
        print(f"{label:<10}: {theta_final[i]:.5f}")
        
elif hamiltonian_type == "general_local_zz":
    L = config["L"]
    hx_final = theta_final[:L]
    hz_final = theta_final[L:2*L]
    Jzz_final = theta_final[2*L:]
    
    print("Local X fields (hx_i):")
    for i in range(L):
        print(f"  Qubit {i}: {hx_final[i]:.5f}")
    
    print("\nLocal Z fields (hz_i):")
    for i in range(L):
        print(f"  Qubit {i}: {hz_final[i]:.5f}")
    
    print("\nZZ couplings (Jzz_i for bond i,i+1):")
    for i in range(L-1):
        print(f"  Bond {i}-{i+1}: {Jzz_final[i]:.5f}")

gamma_z_traj, gamma_m_traj = extract_noise_rates(
    params, t_grid_long, OPS_XYZ, L
)

print("\nLearned dephasing rates γ_z(t):")
print(gamma_z_traj)

print("\nLearned damping rates γ_m(t):")
print(gamma_m_traj)
    
print(f"\nNN Parameter L2 Norm: {float(nn_l2_norm):.4e}")

# 5. COMPARE WITH TRUE PARAMETERS IF AVAILABLE
if theta_true is not None:
    rel_error = relative_absolute_error(theta_true, theta_final)
    print(f"\nTrue Params : {np.round(np.array(theta_true), 5)}")
    print(f"Relative Abs Error vs True: {rel_error:.4f}")
    
    # Plot histogram of parameters
    fig_hist = plot_hamiltonian_histogram(
        np.array(theta_true), 
        np.array(theta_init), 
        np.array(theta_final), 
        hamiltonian_type, 
        L
    )
    plt.show()
    
    # True dynamics (no NN) - need to create for density matrices
    def make_rhs_fun_true():
        """Create RHS function for true Hamiltonian (without noise for comparison)"""
        H_true = xyz_hamiltonian_from_theta(L, theta_true, OPS_XYZ, hamiltonian_type)
        
        def rhs_ode_true(t: float, rho_vec: Array, params: dict):
            rho = unvectorize_density_matrix(rho_vec, 2**L)
            # Unitary evolution only for true system
            drho_dt = -1j * (H_true @ rho - rho @ H_true)
            return vectorize_density_matrix(drho_dt)
        
        return rhs_ode_true
    
    # Create true evolution function
    rhs_fun_true = make_rhs_fun_true()
    
    # Evolve with true parameters (unitary only, no noise for comparison)
    rho_true = evolve_density_matrix(rho0, t_grid_long, rhs_fun_true, {"theta": theta_true, "nn": None})
    
    # Calculate observables from density matrices
    def calculate_observables_from_rho(rho_traj, obs_dict):
        """Calculate expectation values from density matrices"""
        results = {}
        for name, op in obs_dict.items():
            exp_val = jnp.array([jnp.trace(rho @ op).real for rho in rho_traj])
            results[name] = np.array(exp_val)
        return results
    
    obs_dict, _ = make_observables(L)

    # Get observables
    obs_true = calculate_observables_from_rho(rho_true, obs_dict)
    obs_model = calculate_observables_from_rho(rho_traj_model, obs_dict)
    obs_vanilla = calculate_observables_from_rho(rho_traj_vanilla, obs_dict)
    
    # Calculate fidelities (for mixed states, use quantum fidelity)
    def quantum_fidelity(rho1, rho2):
        """Quantum fidelity for mixed states: F(ρ1, ρ2) = (Tr√(√ρ1 ρ2 √ρ1))^2"""
        # For simplicity, use Uhlmann fidelity or trace distance
        # Using 1 - 0.5 * ||ρ1 - ρ2||_1 (trace distance)
        return 1 - 0.5 * jnp.sum(jnp.abs(jnp.linalg.eigvals(rho1 - rho2)))
    
    # Calculate fidelities
    fid_nde = np.array([quantum_fidelity(rho_true[k], rho_traj_model[k]) for k in range(len(rho_true))])
    fid_van = np.array([quantum_fidelity(rho_true[k], rho_traj_vanilla[k]) for k in range(len(rho_true))])
    
    infid_nde = 1 - fid_nde
    infid_van = 1 - fid_van
    
    # Plots with True comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {hamiltonian_type}")
    ax1.plot(t_grid_long, fid_nde, 'r', label='NDE Fidelity', linewidth=2)
    ax1.plot(t_grid_long, fid_van, 'b--', label='Vanilla Fidelity', linewidth=2)
    ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1)
    ax1.legend()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fidelity')
    ax1.set_title("Fidelity to True Evolution")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    ax2.loglog(t_grid_long, infid_nde, 'r', label='NDE Infidelity', linewidth=2)
    ax2.loglog(t_grid_long, infid_van, 'b--', label='Vanilla Infidelity', linewidth=2)
    ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1)
    ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Infidelity (log scale)')
    ax2.set_title("Infidelity (log scale)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Plot observables
    if hamiltonian_type == "general_local_zz":
        # Separate observables
        single_X_obs = [f'X_{i}' for i in range(L)]
        single_Z_obs = [f'Z_{i}' for i in range(L)]
        zz_obs = [f'Z_{i} Z_{i+1}' for i in range(L-1)]
        
        all_obs_to_plot = single_X_obs + single_Z_obs + zz_obs
        n_obs = len(all_obs_to_plot)
        
        # Calculate grid dimensions
        n_cols = min(4, n_obs)  # Maximum 4 columns
        n_rows = (n_obs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for idx, obs_name in enumerate(all_obs_to_plot):
            ax = axes_flat[idx]
            if obs_name in obs_true:
                ax.plot(t_grid_long, obs_true[obs_name], 'k-', label='True', linewidth=2)
            ax.plot(t_grid_long, obs_model[obs_name], 'r--', label='NDE', linewidth=2)
            ax.plot(t_grid_long, obs_vanilla[obs_name], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
            ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
            ax.set_xlabel('Time')
            ax.set_ylabel(f'⟨{obs_name}⟩')
            ax.set_title(obs_name)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)
            
            # Only show legend on first plot
            if idx == 0:
                ax.legend()
        
        # Hide unused subplots
        for idx in range(len(all_obs_to_plot), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.suptitle(f"All Observables | L = {L} | {config['initial_state_kind']} | {hamiltonian_type}", y=1.02)
        plt.tight_layout()
        plt.show()

    elif hamiltonian_type == "uniform_xyz":
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {hamiltonian_type}")
        
        singles = ['X_0', 'Y_0', 'Z_0']
        doubles = ['X_0 X_1', 'Y_0 Y_1', 'Z_0 Z_1']
        
        for i, ob in enumerate(singles):
            ax = axes[0, i]
            ax.plot(t_grid_long, obs_true[ob], 'k-', label='True', linewidth=2)
            ax.plot(t_grid_long, obs_model[ob], 'r--', label='NDE', linewidth=2)
            ax.plot(t_grid_long, obs_vanilla[ob], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
            ax.set_title(ob)
            ax.legend()
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        if L >= 2:
            for i, ob in enumerate(doubles):
                ax = axes[1, i]
                if ob in obs_true:
                    ax.plot(t_grid_long, obs_true[ob], 'k-', label='True', linewidth=2)
                ax.plot(t_grid_long, obs_model[ob], 'r--', label='NDE', linewidth=2)
                ax.plot(t_grid_long, obs_vanilla[ob], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
                ax.set_title(ob)
                ax.legend()
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Real Experiment (No True data to compare)
        print("\n(No Ground Truth available for comparison plots)")
        # Plot only the model's predictions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"L = {L} | {config['initial_state_kind']}")
        
        singles = ['X_0', 'Y_0', 'Z_0']
        doubles = ['X_0 X_1', 'Y_0 Y_1', 'Z_0 Z_1']
        
        for i, ob in enumerate(singles):
            ax = axes[0, i]
            ax.plot(t_grid_long, obs_model[ob], 'r--', label='NDE', linewidth=2)
            ax.plot(t_grid_long, obs_vanilla[ob], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
            ax.set_title(ob)
            ax.legend()
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        if L >= 2:
            for i, ob in enumerate(doubles):
                ax = axes[1, i]
                ax.plot(t_grid_long, obs_model[ob], 'r--', label='NDE', linewidth=2)
                ax.plot(t_grid_long, obs_vanilla[ob], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
                ax.set_title(ob)
                ax.legend()
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.3)
                ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        
        plt.tight_layout()
        plt.show()

# 6. PLOT TRAINING LOSS
plt.figure(figsize=(8, 5))
plt.plot(losses, 'b-', linewidth=2)
plt.title(f"Training Loss (Noise: {include_noise})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()

# 7. PLOT NOISE RATES (if noise is included)
if include_noise:
    fig_noise = plot_noise_rates(t_grid_long, params, OPS_XYZ, NN_MAP_FUN, L)
    plt.show()
    
    # Calculate and plot purity
    purity_model = calculate_purity(rho_traj_model)
    purity_vanilla = calculate_purity(rho_traj_vanilla)
    
    fig_purity, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_grid_long, purity_model, 'r-', label='NDE Model', linewidth=2)
    ax.plot(t_grid_long, purity_vanilla, 'b--', label='Vanilla (θ only)', linewidth=2)
    ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Purity Tr(ρ²)')
    ax.set_title('State Purity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)