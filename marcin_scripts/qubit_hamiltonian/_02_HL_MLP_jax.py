#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 19:27:51 2025
Modified on Tue Dec 09 11:25 2025 by oscar

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
    "L": 5,                       # System Size 
                                   

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
    "hamiltonian_type": "uniform_xyz",  # Options: "uniform_xyz" or "general_local_zz"
    
    # For "uniform_xyz" initialization:
    "Jx_init": 0.5,               # Initial guess for Nearest-Neighbor XX coupling strength.
    "Jy_init": 0.5,               # Initial guess for Nearest-Neighbor YY coupling strength.
    "Jz_init": 0.5,               # Initial guess for Nearest-Neighbor ZZ coupling strength.
    "hx_init": 0.0,               # Initial guess for local X magnetic field strength.
    "hy_init": 0.0,               # Initial guess for local Y magnetic field strength.
    "hz_init": 0.5,               # Initial guess for local Z magnetic field strength.
    
    # For "general_local_zz" initialization:
    "hx_list_init": [0.5]*6,  # X field initial guess for each qubit #L
    "hz_list_init": [0.5]*6,  # Z field initial guess for each qubit #L
    "Jzz_list_init": [0.5]*5,      # ZZ coupling initial guess for each bond #(L-1)

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
    "print_every": 20,            # Logging frequency (in epochs).

    # Curriculum Phases (Fractions must sum to 1.0): they don't in this example
    "PHASE1_SPLIT": 0.4,         # Phase 1 (Warm-up): Train BOTH (θ + NN). Finds the general solution region.
    "PHASE2_SPLIT": 0.4,         # Phase 2 (Distillation): Freeze NN, Train θ. Forces the interpretable 
                                  # Ansatz to absorb the dynamics learned by the NN.
    "PHASE3_SPLIT": 0.2,          # Phase 3 (Refinement): Freeze θ, Train NN. Cleans up residual numerical 
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

def get_theta_shape(L: int, hamiltonian_type: str) -> int:
    '''
    Returns the expected number of parameters for given L and hamiltonian_type
    '''
    if hamiltonian_type == "uniform_xyz":
        return 6  # Jx, Jy, Jz, hx, hy, hz
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)  # hx_i (L), hz_i (L), Jzz_i (L-1)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def build_xyz_basis(L: int, hamiltonian_type: str = "uniform_xyz", dtype=jnp.complex64):
    '''
    Builds operators for Hamiltonian construction.
    
    For "uniform_xyz": returns 6 operators as before
    For "general_local_zz": returns 2L + (L-1) operators for hx_i, hz_i, Jzz_i
    
    IMPORTANT: For consistency, all operators are returned with POSITIVE coefficients.
    The sign is determined by the theta parameters.
    '''
    sx, sy, sz, id2 = paulis(dtype)
    dim = 2**L
    
    if hamiltonian_type == "uniform_xyz":
        # Original XYZ model with uniform parameters
        ops_out = []
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L - 1):
                ops = [id2] * L
                ops[i] = pauli
                ops[i+1] = pauli  # NN term
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L):
                ops = [id2] * L
                ops[i] = pauli  # Single qubit term
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        return ops_out
    
    elif hamiltonian_type == "general_local_zz":
        # General model with local X, Z fields and ZZ interactions
        ops_out = []
        
        # 1. Local X fields for each qubit (L operators)
        for i in range(L):
            ops = [id2] * L
            ops[i] = sx
            ops_out.append(kron_n(ops))
        
        # 2. Local Z fields for each qubit (L operators)
        for i in range(L):
            ops = [id2] * L
            ops[i] = sz
            ops_out.append(kron_n(ops))
        
        # 3. ZZ interactions for each bond (L-1 operators)
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = sz
            ops[i+1] = sz
            ops_out.append(kron_n(ops))
        
        return ops_out
    
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list, hamiltonian_type: str = "uniform_xyz") -> Array:
    '''
    Creates Hamiltonian from list of weights (theta) and list of operators.
    
    Supports different parameterizations:
    - uniform_xyz: 6 parameters (Jx, Jy, Jz, hx, hy, hz)
    - general_local_zz: 2L + (L-1) parameters (hx_i, hz_i, Jzz_i)
    '''
    expected_shape = get_theta_shape(L, hamiltonian_type)
    
    # Check parameter count
    if len(theta) != expected_shape:
        raise ValueError(
            f"For L={L} and hamiltonian_type='{hamiltonian_type}', "
            f"expected {expected_shape} parameters, got {len(theta)}"
        )
    
    # Check operator count
    if len(OPS_XYZ) != expected_shape:
        raise ValueError(
            f"For L={L} and hamiltonian_type='{hamiltonian_type}', "
            f"expected {expected_shape} operators, got {len(OPS_XYZ)}"
        )
    
    # Build Hamiltonian
    H = jnp.zeros((2**L, 2**L), dtype=jnp.complex64)
    for i in range(expected_shape):
        H += theta[i] * OPS_XYZ[i]
    
    return H

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
    '''
    Build Single and 2-qubit observables for ALL qubits and ALL connected pairs.
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


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

# ============================================================
# 2. NDE LOGIC
# ============================================================

def mlp_forward(params, x):
    '''Makes forward step of NN with specific activation function. Returns array of H coefficients (?)
    NN structure [1]->[64]->[6]'''
    h = x
    for layer in params[:-1]: h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def init_mlp_params(layer_sizes, key, scale=0.1):
    '''Initialize NN parameters randomly given the sizes of input and output '''
    params = []; keys = random.split(key, len(layer_sizes)-1)
    for k, (m,n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = scale * random.normal(k, (m,n)); b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params

def get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN):
    #t_input attay of times (only 1 for time_dependent)
    t_input = jnp.array([[t]]); return NN_MAP_FUN(nn_params, t_input)[0] 

def get_nn_state_dependent_correction(nn_params, psi, NN_MAP_FUN, dim):
    psi_vec_in = jnp.concatenate([psi.real, psi.imag])
    NN_out_2D = NN_MAP_FUN(nn_params, psi_vec_in)
    return NN_out_2D[:dim] + 1j * NN_out_2D[dim:]

def make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type):
    '''Return function for rhs of NN equation'''
    dim = 2**L
    def H_phys(params):
        '''Builds H from theta parameters'''
        if MODEL_TYPE == "white": 
            return xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ, hamiltonian_type)
        else: 
            return jnp.zeros((dim, dim), dtype=jnp.complex64) 
        
    def H_NN_time_dependent(nn_params, t):
        #gets H coefficients from nn parameters
        coeffs = get_nn_coeffs_from_params(nn_params, t, NN_MAP_FUN)
        return sum(coeffs[k] * OPS_XYZ[k] for k in range(len(OPS_XYZ)))
        
    def rhs_ode(t: float, psi: Array, params: dict):
        '''Gives ODE from NN parameters'''
        H_A = H_phys(params); phys_term = -1j * (H_A @ psi) #physical term from theta parameters
        #NN correction term
        if NN_MODEL_TYPE == "time_dependent":
            H_corr = H_NN_time_dependent(params["nn"], t)
            return phys_term - 1j * (H_corr @ psi)
        elif NN_MODEL_TYPE == "state_dependent":
            corr_term = get_nn_state_dependent_correction(params["nn"], psi, NN_MAP_FUN, dim)
            return phys_term + corr_term
        return phys_term
    return rhs_ode





#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

def rk4_step(psi, t, dt, rhs_fun, params):
    dt_c = jnp.asarray(dt, dtype=psi.dtype)
    k1 = rhs_fun(t, psi, params)
    k2 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c,     psi + dt_c*k3, params)
    psi_next = psi + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return psi_next / jnp.linalg.norm(psi_next) 

def evolve_trajectory(psi0, t_grid, rhs_fun, params):
    '''
    Performs full time evolution of initial state according to specific differential equation (rhs_fun),
    with given variational parameters, and for every time in a given list (t_grid)
    Returns list of all states in the trajectory
    '''
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]

    @jax.jit
    def scan_fn(psi_prev, t_dt):
        t_prev, dt = t_dt
        psi_next = rk4_step(psi_prev, t_prev, dt, rhs_fun, params)
        # return (next_carry, output) as required by lax.scan
        return psi_next, psi_next

    _, state_traj_scan = jax.lax.scan(scan_fn, psi0, (t_prev_grid, dt_grid))
    return jnp.concatenate([psi0[None, ...], state_traj_scan], axis=0)

# ============================================================
# 3. LOSS AND DATA LOADING
# ============================================================
def nde_loss(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, lambda_reg, t_grid_shots, psi0, counts_shots):
    '''Evolves differential equation with given parameters outputed from the NN and calculates the loss function'''
    rhs_fun = make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type)
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
        
        # Load hamiltonian_type from data (if exists) or use from config
        if "hamiltonian_type" in df_config:
            config["hamiltonian_type"] = df_config["hamiltonian_type"]
        
        # IMPORTANT FIX: For general_local_zz, we need to extract the true parameters properly
        hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
        
        if hamiltonian_type == "general_local_zz" and theta_true_array is not None:
            L = config["L"]
            # theta_true_array should be in the format: [hx_0, hx_1, ..., hz_0, hz_1, ..., Jzz_0, Jzz_1, ...]
            # Extract and store them in config for reference
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
        print(f"   -> Hamiltonian Type: {config.get('hamiltonian_type', 'uniform_xyz')}")
        print(f"   -> Using Solver dt: {config['dt']} (User defined)")
        
        t_grid_shots = df_counts.index.values.astype(np.float32)
        counts_shots = df_counts.values.astype(np.int32)
        
        return t_grid_shots, counts_shots, initial_state_vector, theta_true_array
        
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    


 
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################   

# ============================================================
# 4. TRAINING HELPERS
# ============================================================
def adam_init(params):
    '''Initialize adam parameters'''
    m = jtu.tree_map(jnp.zeros_like, params); v = jtu.tree_map(jnp.zeros_like, params)
    return {"step": 0, "m": m, "v": v}

def adam_update(params, grads, opt_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    '''Adam optimization step'''
    step = opt_state["step"] + 1
    m = jtu.tree_map(lambda m, g: beta1*m + (1-beta1)*g, opt_state["m"], grads)
    v = jtu.tree_map(lambda v, g: beta2*v + (1-beta2)*(g*g), opt_state["v"], grads)
    m_hat = jtu.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jtu.tree_map(lambda x: x / (1 - beta2**step), v)
    params_new = jtu.tree_map(lambda p, mh, vh: p - lr*mh/(jnp.sqrt(vh)+eps), params, m_hat, v_hat)
    return params_new, {"step": step, "m": m, "v": v}

def get_trainable_mask(params, train_theta, train_nn):
    # Create a mask PyTree that mirrors `params` structure. Each leaf is an array
    # of booleans with the same shape as the corresponding parameter leaf.
    #Also selects if the theta or nn parameters will be trained or not
    mask = {}
    # theta is an array
    mask["theta"] = jnp.ones_like(params["theta"], dtype=bool) if train_theta else jnp.zeros_like(params["theta"], dtype=bool)
    # nn is a pytree (list of layer dicts); create masks of the same shape
    mask_nn = jtu.tree_map(lambda p: jnp.ones_like(p, dtype=bool) if train_nn else jnp.zeros_like(p, dtype=bool), params["nn"]) 
    mask["nn"] = mask_nn
    return mask


def make_step_fn(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, lambda_reg, learning_rate):
    """
    Returns a jitted step function with signature:
      step_fn(params, opt_state, t_grid_shots, psi0, counts_shots, trainable_mask)
    which returns (params_new, opt_state_new, loss_val, aux)
    where aux is the auxiliary tuple returned by `nde_loss` (loss_nll, reg, psi_traj).
    """
    # Create a grad function once (it will be traced/compiled inside the jitted step)
    grad_fn = jax.value_and_grad(nde_loss, has_aux=True)

    @jax.jit
    def step_fn(params, opt_state, t_grid_shots, psi0, counts_shots, trainable_mask):
        '''Execute nn + optimizer step
        params: simulation parameters (theta + nn)
        opt_state: optimization state
        trainable_mask: jnp.tree structure that contains the parameters to train 
                        (with theta and nn activated or deactivated depending on the training phase)
        '''
        # Compute loss and gradients (evolves trajectory with current parameters and calculates gradient)
        (loss_val, aux), grads = grad_fn(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, lambda_reg, t_grid_shots, psi0, counts_shots)
        # Apply mask to gradients so only selected parameters are updated
        masked_grads = jtu.tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), grads, trainable_mask)
        # Update parameters with Adam step
        params_new, opt_state_new = adam_update(params, masked_grads, opt_state, lr=learning_rate)
        return params_new, opt_state_new, loss_val, aux

    return step_fn

def train_phase(params_init, N_epochs, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type, t_grid_shots, psi0, counts_shots, train_theta, train_nn, phase_name, step_fn):
    '''Execute a phase of the training
    Args:
    params_init: theta params + NN initial params
    N_epochs (of this particular phase)
    config (from the top of the script)
    OPS_XYZ: List of possible pauli strings in H
    NN_MAP_FUN: function to perform forward step (contains activation function)
    NN_MODEL_TYPE: time dependent or state dependent
    MODEL_TYPE: same as NN_MODEL_TYPE (redundant variable)
    hamiltonian type: (either uniform_xyz, or general_local_zz)
    t_grid_shots: timestamps where we sample shots
    psi0: initial state
    count_shots: number of shots per timestamp
    train_theta: true or false. to activate variational training of physical parameters
    train_nn: true or false: to train nn
    phase_name: for identification
    step_fn: function for step of NN. Contains activation function for NN and other stuff
    '''
    # Return early if no training epochs
    if N_epochs <= 0:
        return params_init, []

    # Determine if theta is trained or not
    # For white-box mode, θ can be trained if specified
    # For black-box mode, θ is never trained (always frozen)
    train_theta_current = train_theta and (MODEL_TYPE == "white")

    # Print phase information
    print(f"\n--- Phase: {phase_name} ({'θ TRAINED' if train_theta_current else 'θ Frozen'}, "
        f"{'φ TRAINED' if train_nn else 'φ Frozen'}) ---")

    # Initialize optimizer state and loss tracking
    params = params_init
    #optimizer_state
    opt_state = adam_init(params_init)
    losses = []

    # Create mask to selectively train only specified parameters
    trainable_mask = get_trainable_mask(params, train_theta_current, train_nn)
    
    # Prepare true params for logging if available
    # Note: For general_local_zz, we won't have individual true parameters in config
    # They come from theta_true_data loaded from metadata
    num_params = len(params["theta"])
    print(f"  Training {num_params} parameters")
    
    for epoch in range(1, N_epochs + 1):
        #call forward function
        params, opt_state, loss_val, (loss_nll, reg, psi_traj) = step_fn(params, opt_state, t_grid_shots, psi0, counts_shots, trainable_mask)
        loss_val, loss_nll, reg = jax.device_get((loss_val, loss_nll, reg))
        losses.append(float(loss_val))
        #Print information
        if epoch % config["print_every"] == 0 or epoch == N_epochs:
            print(f"[{phase_name} {epoch:03d}/{N_epochs}] total_loss = {loss_val:.4e} | nll = {loss_nll:.4e} | reg = {reg:.4e}")
            if MODEL_TYPE == "white" and num_params <= 20:  # Don't print too many params
                print(f"  theta_curr : {np.round(np.array(params['theta']), 3)}")
    return params, losses



#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


def plot_hamiltonian_histogram(theta_true, theta_init, theta_final, hamiltonian_type, L):
    """
    Plot histogram comparing true, initial, and learned Hamiltonian parameters.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare labels based on hamiltonian_type
    if hamiltonian_type == "uniform_xyz":
        labels = ["Jx", "Jy", "Jz", "hx", "hy", "hz"]
    elif hamiltonian_type == "general_local_zz":
        labels = [f"hx_{i}" for i in range(L)] + \
                 [f"hz_{i}" for i in range(L)] + \
                 [f"Jzz_{i}" for i in range(L-1)]
    
    n_params = len(theta_final)
    x = np.arange(n_params)
    width = 0.25
    
    # Plot bars
    if theta_true is not None:
        ax.bar(x - width, theta_true, width, label='True', alpha=0.8, color='green')
    ax.bar(x, theta_init, width, label='Initial', alpha=0.8, color='blue')
    ax.bar(x + width, theta_final, width, label='Learned', alpha=0.8, color='red')
    
    # Customize plot
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Value')
    ax.set_title(f'Hamiltonian Parameters Comparison ({hamiltonian_type})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:n_params], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (init, final) in enumerate(zip(theta_init, theta_final)):
        ax.text(i, max(init, final) + 0.05, f'{final:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


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
hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
NUM_COEFFICIENTS = get_theta_shape(L, hamiltonian_type)
print(f"Number of Hamiltonian parameters: {NUM_COEFFICIENTS}")

#Define NN characteristics
NN_MODEL_TYPE = config["NN_MODEL_TYPE"]; NN_MAP_FUN = mlp_forward #type of activation function
if NN_MODEL_TYPE == "time_dependent": 
    NN_INPUT_DIM = 1
    NN_OUTPUT_DIM = NUM_COEFFICIENTS  # One coefficient per Hamiltonian operator
elif NN_MODEL_TYPE == "state_dependent": 
    NN_INPUT_DIM = 2 * dim
    NN_OUTPUT_DIM = 2 * dim 
else: 
    raise ValueError(f"Unknown NN_MODEL_TYPE")

layer_sizes = [NN_INPUT_DIM] + config["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
key = random.PRNGKey(config["seed_init"]); key, k_nn, k_th = random.split(key, 3)
nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)

# Initialize theta based on hamiltonian_type
if hamiltonian_type == "uniform_xyz":
    theta_init_list = [
        config["Jx_init"], config["Jy_init"], config["Jz_init"],
        config["hx_init"], config["hy_init"], config["hz_init"]
    ]
elif hamiltonian_type == "general_local_zz":
    # Get initialization lists
    L = config["L"]
    hx_init_list = config.get("hx_list_init", [0.0] * L)
    hz_init_list = config.get("hz_list_init", [0.5] * L)
    Jzz_init_list = config.get("Jzz_list_init", [0.5] * (L-1))
    
    # Validate lengths
    if len(hx_init_list) != L:
        raise ValueError(f"hx_list_init must have length L={L}, got {len(hx_init_list)}")
    if len(hz_init_list) != L:
        raise ValueError(f"hz_list_init must have length L={L}, got {len(hz_init_list)}")
    if len(Jzz_init_list) != L-1:
        raise ValueError(f"Jzz_list_init must have length L-1={L-1}, got {len(Jzz_init_list)}")
    
    theta_init_list = list(hx_init_list) + list(hz_init_list) + list(Jzz_init_list)
else:
    raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

#Introduce variation to initial parameters to simulate worst starting point
theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
if config["INIT_PERTURB_SCALE"] > 0: 
    theta_init += config["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_COEFFICIENTS,))

params = {"theta": theta_init, "nn": nn_params}

# 3. TRAIN
#defines step function (does not execute it)
step_fn = make_step_fn(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], hamiltonian_type, config["lambda_reg"], config["learning_rate"])
N_total = config["N_epochs"]
#sepatate epochs in sections
P1 = int(N_total * config["PHASE1_SPLIT"]); P2 = int(N_total * config["PHASE2_SPLIT"]); P3 = N_total - P1 - P2

#Warm up (both VQE and NN)
params, l1 = train_phase(params, P1, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], hamiltonian_type, t_grid_shots, psi0, counts_shots, train_theta=config["learn_theta"], train_nn=True, phase_name="P1 Warm-up", step_fn=step_fn)
params, l2 = train_phase(params, P2, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], hamiltonian_type, t_grid_shots, psi0, counts_shots, train_theta=config["learn_theta"], train_nn=False, phase_name="P2 Distill", step_fn=step_fn)
params, l3 = train_phase(params, P3, config, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], hamiltonian_type, t_grid_shots, psi0, counts_shots, train_theta=False, train_nn=True, phase_name="P3 Refine", step_fn=step_fn)
losses = l1 + l2 + l3

# 4. DIAGNOSTICS
T_extrap = config["t_max"] * config["T_extrapolate_factor"]
t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, config["dt"])

rhs_fun_model = make_rhs_fun(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, config["MODEL_TYPE"], hamiltonian_type)

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

if hamiltonian_type == "uniform_xyz":
    labels = ["Jx","Jy","Jz","hx","hy","hz"]
    for i in range(6):
        print(f"{labels[i]:<10}: {theta_final[i]:.5f}")
        
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
    
    # True dynamics (no NN)
    rhs_fun_true = lambda t, psi, p: -1j * xyz_hamiltonian_from_theta(L, theta_true, OPS_XYZ, hamiltonian_type) @ psi
    psi_true = evolve_trajectory(psi0, t_grid_long, rhs_fun_true, {"theta": theta_true, "nn": None})
    obs_true = calc_obs(psi_true)
    
    fid_nde = 1 - np.array([fidelity(psi_true[k], psi_model[k]) for k in range(len(psi_true))])
    fid_van = 1 - np.array([fidelity(psi_true[k], psi_vanilla[k]) for k in range(len(psi_true))])
   
    # Plots with True comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {hamiltonian_type}")
    ax1.plot(t_grid_long, 1-fid_nde, 'r', label='NDE Fidelity')
    ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax1.legend(); ax1.set_title("Fidelity")
    ax2.loglog(t_grid_long, fid_nde, 'r', label='NDE Infidelity')
    ax2.loglog(t_grid_long, fid_van, 'b--', label='Vanilla Infidelity')
    ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax2.legend(); ax2.set_title("Infidelity")
    plt.tight_layout(); plt.show()


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



# 7. PLOT TRAINING LOSS
plt.figure(figsize=(5,4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()