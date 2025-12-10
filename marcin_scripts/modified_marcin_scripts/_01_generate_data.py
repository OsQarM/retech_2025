#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 18:05:09 2025
Modified on Tue Dec 2 12:01 2025

@author: marcin
modified by: oscar
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
import copy 

Array = jnp.ndarray

# ============================================================
# 0. CONFIG BLOCK: Defines the True System and Data Collection
# ============================================================
CONFIG = {
    # SYSTEM DEFINITION & DATA SAMPLING PARAMETERS
    "L": 6,                      # Number of qubits (N spins).
    "t_max": 1.0,                # Total time evolution (in units of ħ).
    "dt": 0.01,                  # RK4 Integration time step.
    "N_time_shots": 101,         # Number of discrete measurement points (J).
    "N_shots_per_time": 200,   # Number of shots taken at each discrete measurement time (R).
    "initial_state_kind": "all_plus", 
    "seed_data": 1234,           
    
    # SYSTEM TYPE
    "system_type": "open",       # "closed" or "open"
    "noise_types": ["dephasing"],  # ["thermal", "dephasing"] or []
    "gamma_thermal": 0.05,       # Thermal relaxation rate
    "gamma_dephasing": 0.03,     # Dephasing rates
    
    # TRUE HAMILTONIAN PARAMETERS (Non-Homogeneous)
    # For L=6: 5 bonds for J, 6 sites for h
    "Jx_true": [0.9, 0.85, 0.8, 0.75, 0.7],    # L-1 values
    "Jy_true": [1.1, 1.05, 1.0, 0.95, 0.9],    # L-1 values
    "Jz_true": [0.4, 0.35, 0.3, 0.25, 0.2],    # L-1 values
    "hx_true": [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],  # L values
    "hy_true": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],  # L values
    "hz_true": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],  # L values
    
    # NOISE PARAMETERS (Non-Homogeneous)
    "gamma_thermal_true": [0.05] * 6,    # L values
    "gamma_dephasing_true": [0.03] * 6,  # L values
}

# ============================================================
# 1. CORE SIMULATION UTILITIES
# ============================================================

def paulis(dtype=jnp.complex64):
    '''
    Returns Pauli matrices and identity
    '''
    sx = jnp.array([[0., 1.],[1., 0.]], dtype=dtype)
    sy = jnp.array([[0., -1j],[1j, 0.]], dtype=dtype)
    sz = jnp.array([[1., 0.],[0., -1.]], dtype=dtype)
    id2 = jnp.eye(2, dtype=dtype)
    return sx, sy, sz, id2

def kron_n(ops):
    '''
    Makes tensor product of a list of matrices
    '''
    out = ops[0]
    for A in ops[1:]: out = jnp.kron(out, A)
    return out

def build_xyz_individual_ops(L: int, dtype=jnp.complex64):
    '''
    Build individual Pauli operators for each site and bond
    Returns list of operators in order:
    [X0X1, X1X2, ..., X_{L-2}X_{L-1},  # (L-1) terms
     Y0Y1, Y1Y2, ..., Y_{L-2}Y_{L-1},  # (L-1) terms
     Z0Z1, Z1Z2, ..., Z_{L-2}Z_{L-1},  # (L-1) terms
     X0, X1, ..., X_{L-1},             # L terms
     Y0, Y1, ..., Y_{L-1},             # L terms
     Z0, Z1, ..., Z_{L-1}]             # L terms
    Total: 6L - 3 operators
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
    Returns:
    - sigma_minus_list: [σ⁻ for each qubit]
    - sigma_plus_list: [σ⁺ for each qubit]
    - sigma_z_list: [σ^z for each qubit]
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
    Creates Hamiltonian from non-homogeneous weights (theta)
    theta: array of length 6L - 3
    '''
    return sum(theta[k] * OPS_INDIVIDUAL[k] for k in range(len(theta)))

def prepare_initial_state(L: int, kind: str) -> Array:
    '''
    Create initial state as a jnp array
    '''
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        raise ValueError(f"Unknown initial state kind: {kind}.")
    return psi

def vectorize_density_matrix(rho):
    '''Convert density matrix to vectorized form'''
    return rho.reshape(-1, 1)

def devectorize_density_matrix(rho_vec, dim):
    '''Convert vectorized density matrix back to matrix form'''
    return rho_vec.reshape(dim, dim)

def liouvillian_superoperator(H, jump_operators, gamma_list):
    '''
    Construct Lindblad superoperator L[ρ] = -i[H, ρ] + ∑_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    
    Returns: L_super (dim^2 × dim^2 matrix acting on vectorized ρ)
    '''
    dim = H.shape[0]
    I = jnp.eye(dim)
    
    # Hamiltonian part: -i(H ⊗ I - I ⊗ H^T)
    H_super = -1j * (jnp.kron(H, I) - jnp.kron(I, H.T))
    
    # Initialize superoperator
    L_super = H_super
    
    # Add dissipative terms for each jump operator
    for L_k, gamma in zip(jump_operators, gamma_list):
        if gamma > 0:
            Lk_dag = L_k.conj().T
            # L_k ρ L_k† term
            term1 = gamma * jnp.kron(L_k, L_k.conj())
            # -½{L_k†L_k, ρ} term
            Lk_dag_Lk = Lk_dag @ L_k
            term2 = -0.5 * gamma * (jnp.kron(Lk_dag_Lk, I) + jnp.kron(I, Lk_dag_Lk.conj()))
            L_super += term1 + term2
    
    return L_super

def lindblad_rhs(t, rho_vec, params):
    '''
    Right-hand side for Lindblad equation: dρ/dt = L[ρ]
    rho_vec is vectorized density matrix
    '''
    H = params["H"]
    jump_ops = params.get("jump_operators", [])
    gamma_list = params.get("gamma_list", [])
    
    if jump_ops and gamma_list:
        # Open system: Lindblad dynamics
        L_super = liouvillian_superoperator(H, jump_ops, gamma_list)
        return L_super @ rho_vec
    else:
        # Closed system: von Neumann equation
        dim = int(jnp.sqrt(rho_vec.shape[0]))
        rho = devectorize_density_matrix(rho_vec, dim)
        drho_dt = -1j * (H @ rho - rho @ H)
        return vectorize_density_matrix(drho_dt)

def rk4_step_density_matrix(rho_vec, t, dt, rhs_fun, params):
    '''
    RK4 step for density matrix evolution (vectorized form)
    '''
    dt_c = jnp.asarray(dt, dtype=rho_vec.dtype)
    k1 = rhs_fun(t, rho_vec, params)
    k2 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, rho_vec + dt_c*k3, params)
    rho_next_vec = rho_vec + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Ensure trace preservation (renormalize if needed)
    dim = int(jnp.sqrt(rho_next_vec.shape[0]))
    rho_next = devectorize_density_matrix(rho_next_vec, dim)
    trace = jnp.trace(rho_next)
    if jnp.abs(trace - 1.0) > 1e-8:
        rho_next = rho_next / trace
        rho_next_vec = vectorize_density_matrix(rho_next)
    
    return rho_next_vec

def evolve_trajectory_density_matrix(rho0_vec, t_grid, rhs_fun, params):
    '''
    Evolve density matrix using Lindblad/von Neumann equation
    '''
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    @jax.jit
    def scan_fn(rho_prev_vec, t_dt):
        t_prev, dt = t_dt
        rho_next_vec = rk4_step_density_matrix(rho_prev_vec, t_prev, dt, rhs_fun, params)
        return rho_next_vec, rho_next_vec
    
    _, rho_traj_scan = jax.lax.scan(scan_fn, rho0_vec, (t_prev_grid, dt_grid))
    return jnp.concatenate([rho0_vec[None, :], rho_traj_scan], axis=0)

def generate_dataset(config, OPS_INDIVIDUAL):
    '''
    Generate dataset with non-homogeneous Hamiltonian and optional open system dynamics
    '''
    L = config["L"]
    t_max = config["t_max"]
    dt = config["dt"]
    N_shots = config["N_shots_per_time"]
    seed = config["seed_data"]
    system_type = config["system_type"]
    
    # Flatten true parameters for non-homogeneous Hamiltonian
    theta_true = jnp.array(
        config["Jx_true"] + config["Jy_true"] + config["Jz_true"] +
        config["hx_true"] + config["hy_true"] + config["hz_true"], 
        dtype=jnp.float32
    )
    
    psi0 = prepare_initial_state(L, config["initial_state_kind"])
    
    # Time grids
    t_grid_fine = jnp.arange(0.0, t_max + dt/2, dt)
    t_grid_shots = jnp.linspace(0.0, t_max, config["N_time_shots"])
    
    # Build Hamiltonian
    H_true = xyz_hamiltonian_nonhomogeneous(theta_true, OPS_INDIVIDUAL)
    
    # Set up jump operators if open system
    jump_operators = []
    gamma_list = []
    
    if system_type == "open":
        sigma_minus_list, sigma_plus_list, sigma_z_list = build_noise_operators(L)
        
        # Thermal noise (amplitude damping)
        if "thermal" in config["noise_types"]:
            jump_operators.extend(sigma_minus_list)  # Relaxation
            gamma_list.extend(config["gamma_thermal_true"])
        
        # Dephasing noise
        if "dephasing" in config["noise_types"]:
            jump_operators.extend(sigma_z_list)  # Dephasing
            gamma_list.extend(config["gamma_dephasing_true"])
    
    # Prepare initial density matrix
    rho0 = jnp.outer(psi0, psi0.conj())
    rho0_vec = vectorize_density_matrix(rho0)
    
    # Parameters for evolution
    params_true = {
        "H": H_true,
        "jump_operators": jump_operators,
        "gamma_list": gamma_list,
        "theta": theta_true
    }
    
    # Evolve trajectory
    rho_traj_vec = evolve_trajectory_density_matrix(rho0_vec, t_grid_fine, lindblad_rhs, params_true)
    
    # Select states at shot times
    idx_shots = np.searchsorted(np.array(t_grid_fine), np.array(t_grid_shots))
    rho_traj_shots_vec = rho_traj_vec[idx_shots]
    
    # Generate shot data
    dim = 2**L
    rng = np.random.default_rng(seed)
    counts_shots = np.zeros((config["N_time_shots"], dim), dtype=np.int32)
    
    for k in range(config["N_time_shots"]):
        # Convert vectorized density matrix back to matrix
        rho_vec = rho_traj_shots_vec[k]
        rho = devectorize_density_matrix(rho_vec, dim)
        
        # Diagonal elements = probabilities
        probs = jnp.diag(rho).real
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Sample from probability distribution
        samples = rng.choice(dim, size=N_shots, p=probs)
        hist = np.bincount(samples, minlength=dim)
        counts_shots[k] = hist

    return t_grid_shots, psi0, theta_true, counts_shots, jump_operators, gamma_list

# ============================================================
# 2. DATA I/O (PANDAS/NPZ)
# ============================================================

def config_to_dataframe_nonhomogeneous(config_dict):
    '''
    Convert config dict to DataFrame, handling lists for non-homogeneous parameters
    '''
    data = {}
    for key, value in config_dict.items():
        if isinstance(value, list):
            # Store lists as strings or expand into multiple columns
            if len(value) <= 10:  # For small lists, store as string
                data[key] = str(value)
            else:
                # For large lists, store summary statistics
                data[f"{key}_mean"] = float(np.mean(value))
                data[f"{key}_std"] = float(np.std(value))
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            data[key] = str(np.array(value).tolist())
        elif isinstance(value, (float, np.float32, np.float64)):
            data[key] = float(value)
        elif isinstance(value, int):
            data[key] = int(value)
        else:
            data[key] = str(value)
    return pd.DataFrame([data])

def save_data_to_files_nonhomogeneous(t_grid, counts, L, T_max, R_shots, J_steps, 
                                     theta_true, psi0, config_dict, jump_ops=None, gamma_list=None):
    '''
    Save data with support for non-homogeneous parameters
    '''
    filename_core = f"L{L}_T{T_max:.2f}_R{R_shots}_J{J_steps}_nonhomogeneous"
    
    if config_dict.get("system_type", "closed") == "open":
        noise_str = "_".join(config_dict.get("noise_types", []))
        filename_core += f"_{noise_str}"
    
    output_filename_counts = f'experimental_data_{filename_core}_counts.csv'
    output_filename_metadata = f'experimental_data_{filename_core}_metadata.npz'
    output_filename_config_df = f'experimental_data_{filename_core}_config_df.csv'

    # 1. Save Counts
    dim = 2**L
    L_int = int(L)
    bitstring_labels = [format(i, f'0{L_int}b') for i in range(dim)]
    df_counts = pd.DataFrame(counts, index=t_grid, columns=bitstring_labels)
    df_counts.index.name = 'Time'
    df_counts.to_csv(output_filename_counts)
    
    # 2. Save Arrays
    save_dict = {
        'L': L,
        'initial_state': np.array(psi0),
        'theta_true': np.array(theta_true),
        'system_type': config_dict.get("system_type", "closed"),
        'noise_types': config_dict.get("noise_types", [])
    }
    
    if jump_ops is not None and gamma_list is not None:
        # Save jump operators as list of arrays
        jump_ops_array = [np.array(op) for op in jump_ops]
        save_dict['jump_operators'] = np.array(jump_ops_array, dtype=object)
        save_dict['gamma_list'] = np.array(gamma_list)
    
    np.savez(output_filename_metadata, **save_dict)
    
    # 3. Save FULL CONFIG
    df_config = config_to_dataframe_nonhomogeneous(copy.deepcopy(config_dict))
    df_config.to_csv(output_filename_config_df, index=False)
    
    return output_filename_counts, output_filename_metadata, output_filename_config_df

# ============================================================
# 3. PARAMETER UTILITIES
# ============================================================

def get_parameter_labels(L):
    '''
    Generate labels for non-homogeneous parameters
    '''
    labels = []
    # Coupling terms
    for pauli in ['Jx', 'Jy', 'Jz']:
        for i in range(L - 1):
            labels.append(f"{pauli}_{i}{i+1}")
    # Field terms
    for pauli in ['hx', 'hy', 'hz']:
        for i in range(L):
            labels.append(f"{pauli}_{i}")
    return labels

def flatten_parameters(config, param_type="true"):
    '''
    Flatten non-homogeneous parameters from config dict
    param_type: "true", "init", or "gamma"
    '''
    L = config["L"]
    
    if param_type == "true":
        return (
            config["Jx_true"] + config["Jy_true"] + config["Jz_true"] +
            config["hx_true"] + config["hy_true"] + config["hz_true"]
        )
    elif param_type == "init":
        return (
            config["Jx_init"] + config["Jy_init"] + config["Jz_init"] +
            config["hx_init"] + config["hy_init"] + config["hz_init"]
        )
    elif param_type == "gamma":
        return config["gamma_thermal_true"] + config["gamma_dephasing_true"]
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("--- Running Data Generation (Non-Homogeneous + Open System) ---")
    
    # Initialize non-homogeneous parameters if not defined
    L = CONFIG["L"]
    if "Jx_init" not in CONFIG:
        CONFIG["Jx_init"] = [0.5] * (L - 1)
        CONFIG["Jy_init"] = [0.5] * (L - 1)
        CONFIG["Jz_init"] = [0.5] * (L - 1)
        CONFIG["hx_init"] = [0.0] * L
        CONFIG["hy_init"] = [0.0] * L
        CONFIG["hz_init"] = [0.5] * L
    
    # Build individual operators
    OPS_INDIVIDUAL = build_xyz_individual_ops(CONFIG["L"])
    
    # Generate dataset
    t_grid_shots, psi0, theta_true, counts_shots, jump_ops, gamma_list = generate_dataset(CONFIG, OPS_INDIVIDUAL)
    
    # Save data
    f_counts, f_meta, f_conf = save_data_to_files_nonhomogeneous(
        t_grid_shots, counts_shots, 
        CONFIG["L"], CONFIG["t_max"], CONFIG["N_shots_per_time"], CONFIG["N_time_shots"], 
        theta_true, psi0, CONFIG, jump_ops, gamma_list
    )
    
    print(f"\n✅ Data Saved.")
    print(f"   System: {CONFIG['system_type']}")
    if CONFIG["system_type"] == "open":
        print(f"   Noise types: {CONFIG['noise_types']}")
    print(f"   Counts: {f_counts}")
    print(f"   Config: {f_conf}")
    
    # Print parameter labels and values
    labels = get_parameter_labels(L)
    print(f"\nTrue Parameters ({len(labels)} total):")
    for i, (label, value) in enumerate(zip(labels, theta_true)):
        print(f"  {label}: {value:.3f}")