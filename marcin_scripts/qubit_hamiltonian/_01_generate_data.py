#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 18:05:09 2025
Modified on Tue Dec 09 11:25 2025 by oscar

@author: marcin
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import pandas as pd
import copy 

Array = jnp.ndarray

# ============================================================
# 0. CONFIG BLOCK: Defines the True System and Data Collection
# ============================================================
CONFIG = {
    # SYSTEM DEFINITION & DATA SAMPLING PARAMETERS
    "L": 5,                      # Number of qubits (N spins).
    "t_max": 1.0,                # Total time evolution (in units of ħ).
    "dt": 0.01,                  # RK4 Integration time step.
    "N_time_shots": 101,         # Number of discrete measurement points (J).
    "N_shots_per_time": 500,     # Number of shots taken at each discrete measurement time (R).
    "initial_state_kind": "all_plus", 
    "seed_data": 1234,           
    
    # NEW: HAMILTONIAN PARAMETERIZATION OPTION
    "hamiltonian_type": "uniform_xyz",  # Options: "uniform_xyz" or "general_local_zz"
    
    # TRUE HAMILTONIAN PARAMETERS (Depends on hamiltonian_type)
    # For "uniform_xyz" (6 parameters for all qubits):
    "Jx_true": 0.9, "Jy_true": 1.1, "Jz_true": 0.4,
    "hx_true": 0.3, "hy_true": 0.1, "hz_true": 0.2,
    
    # For "general_local_zz" (2L + (L-1) parameters):
    # Local fields for each qubit: hx_i, hz_i
    # ZZ interactions for each bond: Jzz_i (for bond between qubit i and i+1)
    "hx_list_true": [0.3, 0.8, 0.4, 0.9, 0.2, 0.5],  # X field for each qubit
    "hz_list_true": [0.65, 0.2, 0.15, 0.23, 0.7, 0.5],   # Z field for each qubit
    "Jzz_list_true": [0.4, 0.25, 0.1, 0.9, 0.2],       # ZZ coupling for each bond (L-1 values)
    
    # NEW: CHOICE OF DYNAMICS AND NOISE PARAMETERS
    "dynamics_type": "schrodinger",  # Options: "schrodinger" or "lindblad"
    "T1": 10000.0,                   # T1 decay time (inverse of decay rate) - FIXED DEFAULT
    "T2": 10.0,                    # T2 dephasing time (inverse of dephasing rate) - FIXED DEFAULT
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

def get_theta_shape(L: int, hamiltonian_type: str) -> tuple:
    '''
    Returns the expected number of parameters for given L and hamiltonian_type
    '''
    if hamiltonian_type == "uniform_xyz":
        return 6  # Jx, Jy, Jz, hx, hy, hz
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)  # hx_i (L), hz_i (L), Jzz_i (L-1)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def get_theta_true_from_config(config: dict) -> Array:
    '''
    Extract true theta parameters from config based on hamiltonian_type
    '''
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    
    if hamiltonian_type == "uniform_xyz":
        # Original XYZ model
        theta_true = jnp.array([
            config["Jx_true"], config["Jy_true"], config["Jz_true"],
            config["hx_true"], config["hy_true"], config["hz_true"]
        ], dtype=jnp.float32)
    
    elif hamiltonian_type == "general_local_zz":
        # General local model with ZZ interactions
        L = config["L"]
        
        # Check parameter lengths
        hx_list = config.get("hx_list_true", [0.0] * L)
        hz_list = config.get("hz_list_true", [0.0] * L)
        Jzz_list = config.get("Jzz_list_true", [0.0] * (L-1))
        
        # Validate lengths
        if len(hx_list) != L:
            raise ValueError(f"hx_list_true must have length L={L}, got {len(hx_list)}")
        if len(hz_list) != L:
            raise ValueError(f"hz_list_true must have length L={L}, got {len(hz_list)}")
        if len(Jzz_list) != L-1:
            raise ValueError(f"Jzz_list_true must have length L-1={L-1}, got {len(Jzz_list)}")
        
        # Concatenate all parameters
        theta_true = jnp.array(
            list(hx_list) + list(hz_list) + list(Jzz_list),
            dtype=jnp.float32
        )
    
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
    return theta_true

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

def build_lindblad_operators(L: int, T1: float, T2: float, dtype=jnp.complex64):
    '''
    Build Lindblad jump operators for decay and dephasing.
    
    Standard relations:
    - Decay rate: γ = 1/T1
    - Dephasing rate: γ_ϕ = 1/T2 - 1/(2*T1)
    Must satisfy: γ_ϕ ≥ 0 → T2 ≤ 2*T1 for pure dephasing
    '''
    sx, sy, sz, id2 = paulis(dtype)
    # Sigma minus operator: σ- = (σx - iσy)/2
    sigma_minus = (sx - 1j * sy) / 2.0
    
    operators = []
    rates = []
    
    decay_rate = 1.0 / T1 if T1 > 0 else 0.0
    dephasing_rate = max(1.0 / T2 - 1.0 / (2.0 * T1), 0.0)  # Ensure non-negative
    
    for i in range(L):
        # Decay operator for qubit i
        if decay_rate > 0:
            ops_decay = [id2] * L
            ops_decay[i] = sigma_minus
            operators.append(kron_n(ops_decay))
            rates.append(decay_rate)
        
        # Dephasing operator for qubit i
        if dephasing_rate > 0:
            ops_dephase = [id2] * L
            ops_dephase[i] = sz / jnp.sqrt(2.0)  # Normalized: Tr(L†L) = 1
            operators.append(kron_n(ops_dephase))
            rates.append(dephasing_rate)
    
    return operators, rates

def prepare_initial_state(L: int, kind: str, as_density_matrix=False) -> Array:
    '''
    Create initial state as a jnp array. Either with state 0 or state +
    If as_density_matrix=True, returns density matrix instead of state vector
    '''
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        raise ValueError(f"Unknown initial state kind: {kind}.")
    
    if as_density_matrix:
        # Convert to density matrix: ρ = |ψ⟩⟨ψ|
        psi = psi.reshape(-1, 1)  # Column vector
        rho = psi @ psi.conj().T
        return rho
    else:
        return psi

def rk4_step(state, t, dt, rhs_fun, params):
    '''
    Apply Runge-Kutta 4 method to a given function and for a set of variational parameters
    Works for both state vectors and density matrices
    '''
    dt_c = jnp.asarray(dt, dtype=state.dtype)
    k1 = rhs_fun(t, state, params)
    k2 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c,     state + dt_c*k3, params)
    state_next = state + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Normalization depends on whether we're evolving a state vector or density matrix
    if state.ndim == 1:  # State vector
        norm = jnp.linalg.norm(state_next)
        # Safe division
        return state_next / (norm + 1e-12)
    else:  # Density matrix
        # Ensure Hermiticity (symmetrize) and preserve trace
        state_next = 0.5 * (state_next + state_next.conj().T)
        trace = jnp.trace(state_next).real
        return state_next / (trace + 1e-12)


def evolve_trajectory(state0, t_grid, rhs_fun, params):
    '''
    Performs full time evolution of initial state according to specific differential equation (rhs_fun),
    with given variational parameters, and for every time in a given list (t_grid)
    Returns list of all states in the trajectory
    '''
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    def scan_fn(state_prev, t_dt):
        t_prev, dt = t_dt
        state_next = rk4_step(state_prev, t_prev, dt, rhs_fun, params)
        return state_next, state_next
    
    # Compile the scan function
    scan_fn_jitted = jax.jit(scan_fn)
    
    # Run the scan
    _, state_traj_scan = jax.lax.scan(scan_fn_jitted, state0, (t_prev_grid, dt_grid))
    return jnp.concatenate([state0[None, ...], state_traj_scan], axis=0)

def schrodinger_rhs(t, psi, params):
    '''
    Right-hand side for Schrödinger equation: dψ/dt = -i H ψ
    '''
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    return -1j * (H @ psi)

def lindblad_rhs(t, rho, params):
    '''
    Lindblad master equation: dρ/dt = -i[H, ρ] + ∑ γ (L ρ L† - ½{L†L, ρ})
    '''
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    
    # Hamiltonian part: -i[H, ρ]
    drho = -1j * (H @ rho - rho @ H)
    
    # Lindblad dissipators
    jump_ops = params["jump_operators"]
    rates = params["jump_rates"]
    
    # Process all jump operators
    for L_op, gamma in zip(jump_ops, rates):
        # Skip zero-rate operators (though rates shouldn't be zero if included)
        if gamma > 0:
            L_dag = L_op.conj().T
            L_dag_L = L_dag @ L_op
            # Lindblad term: γ (L ρ L† - 0.5{L†L, ρ})
            drho += gamma * (L_op @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))
    
    return drho

def generate_dataset(config, OPS_XYZ=None):
    '''
    Whole algorithm to generate the dataset.
    First reads simulation parameters from configuration file
    Performs time evolution and calculates state trajectory
    Samples from a number of points in time
    Returns list of timestamps, initial state, variational parameters (H coefficients), and sampling result
    '''
    L = config["L"]
    t_max = config["t_max"]
    dt = config["dt"]
    N_shots = config["N_shots_per_time"]
    seed = config["seed_data"]
    dynamics_type = config.get("dynamics_type", "schrodinger")
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    T1 = config.get("T1", 10.0)
    T2 = config.get("T2", 5.0)
    
    # Get true parameters based on hamiltonian_type
    theta_true = get_theta_true_from_config(config)
    
    # Build operators if not provided
    if OPS_XYZ is None:
        OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
    
    # Validate parameter count
    expected_shape = get_theta_shape(L, hamiltonian_type)
    if len(theta_true) != expected_shape:
        raise ValueError(
            f"Parameter count mismatch: expected {expected_shape} parameters "
            f"for hamiltonian_type='{hamiltonian_type}', got {len(theta_true)}"
        )
    
    if len(OPS_XYZ) != expected_shape:
        raise ValueError(
            f"Operator count mismatch: expected {expected_shape} operators "
            f"for hamiltonian_type='{hamiltonian_type}', got {len(OPS_XYZ)}"
        )
    
    # Prepare initial state
    use_density_matrix = (dynamics_type == "lindblad")
    state0 = prepare_initial_state(L, config["initial_state_kind"], as_density_matrix=use_density_matrix)
    
    # Define time grids
    t_grid_fine = jnp.arange(0.0, t_max + dt/2, dt)
    t_grid_shots = jnp.linspace(0.0, t_max, config["N_time_shots"])
    
    # Set up parameters for dynamics
    params_true = {
        "L": L,
        "theta": theta_true,
        "ops_xyz": OPS_XYZ,
        "hamiltonian_type": hamiltonian_type,
    }
    
    # Choose dynamics type
    if dynamics_type == "schrodinger":
        rhs_fun = schrodinger_rhs
        print(f"Using Schrödinger dynamics (noiseless)")
        print(f"Hamiltonian type: {hamiltonian_type} with {len(theta_true)} parameters")
    elif dynamics_type == "lindblad":
        # Build Lindblad jump operators
        jump_ops, jump_rates = build_lindblad_operators(L, T1, T2)
        params_true["jump_operators"] = jump_ops
        params_true["jump_rates"] = jump_rates
        rhs_fun = lindblad_rhs
        print(f"Using Lindblad dynamics with T1={T1}, T2={T2}")
        print(f"Hamiltonian type: {hamiltonian_type} with {len(theta_true)} parameters")
        print(f"Number of jump operators: {len(jump_ops)}")
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
    
    # Calculate trajectory
    print("Calculating trajectory...")
    state_traj_fine = evolve_trajectory(state0, t_grid_fine, rhs_fun, params=params_true)
    print("Trajectory calculated.")
    
    # Select states at shot times
    idx_shots = np.searchsorted(np.array(t_grid_fine), np.array(t_grid_shots))
    state_traj_shots = state_traj_fine[idx_shots]
    
    dim = 2**L
    rng = np.random.default_rng(seed)
    counts_shots = np.zeros((config["N_time_shots"], dim), dtype=np.int32)
    
    # For each shot time, sample according to Born rule
    print("Sampling measurement outcomes...")
    for k in range(config["N_time_shots"]):
        if dynamics_type == "schrodinger":
            # Pure state: probabilities = |ψ|²
            psi_k = np.asarray(state_traj_shots[k])
            probs = np.abs(psi_k)**2
        else:  # lindblad
            # Mixed state: probabilities = diag(ρ)
            rho_k = np.asarray(state_traj_shots[k])
            probs = np.real(np.diag(rho_k))
        
        # Normalize probabilities (numerical safety)
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(dim) / dim  # Uniform if all zero
        
        # Sample from distribution
        samples = rng.choice(dim, size=N_shots, p=probs)
        hist = np.bincount(samples, minlength=dim)
        counts_shots[k] = hist

    return t_grid_shots, state0, theta_true, counts_shots

# ============================================================
# 2. DATA I/O (PANDAS/NPZ)
# ============================================================

def config_to_dataframe(config_dict):
    data = {}
    for key, value in config_dict.items():
        if isinstance(value, (jnp.ndarray, np.ndarray, float)):
             data[key] = float(value)
        elif isinstance(value, int):
             data[key] = int(value)
        else:
             data[key] = value 
    return pd.DataFrame([data])

def save_data_to_files(t_grid, counts, L, T_max, R_shots, J_steps, theta_true, state0, config_dict):
    # Unique filename with dynamics type
    dynamics_type = config_dict.get("dynamics_type", "schrodinger")
    if dynamics_type == "lindblad":
        T1 = config_dict.get("T1", 10.0)
        T2 = config_dict.get("T2", 5.0)
        noise_info = f"_T1{T1:.1f}_T2{T2:.1f}"
    else:
        noise_info = ""
    
    filename_core = f"L{L}_T{T_max:.2f}_R{R_shots}_J{J_steps}_{dynamics_type}{noise_info}"
    output_filename_counts = f'experimental_data_{filename_core}_counts.csv'
    output_filename_metadata = f'experimental_data_{filename_core}_metadata.npz'
    output_filename_config_df = f'experimental_data_{filename_core}_config_df.csv'

    # 1. Save Counts
    dim = 2**L; L_int = int(L)
    bitstring_labels = [format(i, f'0{L_int}b') for i in range(dim)]
    df_counts = pd.DataFrame(counts, index=t_grid, columns=bitstring_labels)
    df_counts.index.name = 'Time'
    df_counts.to_csv(output_filename_counts)


    # Convert density matrix to state vector if needed
    if state0.ndim == 2:  # It's a density matrix
        # Take principal eigenvector
        w, v = np.linalg.eigh(state0)
        idx = np.argmax(w)
        state0_vector = v[:, idx]
    else:
        state0_vector = state0
    
    # 2. Save Arrays (State, True Theta) - save the state vector, not density matrix
    np.savez(output_filename_metadata, 
             L=L, initial_state=state0_vector, theta_true=theta_true)
    
    # 3. Save FULL CONFIG (System params, True params) to CSV
    df_config = config_to_dataframe(copy.deepcopy(config_dict))
    df_config.to_csv(output_filename_config_df, index=False)
    
    return output_filename_counts, output_filename_metadata, output_filename_config_df

if __name__ == "__main__":
    print("--- Running Data Generation ---")
    print(f"Dynamics type: {CONFIG.get('dynamics_type', 'schrodinger')}")
    print(f"Hamiltonian type: {CONFIG.get('hamiltonian_type', 'uniform_xyz')}")
    
    L = CONFIG["L"]
    hamiltonian_type = CONFIG.get("hamiltonian_type", "uniform_xyz")
    
    # Build operators based on hamiltonian_type
    OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
    
    # Print parameter information
    expected_shape = get_theta_shape(L, hamiltonian_type)
    print(f"Expected parameter count: {expected_shape}")
    
    t_grid_shots, state0, theta_true, counts_shots = generate_dataset(CONFIG, OPS_XYZ)
    
    print(f"True parameters: {theta_true}")
    
    f_counts, f_meta, f_conf = save_data_to_files(
        t_grid_shots, counts_shots, 
        CONFIG["L"], CONFIG["t_max"], CONFIG["N_shots_per_time"], CONFIG["N_time_shots"], 
        theta_true, state0, CONFIG
    )
    print(f"\n✅ Data Saved.\n   Counts: {f_counts}\n   Config: {f_conf}")