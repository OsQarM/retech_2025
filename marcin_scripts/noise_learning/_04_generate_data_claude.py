#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Generation Script with Per-Qubit Noise Support
Supports both global and per-qubit noise rates
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import pandas as pd
import copy 

Array = jnp.ndarray

# ============================================================
# CONFIG BLOCK
# ============================================================
CONFIG = {
    # SYSTEM PARAMETERS
    "L": 4,
    "t_max": 1.0,
    "dt": 0.01,
    "N_time_shots": 101,
    "N_shots_per_time": 5000,
    "initial_state_kind": "all_plus",
    "seed_data": 1234,
    
    # HAMILTONIAN TYPE
    "hamiltonian_type": "uniform_xyz",  # "uniform_xyz" or "general_local_zz"
    
    # TRUE HAMILTONIAN PARAMETERS
    # For "uniform_xyz":
    "Jx_true": 0.9, "Jy_true": 1.1, "Jz_true": 0.4,
    "hx_true": 0.3, "hy_true": 0.1, "hz_true": 0.2,
    
    # For "general_local_zz":
    "hx_list_true": [0.3, 0.8, 0.4, 0.9],
    "hz_list_true": [0.65, 0.2, 0.15, 0.43],
    "Jzz_list_true": [0.4, 0.25, 0.1],
    
    # DYNAMICS TYPE
    "dynamics_type": "lindblad",  # "schrodinger" or "lindblad"
    
    # NOISE MODEL
    "noise_model": "local",  # "global" (same for all qubits) or "local" (per-qubit)
    
    # NOISE PARAMETERS
    # For "global" noise model:
    "T1_global": 10.0,
    "T2_global": 5.0,
    
    # For "local" noise model (per-qubit T1 and T2):
    "T1_list": [10.0, 8.0, 12.0, 9.0],   # T1 for each qubit
    "T2_list": [5.0, 4.0, 6.0, 4.5],     # T2 for each qubit
}

# ============================================================
# CORE UTILITIES
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
    if hamiltonian_type == "uniform_xyz":
        return 6
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)
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
        for i in range(L):
            ops = [id2] * L
            ops[i] = sx
            ops_out.append(kron_n(ops))
        for i in range(L):
            ops = [id2] * L
            ops[i] = sz
            ops_out.append(kron_n(ops))
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = sz
            ops[i+1] = sz
            ops_out.append(kron_n(ops))
        return ops_out
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def get_theta_true_from_config(config: dict) -> Array:
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    
    if hamiltonian_type == "uniform_xyz":
        theta_true = jnp.array([
            config["Jx_true"], config["Jy_true"], config["Jz_true"],
            config["hx_true"], config["hy_true"], config["hz_true"]
        ], dtype=jnp.float32)
    
    elif hamiltonian_type == "general_local_zz":
        L = config["L"]
        hx_list = config.get("hx_list_true", [0.0] * L)
        hz_list = config.get("hz_list_true", [0.0] * L)
        Jzz_list = config.get("Jzz_list_true", [0.0] * (L-1))
        
        if len(hx_list) != L or len(hz_list) != L or len(Jzz_list) != L-1:
            raise ValueError("Parameter list lengths don't match L")
        
        theta_true = jnp.array(
            list(hx_list) + list(hz_list) + list(Jzz_list),
            dtype=jnp.float32
        )
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
    return theta_true

def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list, 
                               hamiltonian_type: str = "uniform_xyz") -> Array:
    expected_shape = get_theta_shape(L, hamiltonian_type)
    
    if len(theta) != expected_shape or len(OPS_XYZ) != expected_shape:
        raise ValueError(f"Parameter/operator count mismatch")
    
    H = jnp.zeros((2**L, 2**L), dtype=jnp.complex64)
    for i in range(expected_shape):
        H += theta[i] * OPS_XYZ[i]
    
    return H

def build_lindblad_operators_per_qubit(L: int, T1_list: list, T2_list: list, 
                                       dtype=jnp.complex64):
    """
    Build Lindblad operators with per-qubit noise rates.
    
    Args:
        L: Number of qubits
        T1_list: List of T1 times (length L)
        T2_list: List of T2 times (length L)
    
    Returns:
        operators: List of jump operators
        rates: List of corresponding rates
    """
    if len(T1_list) != L or len(T2_list) != L:
        raise ValueError(f"T1_list and T2_list must have length L={L}")
    
    sx, sy, sz, id2 = paulis(dtype)
    sigma_minus = (sx - 1j * sy) / 2.0
    
    operators = []
    rates = []
    
    for i in range(L):
        T1 = T1_list[i]
        T2 = T2_list[i]
        
        # Decay rate for qubit i
        gamma_decay = 1.0 / T1 if T1 > 0 else 0.0
        
        # Dephasing rate for qubit i
        gamma_dephase = max(1.0 / T2 - 1.0 / (2.0 * T1), 0.0) if T1 > 0 and T2 > 0 else 0.0
        
        # Decay operator
        if gamma_decay > 0:
            ops_decay = [id2] * L
            ops_decay[i] = sigma_minus
            operators.append(kron_n(ops_decay))
            rates.append(gamma_decay)
        
        # Dephasing operator
        if gamma_dephase > 0:
            ops_dephase = [id2] * L
            ops_dephase[i] = sz / jnp.sqrt(2.0)
            operators.append(kron_n(ops_dephase))
            rates.append(gamma_dephase)
    
    return operators, rates

def build_lindblad_operators_global(L: int, T1: float, T2: float, dtype=jnp.complex64):
    """
    Build Lindblad operators with global (uniform) noise rates.
    
    Args:
        L: Number of qubits
        T1: Global T1 time
        T2: Global T2 time
    
    Returns:
        operators: List of jump operators
        rates: List of corresponding rates
    """
    sx, sy, sz, id2 = paulis(dtype)
    sigma_minus = (sx - 1j * sy) / 2.0
    
    operators = []
    rates = []
    
    decay_rate = 1.0 / T1 if T1 > 0 else 0.0
    dephasing_rate = max(1.0 / T2 - 1.0 / (2.0 * T1), 0.0) if T1 > 0 and T2 > 0 else 0.0
    
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
            ops_dephase[i] = sz / jnp.sqrt(2.0)
            operators.append(kron_n(ops_dephase))
            rates.append(dephasing_rate)
    
    return operators, rates

def prepare_initial_state(L: int, kind: str, as_density_matrix=False) -> Array:
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        raise ValueError(f"Unknown initial state kind: {kind}")
    
    if as_density_matrix:
        psi = psi.reshape(-1, 1)
        rho = psi @ psi.conj().T
        return rho
    else:
        return psi

def rk4_step(state, t, dt, rhs_fun, params):
    dt_c = jnp.asarray(dt, dtype=state.dtype)
    k1 = rhs_fun(t, state, params)
    k2 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, state + dt_c*k3, params)
    state_next = state + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    if state.ndim == 1:  # State vector
        norm = jnp.linalg.norm(state_next)
        return state_next / (norm + 1e-12)
    else:  # Density matrix
        state_next = 0.5 * (state_next + state_next.conj().T)
        trace = jnp.trace(state_next).real
        return state_next / (trace + 1e-12)

def evolve_trajectory(state0, t_grid, rhs_fun, params):
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    def scan_fn(state_prev, t_dt):
        t_prev, dt = t_dt
        state_next = rk4_step(state_prev, t_prev, dt, rhs_fun, params)
        return state_next, state_next
    
    scan_fn_jitted = jax.jit(scan_fn)
    _, state_traj_scan = jax.lax.scan(scan_fn_jitted, state0, (t_prev_grid, dt_grid))
    return jnp.concatenate([state0[None, ...], state_traj_scan], axis=0)

def schrodinger_rhs(t, psi, params):
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    return -1j * (H @ psi)

def lindblad_rhs(t, rho, params):
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    
    # Hamiltonian evolution
    drho = -1j * (H @ rho - rho @ H)
    
    # Lindblad dissipators
    jump_ops = params["jump_operators"]
    rates = params["jump_rates"]
    
    for L_op, gamma in zip(jump_ops, rates):
        if gamma > 0:
            L_dag = L_op.conj().T
            L_dag_L = L_dag @ L_op
            drho += gamma * (L_op @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))
    
    return drho

def generate_dataset(config, OPS_XYZ=None):
    """Generate dataset with optional per-qubit noise"""
    L = config["L"]
    t_max = config["t_max"]
    dt = config["dt"]
    N_shots = config["N_shots_per_time"]
    seed = config["seed_data"]
    dynamics_type = config.get("dynamics_type", "schrodinger")
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    noise_model = config.get("noise_model", "global")
    
    # Get true Hamiltonian parameters
    theta_true = get_theta_true_from_config(config)
    
    # Build operators
    if OPS_XYZ is None:
        OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
    
    # Validate parameter count
    expected_shape = get_theta_shape(L, hamiltonian_type)
    if len(theta_true) != expected_shape or len(OPS_XYZ) != expected_shape:
        raise ValueError("Parameter/operator count mismatch")
    
    # Prepare initial state
    use_density_matrix = (dynamics_type == "lindblad")
    state0 = prepare_initial_state(L, config["initial_state_kind"], 
                                   as_density_matrix=use_density_matrix)
    
    # Time grids
    t_grid_fine = jnp.arange(0.0, t_max + dt/2, dt)
    t_grid_shots = jnp.linspace(0.0, t_max, config["N_time_shots"])
    
    # Setup parameters
    params_true = {
        "L": L,
        "theta": theta_true,
        "ops_xyz": OPS_XYZ,
        "hamiltonian_type": hamiltonian_type,
    }
    
    # Choose dynamics
    if dynamics_type == "schrodinger":
        rhs_fun = schrodinger_rhs
        print(f"Using Schrödinger dynamics (noiseless)")
        print(f"Hamiltonian: {hamiltonian_type} ({len(theta_true)} params)")
        
    elif dynamics_type == "lindblad":
        # Build Lindblad operators based on noise model
        if noise_model == "global":
            T1 = config.get("T1_global", 10.0)
            T2 = config.get("T2_global", 5.0)
            jump_ops, jump_rates = build_lindblad_operators_global(L, T1, T2)
            print(f"Using Lindblad dynamics (global noise)")
            print(f"  T1 = {T1:.2f}, T2 = {T2:.2f} (all qubits)")
            gamma_deph = 1.0/T2 - 1.0/(2*T1)
            gamma_damp = 1.0/T1
            print(f"  γ_dephasing = {gamma_deph:.4f}, γ_damping = {gamma_damp:.4f}")
            
        elif noise_model == "local":
            T1_list = config.get("T1_list", [10.0] * L)
            T2_list = config.get("T2_list", [5.0] * L)
            
            if len(T1_list) != L or len(T2_list) != L:
                raise ValueError(f"T1_list and T2_list must have length L={L}")
            
            jump_ops, jump_rates = build_lindblad_operators_per_qubit(L, T1_list, T2_list)
            print(f"Using Lindblad dynamics (per-qubit noise)")
            print(f"  T1 per qubit: {[f'{t:.2f}' for t in T1_list]}")
            print(f"  T2 per qubit: {[f'{t:.2f}' for t in T2_list]}")
            
            # Calculate rates for each qubit
            gamma_deph_list = [1.0/T2_list[i] - 1.0/(2*T1_list[i]) for i in range(L)]
            gamma_damp_list = [1.0/T1_list[i] for i in range(L)]
            print(f"  γ_dephasing per qubit: {[f'{g:.4f}' for g in gamma_deph_list]}")
            print(f"  γ_damping per qubit: {[f'{g:.4f}' for g in gamma_damp_list]}")
        else:
            raise ValueError(f"Unknown noise_model: {noise_model}")
        
        params_true["jump_operators"] = jump_ops
        params_true["jump_rates"] = jump_rates
        rhs_fun = lindblad_rhs
        print(f"Hamiltonian: {hamiltonian_type} ({len(theta_true)} params)")
        print(f"Jump operators: {len(jump_ops)}")
    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
    
    # Evolve
    print("Calculating trajectory...")
    state_traj_fine = evolve_trajectory(state0, t_grid_fine, rhs_fun, params=params_true)
    print("Trajectory calculated.")
    
    # Sample at measurement times
    idx_shots = np.searchsorted(np.array(t_grid_fine), np.array(t_grid_shots))
    state_traj_shots = state_traj_fine[idx_shots]
    
    dim = 2**L
    rng = np.random.default_rng(seed)
    counts_shots = np.zeros((config["N_time_shots"], dim), dtype=np.int32)
    
    print("Sampling measurement outcomes...")
    for k in range(config["N_time_shots"]):
        if dynamics_type == "schrodinger":
            psi_k = np.asarray(state_traj_shots[k])
            probs = np.abs(psi_k)**2
        else:  # lindblad
            rho_k = np.asarray(state_traj_shots[k])
            probs = np.real(np.diag(rho_k))
        
        # Normalize
        probs = np.maximum(probs, 0)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(dim) / dim
        
        samples = rng.choice(dim, size=N_shots, p=probs)
        hist = np.bincount(samples, minlength=dim)
        counts_shots[k] = hist
    
    return t_grid_shots, state0, theta_true, counts_shots

# ============================================================
# DATA I/O
# ============================================================
def config_to_dataframe(config_dict):
    data = {}
    for key, value in config_dict.items():
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            # Skip arrays (handled separately)
            continue
        elif isinstance(value, (list, tuple)):
            # Convert lists to strings for CSV storage
            data[key] = str(value)
        elif isinstance(value, float):
            data[key] = float(value)
        elif isinstance(value, int):
            data[key] = int(value)
        else:
            data[key] = value
    return pd.DataFrame([data])

def save_data_to_files(t_grid, counts, L, T_max, R_shots, J_steps, 
                       theta_true, state0, config_dict):
    dynamics_type = config_dict.get("dynamics_type", "schrodinger")
    noise_model = config_dict.get("noise_model", "global")
    
    if dynamics_type == "lindblad":
        if noise_model == "global":
            T1 = config_dict.get("T1_global", 10.0)
            T2 = config_dict.get("T2_global", 5.0)
            noise_info = f"_{noise_model}_T1{T1:.1f}_T2{T2:.1f}"
        else:  # local
            noise_info = f"_{noise_model}_noise"
    else:
        noise_info = ""
    
    filename_core = f"L{L}_T{T_max:.2f}_R{R_shots}_J{J_steps}_{dynamics_type}{noise_info}"
    output_filename_counts = f'experimental_data_{filename_core}_counts.csv'
    output_filename_metadata = f'experimental_data_{filename_core}_metadata.npz'
    output_filename_config_df = f'experimental_data_{filename_core}_config_df.csv'
    
    # Save counts
    dim = 2**L
    L_int = int(L)
    bitstring_labels = [format(i, f'0{L_int}b') for i in range(dim)]
    df_counts = pd.DataFrame(counts, index=t_grid, columns=bitstring_labels)
    df_counts.index.name = 'Time'
    df_counts.to_csv(output_filename_counts)
    
    # Convert density matrix to state vector if needed
    if state0.ndim == 2:
        w, v = np.linalg.eigh(state0)
        idx = np.argmax(w)
        state0_vector = v[:, idx]
    else:
        state0_vector = state0
    
    # Save metadata
    metadata_dict = {
        'L': L,
        'initial_state': state0_vector,
        'theta_true': theta_true
    }
    
    # Add noise parameters if applicable
    if dynamics_type == "lindblad":
        if noise_model == "global":
            T1 = config_dict.get("T1_global", 10.0)
            T2 = config_dict.get("T2_global", 5.0)
            metadata_dict['T1_global'] = T1
            metadata_dict['T2_global'] = T2
            metadata_dict['gamma_dephasing_true'] = 1.0/T2 - 1.0/(2*T1)
            metadata_dict['gamma_damping_true'] = 1.0/T1
        else:  # local
            T1_list = config_dict.get("T1_list", [10.0] * L)
            T2_list = config_dict.get("T2_list", [5.0] * L)
            metadata_dict['T1_list'] = np.array(T1_list)
            metadata_dict['T2_list'] = np.array(T2_list)
            gamma_deph_list = [1.0/T2_list[i] - 1.0/(2*T1_list[i]) for i in range(L)]
            gamma_damp_list = [1.0/T1_list[i] for i in range(L)]
            metadata_dict['gamma_dephasing_list_true'] = np.array(gamma_deph_list)
            metadata_dict['gamma_damping_list_true'] = np.array(gamma_damp_list)
    
    np.savez(output_filename_metadata, **metadata_dict)
    
    # Save config
    df_config = config_to_dataframe(copy.deepcopy(config_dict))
    df_config.to_csv(output_filename_config_df, index=False)
    
    print(f"\n✅ Data saved:")
    print(f"   Counts: {output_filename_counts}")
    print(f"   Metadata: {output_filename_metadata}")
    print(f"   Config: {output_filename_config_df}")
    
    return output_filename_counts, output_filename_metadata, output_filename_config_df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("DATA GENERATION")
    print("="*60)
    print(f"System size: L = {CONFIG['L']}")
    print(f"Dynamics: {CONFIG.get('dynamics_type', 'schrodinger')}")
    print(f"Hamiltonian: {CONFIG.get('hamiltonian_type', 'uniform_xyz')}")
    
    if CONFIG.get('dynamics_type') == 'lindblad':
        print(f"Noise model: {CONFIG.get('noise_model', 'global')}")
    
    L = CONFIG["L"]
    hamiltonian_type = CONFIG.get("hamiltonian_type", "uniform_xyz")
    
    # Build operators
    OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
    expected_shape = get_theta_shape(L, hamiltonian_type)
    print(f"Hamiltonian parameters: {expected_shape}")
    
    # Generate data
    t_grid_shots, state0, theta_true, counts_shots = generate_dataset(CONFIG, OPS_XYZ)
    
    print(f"\nTrue Hamiltonian parameters:")
    print(f"  {theta_true}")
    
    # Save
    save_data_to_files(
        t_grid_shots, counts_shots,
        CONFIG["L"], CONFIG["t_max"], CONFIG["N_shots_per_time"], 
        CONFIG["N_time_shots"], theta_true, state0, CONFIG
    )

