#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 18:05:09 2025

@author: marcin
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
    
    # TRUE HAMILTONIAN PARAMETERS (The Ground Truth)
    "Jx_true": 0.9, "Jy_true": 1.1, "Jz_true": 0.4,
    "hx_true": 0.3, "hy_true": 0.1, "hz_true": 0.2,
}

# ============================================================
# 1. CORE SIMULATION UTILITIES
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

def prepare_initial_state(L: int, kind: str) -> Array:
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        raise ValueError(f"Unknown initial state kind: {kind}.")
    return psi

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

def generate_dataset(config, OPS_XYZ):
    L = config["L"]; t_max = config["t_max"]; dt = config["dt"]; N_shots = config["N_shots_per_time"]; seed = config["seed_data"]
    
    theta_true = jnp.array([config["Jx_true"], config["Jy_true"], config["Jz_true"], 
                            config["hx_true"], config["hy_true"], config["hz_true"],], dtype=jnp.float32)

    psi0 = prepare_initial_state(L, config["initial_state_kind"])
    
    t_grid_fine = jnp.arange(0.0, t_max + dt/2, dt)
    t_grid_shots = jnp.linspace(0.0, t_max, config["N_time_shots"])
    
    def rhs_true_dynamics_only(t, psi, params):
        H_T = xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ)
        return -1j * (H_T @ psi)
    
    params_true = {"theta": theta_true, "nn": None}
    psi_traj_true_fine = evolve_trajectory(psi0, t_grid_fine, rhs_true_dynamics_only, params=params_true)
    
    idx_shots = np.searchsorted(np.array(t_grid_fine), np.array(t_grid_shots))
    psi_traj_true_shots = psi_traj_true_fine[idx_shots]
    
    dim = 2**L
    rng = np.random.default_rng(seed)
    counts_shots = np.zeros((config["N_time_shots"], dim), dtype=np.int32)
    for k in range(config["N_time_shots"]):
        psi_k = np.asarray(psi_traj_true_shots[k])
        probs = np.abs(psi_k)**2
        probs = probs / probs.sum()
        samples = rng.choice(dim, size=N_shots, p=probs)
        hist = np.bincount(samples, minlength=dim)
        counts_shots[k] = hist

    return t_grid_shots, psi0, theta_true, counts_shots

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

def save_data_to_files(t_grid, counts, L, T_max, R_shots, J_steps, theta_true, psi0, config_dict):
    # Unique filename
    filename_core = f"L{L}_T{T_max:.2f}_R{R_shots}_J{J_steps}"
    output_filename_counts = f'experimental_data_{filename_core}_counts.csv'
    output_filename_metadata = f'experimental_data_{filename_core}_metadata.npz'
    output_filename_config_df = f'experimental_data_{filename_core}_config_df.csv'

    # 1. Save Counts
    dim = 2**L; L_int = int(L)
    bitstring_labels = [format(i, f'0{L_int}b') for i in range(dim)]
    df_counts = pd.DataFrame(counts, index=t_grid, columns=bitstring_labels)
    df_counts.index.name = 'Time'
    df_counts.to_csv(output_filename_counts)
    
    # 2. Save Arrays (State, True Theta)
    np.savez(output_filename_metadata, 
             L=L, initial_state=psi0, theta_true=theta_true)
    
    # 3. Save FULL CONFIG (System params, True params) to CSV
    df_config = config_to_dataframe(copy.deepcopy(config_dict))
    df_config.to_csv(output_filename_config_df, index=False)
    
    return output_filename_counts, output_filename_metadata, output_filename_config_df

if __name__ == "__main__":
    print("--- Running Data Generation ---")
    OPS_XYZ = build_xyz_basis(CONFIG["L"])
    t_grid_shots, psi0, theta_true, counts_shots = generate_dataset(CONFIG, OPS_XYZ)
    
    f_counts, f_meta, f_conf = save_data_to_files(
        t_grid_shots, counts_shots, 
        CONFIG["L"], CONFIG["t_max"], CONFIG["N_shots_per_time"], CONFIG["N_time_shots"], 
        theta_true, psi0, CONFIG
    )
    print(f"\n✅ Data Saved.\n   Counts: {f_counts}\n   Config: {f_conf}")