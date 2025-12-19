#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

@author: marcin
"""

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import pandas as pd
import copy 
import sys
import yaml

sys.path.append('./')

from model_building import get_theta_shape, build_xyz_basis, get_theta_true_from_config, prepare_initial_state
from model_building import build_lindblad_operators_per_qubit, build_lindblad_operators_global
from model_building import schrodinger_rhs, lindblad_rhs
from time_evolution import evolve_trajectory
from data_saving import save_data_to_files

Array = jnp.ndarray


def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def print_run_info(config):
    '''Print relevant information about data generation'''
    print("="*60)
    print("--- Running Data Generation ---")
    print("="*60)
    print(f"Dynamics type: {config['dynamics_type']}")
    print(f"Hamiltonian type: {config['hamiltonian_type']}")
    print(f"System size: {config['L']}")

    if config['dynamics_type'] == 'lindblad':
        print(f"Noise model: {config['noise_model']}")

    # Print parameter information
    expected_shape = get_theta_shape(config['L'], config['hamiltonian_type'])
    print(f"Expected parameter count: {expected_shape}")
    return

def print_global_linblad_info(T1, T2):

    print(f"Using Lindblad dynamics (global noise)")
    print(f"  T1 = {T1:.2f}, T2 = {T2:.2f} (all qubits)")
    gamma_deph = 1.0/T2 - 1.0/(2*T1)
    gamma_damp = 1.0/T1
    print(f"  γ_dephasing = {gamma_deph:.4f}, γ_damping = {gamma_damp:.4f}")
    return

def print_local_linblad_info(L, T1_list, T2_list):

    print(f"Using Lindblad dynamics (per-qubit noise)")
    print(f"  T1 per qubit: {[f'{t:.2f}' for t in T1_list]}")
    print(f"  T2 per qubit: {[f'{t:.2f}' for t in T2_list]}")
    # Calculate rates for each qubit
    gamma_deph_list = [1.0/T2_list[i] - 1.0/(2*T1_list[i]) for i in range(L)]
    gamma_damp_list = [1.0/T1_list[i] for i in range(L)]
    print(f"  γ_dephasing per qubit: {[f'{g:.4f}' for g in gamma_deph_list]}")
    print(f"  γ_damping per qubit: {[f'{g:.4f}' for g in gamma_damp_list]}")
    return



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
            print_global_linblad_info(T1, T2)

        elif noise_model == "local":

            T1_list = config.get("T1_list", [10.0] * L)
            T2_list = config.get("T2_list", [5.0] * L)
            
            if len(T1_list) != L or len(T2_list) != L:
                raise ValueError(f"T1_list and T2_list must have length L={L}")
            
            jump_ops, jump_rates = build_lindblad_operators_per_qubit(L, T1_list, T2_list)

            print_local_linblad_info(L, T1_list, T2_list)

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


if __name__ == "__main__":

    #choose configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "./data_generation_configuration.yaml"

    #load configuration
    CONFIG = load_config(config_file)
    #Print useful information
    print_run_info(CONFIG)

    L = CONFIG["L"]
    hamiltonian_type = CONFIG['hamiltonian_type']

    # Build operators based on hamiltonian_type
    OPS_XYZ = build_xyz_basis(L, hamiltonian_type)

    t_grid_shots, state0, theta_true, counts_shots = generate_dataset(CONFIG, OPS_XYZ)

    print(f"\nTrue Hamiltonian parameters:")
    print(f"  {theta_true}")

    # Save
    save_data_to_files(
        t_grid_shots, counts_shots,
        CONFIG["L"], CONFIG["t_max"], CONFIG["N_shots_per_time"], 
        CONFIG["N_time_shots"], theta_true, state0, CONFIG
    )