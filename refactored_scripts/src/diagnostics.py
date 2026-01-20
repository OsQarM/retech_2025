
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

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
import glob
import sys
import yaml

sys.path.append('./')

from model_building import make_observables, xyz_hamiltonian_from_theta
from mlp import make_rhs_lindblad, make_rhs_schrodinger
from time_evolution import evolve_lindblad, evolve_trajectory

def print_data_info(config, expected_shape):
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
    print(f"Expected parameter count: {expected_shape}")
    return

def print_training_info(config, num_coefficients):

    print(f"\n{'='*60}")
    print(f"MODEL configURATION")
    print(f"{'='*60}")
    print(f"  Dynamics: {'Lindblad (noisy)' if config["use_noisy_dynamics"] else 'Schrödinger (noiseless)'}")


    print(f"  Hamiltonian type: {config["hamiltonian_type"]}")
    print(f"  Hamiltonian parameters: {num_coefficients}")
    print(f"  Model type: {config['MODEL_TYPE']}")
    print(f"  Learn θ: {config['learn_theta']}")

    if config["use_noisy_dynamics"]:
        print(f"  Learn noise: {config['learn_noise_params']}")
        print(f"  Noise model: {config['noise_model']}")
        if config['noise_model'] == 'global':
            print(f"    → 2 noise parameters total")
        else:
            print(f"    → {2*config['L']} noise parameters (2 per qubit)")
    return

def print_linblad_info(L, T1_list, T2_list, noise_model):

    if noise_model == "global":
        print(f"Using Lindblad dynamics (global noise)")
    elif noise_model == "local":
        print(f"Using Lindblad dynamics (per-qubit noise)")

    print(f"  T1 per qubit: {[f'{t:.2f}' for t in T1_list]}")
    print(f"  T2 per qubit: {[f'{t:.2f}' for t in T2_list]}")
    # Calculate rates for each qubit
    gamma_deph_list = [1.0/T2_list[i] - 1.0/(2*T1_list[i]) for i in range(L)]
    gamma_damp_list = [1.0/T1_list[i] for i in range(L)]
    print(f"  γ_dephasing per qubit: {[f'{g:.4f}' for g in gamma_deph_list]}")
    print(f"  γ_damping per qubit: {[f'{g:.4f}' for g in gamma_damp_list]}")
    return

def print_hamiltonian_parameters(config, hamiltonian_type, theta_final, nn_l2_norm):
    # Print Hamiltonian parameters
    L = config['L']
    if hamiltonian_type == "uniform_xyz":
        labels = ["Jx", "Jy", "Jz", "hx", "hy", "hz"]
        print("\nLearned Hamiltonian Parameters:")
        for i, label in enumerate(labels):
            print(f"  {label:<10}: {theta_final[i]:.5f}")
    elif hamiltonian_type == "general_local_zz":
        hx_final = theta_final[:L]
        hz_final = theta_final[L:2*L]
        Jzz_final = theta_final[2*L:]
        print("\nLearned Hamiltonian Parameters:")
        print("  Local X fields (hx_i):")
        for i in range(L):
            print(f"    Qubit {i}: {hx_final[i]:.5f}")
        print("  Local Z fields (hz_i):")
        for i in range(L):
            print(f"    Qubit {i}: {hz_final[i]:.5f}")
        print("  ZZ couplings (Jzz_i):")
        for i in range(L-1):
            print(f"    Bond {i}-{i+1}: {Jzz_final[i]:.5f}")
    
    print(f"\nNN L2 Norm: {float(nn_l2_norm):.4e}")
    return

def print_noise_parameters(config, params):
    noise_rates_final = np.array(jax.device_get(params["noise_rates"]))
    print("\nLearned Noise Parameters:")
    if config["noise_model"] == "global":
        print(f"  Dephasing rate (global): {noise_rates_final[0]:.5f}")
        print(f"  Damping rate (global): {noise_rates_final[1]:.5f}")
    else:
        print("  Dephasing rates (per qubit):")
        for i in range(config['L']):
            print(f"    Qubit {i}: {noise_rates_final[i]:.5f}")
        print("  Damping rates (per qubit):")
        for i in range(config['L']):
            print(f"    Qubit {i}: {noise_rates_final[config['L']+i]:.5f}")


def relative_absolute_error(theta_true, theta_learned):
    """Calculate relative error"""
    error = np.sum(np.abs(np.array(theta_true) - np.array(theta_learned)))
    norm = np.sum(np.abs(np.array(theta_true)))
    return error / (norm + 1e-12)


def print_relative_error(theta_true_array, theta_final):
    theta_true = np.array(theta_true_array)
    rel_error = relative_absolute_error(theta_true, theta_final)
    print(f"\nTrue Hamiltonian Parameters: {np.round(theta_true, 4)}")
    print(f"Relative Error: {rel_error:.4f}")


def calculate_observables_pure(psi_traj, obs_dict):
    """Calculate observables from pure state trajectory"""
    results = {}
    for name, op in obs_dict.items():
        exp_val = jnp.array([jnp.vdot(psi, op @ psi).real for psi in psi_traj])
        results[name] = np.array(exp_val)
    return results

def calculate_observables_mixed(rho_traj, obs_dict):
    """Calculate observables from density matrix trajectory"""
    results = {}
    for name, op in obs_dict.items():
        exp_val = jnp.array([jnp.trace(rho @ op).real for rho in rho_traj])
        results[name] = np.array(exp_val)
    return results

def calculate_fidelity_pure(psi1, psi2):
    """Fidelity between pure states"""
    return np.abs(np.vdot(psi1, psi2))**2

def calculate_fidelity_mixed(rho1, rho2):
    """Fidelity between density matrices"""
    # Uhlmann fidelity: F = Tr(sqrt(sqrt(rho1) @ rho2 @ sqrt(rho1)))^2
    # Simplified for efficiency
    return 1 - 0.5 * np.sum(np.abs(np.linalg.eigvals(rho1 - rho2)))

def calculate_purity(rho_traj):
    """Calculate purity Tr(ρ²) for density matrix trajectory"""
    purities = []
    for rho in rho_traj:
        purity = np.trace(rho @ rho).real
        purities.append(purity)
    return np.array(purities)

def generate_diagnostic_trajectories(config, OPS_XYZ, NN_MAP_FUN, dephasing_ops, damping_ops, state0, t_grid_long, params, theta_true):

    use_noisy = config["use_noisy_dynamics"]
    L = config['L']
    hamiltonian_type = config['hamiltonian_type']

    # Create evolution functions based on dynamics type
    if use_noisy:
        # Noisy dynamics
        rhs_model = make_rhs_lindblad(L, OPS_XYZ, NN_MAP_FUN, config["NN_MODEL_TYPE"],
                                     config["MODEL_TYPE"], hamiltonian_type,
                                     dephasing_ops, damping_ops, config["noise_model"])
        traj_model = evolve_lindblad(state0, t_grid_long, rhs_model, params)
        
        # Vanilla (theta only, with noise if learned)
        params_vanilla = {"theta": params["theta"], 
                         "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
        if "noise_rates" in params:
            params_vanilla["noise_rates"] = params["noise_rates"]
        traj_vanilla = evolve_lindblad(state0, t_grid_long, rhs_model, params_vanilla)
        
        # True trajectory if available
        if theta_true is not None:
            params_true = {"theta": jnp.array(theta_true), 
                          "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
            if "noise_rates" in params:
                params_true["noise_rates"] = params["noise_rates"]
            traj_true = evolve_lindblad(state0, t_grid_long, rhs_model, params_true)
        else:
            traj_true = None
        
        # Calculate observables
        obs_dict = make_observables(L)
        obs_model = calculate_observables_mixed(traj_model, obs_dict)
        obs_vanilla = calculate_observables_mixed(traj_vanilla, obs_dict)
        obs_true = calculate_observables_mixed(traj_true, obs_dict) if traj_true is not None else None
        
    else:
        # Noiseless dynamics
        rhs_model = make_rhs_schrodinger(L, OPS_XYZ, NN_MAP_FUN, config["NN_MODEL_TYPE"],
                                        config["MODEL_TYPE"], hamiltonian_type)
        traj_model = evolve_trajectory(state0, t_grid_long, rhs_model, params)
        
        # Vanilla (theta only)
        params_vanilla = {"theta": params["theta"],
                         "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
        traj_vanilla = evolve_trajectory(state0, t_grid_long, rhs_model, params_vanilla)
        
        # True trajectory if available
        if theta_true is not None:
            H_true = xyz_hamiltonian_from_theta(L, jnp.array(theta_true), OPS_XYZ, hamiltonian_type)
            rhs_true = lambda t, psi, p: -1j * H_true @ psi
            traj_true = evolve_trajectory(state0, t_grid_long, rhs_true, {})
        else:
            traj_true = None
        
        # Calculate observables
        obs_dict = make_observables(L)
        obs_model = calculate_observables_pure(traj_model, obs_dict)
        obs_vanilla = calculate_observables_pure(traj_vanilla, obs_dict)
        obs_true = calculate_observables_pure(traj_true, obs_dict) if traj_true is not None else None
    
    return traj_model, traj_vanilla, traj_true, obs_true, obs_model, obs_vanilla


def extract_learned_mps_ham_probs(config, OPS_XYZ, NN_MAP_FUN, state0, t_grid, params):
    L = config['L']
    hamiltonian_type = config['hamiltonian_type']

    # Noiseless dynamics
    rhs_model = make_rhs_schrodinger(L, OPS_XYZ, NN_MAP_FUN, config["NN_MODEL_TYPE"],
                                    config["MODEL_TYPE"], hamiltonian_type)
    traj_model = evolve_trajectory(state0, t_grid, rhs_model, params)
    
    # Vanilla (theta only)
    params_vanilla = {"theta": params["theta"],
                        "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
    traj_vanilla = evolve_trajectory(state0, t_grid, rhs_model, params_vanilla)
    
    # Negative log-likelihood
    probs_model = jnp.abs(traj_model)**2
    probs_model = probs_model / probs_model.sum(axis=1, keepdims=True)
    probs_model = jnp.clip(probs_model, 1e-9, 1.0)
    probs_model = probs_model[-1, :]

    probs_vanilla = jnp.abs(traj_vanilla)**2
    probs_vanilla = probs_vanilla / probs_vanilla.sum(axis=1, keepdims=True)
    probs_vanilla = jnp.clip(probs_vanilla, 1e-9, 1.0)
    probs_vanilla = probs_vanilla[-1, :]

    return probs_model, probs_vanilla

