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

sys.path.append('../src/')

from model_building import get_theta_shape, build_xyz_basis, get_theta_true_from_config, prepare_initial_state, define_dynamics
from time_evolution import evolve_trajectory
from data_saving import save_data_to_files
from diagnostics import print_data_info, print_linblad_info
from sampling import sample_bitstrings_from_trajectory

Array = jnp.ndarray


def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def generate_dataset(config, OPS_XYZ=None):
    """Generate dataset with optional per-qubit noise"""
    L = config["L"]
    t_max = config["t_max"]
    dt = config["dt"]
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
    rhs_fun, params_true, T1_list, T2_list = define_dynamics(config, theta_true, params_true)

    if dynamics_type == "lindblad":
        print_linblad_info(L, T1_list, T2_list, noise_model)
    
    # Evolve
    print("Calculating trajectory...")
    state_traj_fine = evolve_trajectory(state0, t_grid_fine, rhs_fun, params=params_true)
    print("Trajectory calculated.")

    #Sample  
    t_grid_shots, counts_shots = sample_bitstrings_from_trajectory(config, seed, L, state_traj_fine, t_grid_fine, t_grid_shots, dynamics_type)

    return t_grid_shots, state0, theta_true, counts_shots
    


if __name__ == "__main__":

    #choose configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config/data_generation_configuration.yaml"

    #load configuration
    CONFIG = load_config(config_file)
    #Print useful information
    
    L = CONFIG["L"]
    hamiltonian_type = CONFIG['hamiltonian_type']

    expected_shape = get_theta_shape(L, hamiltonian_type)

    print_data_info(CONFIG, expected_shape)

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