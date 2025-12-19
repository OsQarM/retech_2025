
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

@author: marcin
"""

import sys

sys.path.append('./')


def print_run_info(config, expected_shape):
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