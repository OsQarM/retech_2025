#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

@author: marcin
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import copy 



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
        'theta_true': theta_true,
        'dynamics_type': dynamics_type,
        'noise_model': noise_model
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
            
            # Calculate rates
            gamma_deph_list = []
            gamma_damp_list = []
            for i in range(L):
                gamma_damp = 1.0/T1_list[i] if T1_list[i] > 0 else 0.0
                gamma_deph = max(1.0/T2_list[i] - 1.0/(2*T1_list[i]), 0.0) if T1_list[i] > 0 and T2_list[i] > 0 else 0.0
                gamma_deph_list.append(gamma_deph)
                gamma_damp_list.append(gamma_damp)
            
            metadata_dict['gamma_dephasing_list_true'] = np.array(gamma_deph_list)
            metadata_dict['gamma_damping_list_true'] = np.array(gamma_damp_list)
    
    np.savez(output_filename_metadata, **metadata_dict)
    
    # Save config - ensure noise parameters are included
    df_config = config_to_dataframe(copy.deepcopy(config_dict))
    
    # Explicitly add noise parameters to CSV for easier reading
    if dynamics_type == "lindblad":
        if noise_model == "global":
            df_config['T1_global'] = T1
            df_config['T2_global'] = T2
        else:
            df_config['T1_list'] = str(T1_list)
            df_config['T2_list'] = str(T2_list)
    
    df_config.to_csv(output_filename_config_df, index=False)
    
    print(f"\n✅ Data saved:")
    print(f"   Counts: {output_filename_counts}")
    print(f"   Metadata: {output_filename_metadata}")
    print(f"   Config: {output_filename_config_df}")
    
    # Print noise summary
    if dynamics_type == "lindblad":
        print(f"\n📊 Noise parameters saved:")
        if noise_model == "global":
            print(f"   Global: T1={T1:.2f}, T2={T2:.2f}")
        else:
            print(f"   Local - T1 per qubit: {T1_list}")
            print(f"   Local - T2 per qubit: {T2_list}")
    
    return output_filename_counts, output_filename_metadata, output_filename_config_df