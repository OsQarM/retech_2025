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