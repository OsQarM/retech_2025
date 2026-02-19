#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Hamiltonian Learning Script with Proper Noise Handling
Supports both pure Schrödinger and Lindblad dynamics
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

from diagnostics import calculate_fidelity_mixed, calculate_fidelity_pure, calculate_purity


def plot_hamiltonian_parameters(theta_true, theta_init, theta_final, hamiltonian_type, L, config):
    """Plot learned vs true Hamiltonian parameters"""
    if hamiltonian_type == "uniform_xyz":
        labels = ["Jx", "Jy", "Jz", "hx", "hy", "hz"]
    elif hamiltonian_type == "general_local_zz":
        labels = ([f"hx_{i}" for i in range(L)] + 
                 [f"hz_{i}" for i in range(L)] + 
                 [f"Jzz_{i}" for i in range(L-1)])
    else:
        labels = [f"θ_{i}" for i in range(len(theta_final))]
    
    n_params = len(theta_final)
    x = np.arange(n_params)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(10, n_params), 6))
    print(theta_true)
    
    if theta_true is not None and len(theta_true) == n_params:
        ax.bar(x - width, theta_true, width, label='True', alpha=0.8, color='green')
    ax.bar(x, theta_init, width, label='Initial', alpha=0.8, color='blue')
    ax.bar(x + width, theta_final, width, label='Learned', alpha=0.8, color='red')
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title(f'Hamiltonian Parameters ({hamiltonian_type}, L={L})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:n_params], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    N = config['L']
    h_type = config['hamiltonian_type']
    noise =  config['use_noisy_dynamics']
    noise_type = config['noise_model']

    filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"
    
    # Save parameters to .npy files
    np.save(f'../results/hamiltonian_true_{filename_core}.npy', theta_true)
    np.save(f'../results/hamiltonian_init_{filename_core}.npy', theta_init)
    np.save(f'../results/hamiltonian_learned_{filename_core}.npy', theta_final)

    filename = f'./hamiltonian_parameters_{filename_core}'
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)

    return fig


def plot_noise_parameters(learned_rates, true_rates, noise_model, L, config):
    """Plot learned vs true noise rates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    learned_rates = [abs(i) for i in learned_rates]
    true_rates = [abs(i) for i in true_rates]

    if noise_model == "global":
        # Single rate for all qubits
        labels = ['Global']
        x = [0]
        width = 0.3
        
        if true_rates is not None:
            ax1.bar([x[0] - width/2], [true_rates[0]], width, 
                   label='True', alpha=0.8, color='green')
            ax1.bar([x[0] + width/2], [learned_rates[0]], width,
                label='Learned', alpha=0.8, color='red')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_ylabel('Dephasing Rate γ_z')
            ax1.set_title('Dephasing Rates')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if true_rates is not None:
            ax2.bar([x[0] - width/2], [true_rates[1]], width,
                   label='True', alpha=0.8, color='green')
            ax2.bar([x[0] + width/2], [learned_rates[1]], width,
                label='Learned', alpha=0.8, color='red')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Damping Rate γ_m')
            ax2.set_title('Damping Rates')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
    else:  # local
        labels = [f'Q{i}' for i in range(L)]
        x = np.arange(L)
        width = 0.35
        
        if true_rates is not None and len(true_rates) >= 2*L:
            ax1.bar(x - width/2, true_rates[:L], width,
                   label='True', alpha=0.8, color='green')
            ax1.bar(x + width/2, learned_rates[:L], width,
                label='Learned', alpha=0.8, color='red')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_ylabel('Dephasing Rate γ_z')
            ax1.set_title('Dephasing Rates (Per Qubit)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if true_rates is not None and len(true_rates) >= 2*L:
            ax2.bar(x - width/2, true_rates[L:2*L], width,
                   label='True', alpha=0.8, color='green')
            ax2.bar(x + width/2, learned_rates[L:2*L], width,
                label='Learned', alpha=0.8, color='red')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Damping Rate γ_m')
            ax2.set_title('Damping Rates (Per Qubit)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    

    N = config['L']
    h_type = config['hamiltonian_type']
    noise =  config['use_noisy_dynamics']
    noise_type = config['noise_model']


    filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"

    # Save noise parameters
    np.save(f'../results/noise_true_{filename_core}.npy', true_rates)
    np.save(f'../results/noise_learned_{filename_core}.npy', learned_rates)

    filename = f'./noise_parameters_{filename_core}'
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)

    return fig


def plot_mixed_state_fidelity(traj_model, traj_vanilla, traj_true, config, t_grid_long, L):

       traj_model_np = np.array(jax.device_get(traj_model))
       traj_van_np = np.array(jax.device_get(traj_vanilla))
       if traj_true is not None:
           traj_true_np = np.array(jax.device_get(traj_true))
           fid_model = np.array([calculate_fidelity_mixed(traj_true_np[k], traj_model_np[k])
                                 for k in range(len(traj_model_np))])
           fid_van = np.array([calculate_fidelity_mixed(traj_true_np[k], traj_van_np[k])
                               for k in range(len(traj_van_np))])
           inf_model = 1.0 - fid_model
           inf_van = 1.0 - fid_van  
           fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
           fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {config['hamiltonian_type']} (mixed)")
           ax1.set_yscale('log')
           ax1.plot(t_grid_long, 1.0 - fid_model, 'r', label='NDE Infidelity')
           ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1)
           ax1.set_title('Infidelity (linear)'); ax1.legend()
           ax2.set_yscale('log')
           ax2.plot(t_grid_long + 1e-12, inf_model, 'r', label='NDE Infidelity')
           ax2.plot(t_grid_long + 1e-12, inf_van, 'b--', label='Vanilla Infidelity')
           ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1)
           ax2.set_title('Infidelity (log-log)'); ax2.legend()

           N = config['L']
           h_type = config['hamiltonian_type']
           noise =  config['use_noisy_dynamics']
           noise_type = config['noise_model']

           filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"

           # Save trajectories and fidelities
           np.save(f'../results/traj_model_{filename_core}.npy', traj_model_np)
           np.save(f'../results/traj_vanilla_{filename_core}.npy', traj_van_np)
           if traj_true is not None:
               np.save(f'../results/traj_true_{filename_core}.npy', traj_true_np)
               np.save(f'../results/fidelity_model_{filename_core}.npy', fid_model)
               np.save(f'../results/fidelity_vanilla_{filename_core}.npy', fid_van)


           filename = f'./mixed_state_fidelity_{filename_core}'
           # Adjust layout
           plt.tight_layout()
           plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)

       else:
            print("  No ground-truth trajectory available for fidelity comparison (mixed)")

       return traj_model_np, traj_van_np

def plot_purity(traj_model_np, traj_van_np, t_grid_long, config):
    
       purity_model = calculate_purity(traj_model_np)
       purity_van = calculate_purity(traj_van_np)
       fig = plt.figure(figsize=(6,4))
       plt.plot(t_grid_long, purity_model, 'r', label='NDE Purity')
       plt.plot(t_grid_long, purity_van, 'b--', label='Vanilla Purity')
       plt.axvspan(0, config["t_max"], color='gray', alpha=0.1)
       plt.title('Purity over time'); plt.xlabel('Time'); plt.ylabel('Tr(ρ²)'); plt.legend(); plt.grid(True)

       N = config['L']
       h_type = config['hamiltonian_type']
       noise =  config['use_noisy_dynamics']
       noise_type = config['noise_model']


       filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"

       # Save purity data
       np.save(f'../results/purity_model_{filename_core}.npy', purity_model)
       np.save(f'../results/purity_vanilla_{filename_core}.npy', purity_van)
       np.save(f'../results/time_grid_{filename_core}.npy', t_grid_long)
       
       filename = f'./purity_{filename_core}'
       # Adjust layout
       plt.tight_layout()
       plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)

def plot_pure_state_fidelity(traj_model, traj_vanilla, traj_true, config, t_grid_long, L):
       psi_model_np = np.array(jax.device_get(traj_model))
       psi_van_np = np.array(jax.device_get(traj_vanilla))
       if traj_true is not None:
           psi_true_np = np.array(jax.device_get(traj_true))
           fid_nde = 1.0 - np.array([calculate_fidelity_pure(psi_true_np[k], psi_model_np[k]) for k in range(len(psi_true_np))])
           fid_van = 1.0 - np.array([calculate_fidelity_pure(psi_true_np[k], psi_van_np[k]) for k in range(len(psi_true_np))])
           fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
           fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {config['hamiltonian_type']}")
           ax1.set_yscale('log')
           ax1.plot(t_grid_long, 1.0 - fid_nde, 'r', label='NDE Fidelity')
           ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax1.legend(); ax1.set_title('Fidelity')
           ax2.set_yscale('log')
           ax2.plot(t_grid_long, fid_nde, 'r', label='NDE Infidelity')
           ax2.plot(t_grid_long, fid_van, 'b--', label='Vanilla Infidelity')
           ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax2.legend(); ax2.set_title('Infidelity')

           N = config['L']
           h_type = config['hamiltonian_type']
           noise =  config['use_noisy_dynamics']
           noise_type = config['noise_model']

           filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"

           # Save trajectories and fidelities
           np.save(f'../results/psi_model_{filename_core}.npy', psi_model_np)
           np.save(f'../results/psi_vanilla_{filename_core}.npy', psi_van_np)
           if traj_true is not None:
               np.save(f'../results/psi_true_{filename_core}.npy', psi_true_np)
               np.save(f'../results/fidelity_pure_model_{filename_core}.npy', fid_nde)
               np.save(f'../results/fidelity_pure_vanilla_{filename_core}.npy', fid_van)

           filename = f'./pure_fidelity_{filename_core}'
           # Adjust layout
           plt.tight_layout()
           plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)

       else:
           print("  No ground-truth trajectory available for fidelity comparison (pure)")


def plot_observables(t_grid, obs_true, obs_model, obs_vanilla, L, hamiltonian_type, config):
    """Plot observable trajectories"""
    # Select observables to plot
    if hamiltonian_type == "general_local_zz" and L > 2:
        # Plot single-qubit and nearest-neighbor observables
        single_obs = [f'X_{i}' for i in range(min(L, 4))] + [f'Z_{i}' for i in range(min(L, 4))]
        pair_obs = [f'Z_{i} Z_{i+1}' for i in range(min(L-1, 3))]
        obs_to_plot = single_obs + pair_obs
    else:
        obs_to_plot = ['X_0', 'Y_0', 'Z_0']
        if L >= 2:
            obs_to_plot += ['X_0 X_1', 'Y_0 Y_1', 'Z_0 Z_1']
    
    n_obs = len(obs_to_plot)
    n_cols = min(3, n_obs)
    n_rows = (n_obs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, obs_name in enumerate(obs_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if obs_true is not None and obs_name in obs_true:
            ax.plot(t_grid, obs_true[obs_name], 'k-', label='True', linewidth=2)
        if obs_name in obs_model:
            ax.plot(t_grid, obs_model[obs_name], 'r--', label='NDE', linewidth=2)
        if obs_name in obs_vanilla:
            ax.plot(t_grid, obs_vanilla[obs_name], 'b:', label='Vanilla', linewidth=2, alpha=0.7)
        
        ax.axvspan(0, config["t_max"], color='gray', alpha=0.1)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'⟨{obs_name}⟩')
        ax.set_title(obs_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
    
    # Hide unused subplots
    for idx in range(n_obs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Observables | L={L} | {hamiltonian_type}', fontsize=14)

    N = config['L']
    h_type = config['hamiltonian_type']
    noise =  config['use_noisy_dynamics']
    noise_type = config['noise_model']
    filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"

    # Save observables
    if obs_true is not None:
        np.save(f'../results/obs_true_{filename_core}.npy', obs_true)
    np.save(f'../results/obs_model_{filename_core}.npy', obs_model)
    np.save(f'../results/obs_vanilla_{filename_core}.npy', obs_vanilla)
    np.save(f'../results/time_grid_obs_{filename_core}.npy', t_grid)

    filename = f'./observables_{filename_core}'
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)


    return fig


def plot_training_loss(losses, config):
       plt.figure(figsize=(5,4))
       plt.plot(losses)
       plt.title("Training Loss")
       plt.xlabel("Epoch")
       plt.ylabel("Loss")
       plt.grid(True)

       N = config['L']
       h_type = config['hamiltonian_type']
       noise =  config['use_noisy_dynamics']
       noise_type = config['noise_model']

       filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"


       # Save loss history
       np.save(f'../results/loss_history_{filename_core}.npy', np.array(losses))

       filename = f'./loss_{filename_core}'
       # Adjust layout
       plt.tight_layout()
       plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)


def plot_final_probabilities(bitstrings, counts_shots, probs_model, probs_vanilla, config, labels = None):

    sum_counts = sum(counts_shots)
    counts_shots = [i/sum_counts for i in counts_shots]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(bitstrings))
    width = 0.25
    
    ax.bar(x - width, counts_shots, width, label=labels[0] if labels else 'Data 1', alpha=0.7)
    ax.bar(x, probs_model, width, label=labels[1] if labels else 'Data 2', alpha=0.7)
    ax.bar(x + width, probs_vanilla, width, label=labels[2] if labels else 'Data 3', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bitstrings, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    N = config['L']
    h_type = config['hamiltonian_type']
    noise =  config['use_noisy_dynamics']
    noise_type = config['noise_model']


    filename_core = f"L{N}_h_type-{h_type}_noise-{noise}_type{noise_type}"
    filename = f'./final_probabilities_{filename_core}'
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../plots//{filename}.png', bbox_inches='tight', dpi=300)
