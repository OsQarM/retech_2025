#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

@author: marcin
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys


sys.path.append('./')


def sample_bitstrings_from_trajectory(config, seed, L, state_traj_fine, t_grid_fine, t_grid_shots, dynamics_type):
    # Sample at measurement times
    idx_shots = np.searchsorted(np.array(t_grid_fine), np.array(t_grid_shots))
    state_traj_shots = state_traj_fine[idx_shots]
    N_shots = config["N_shots_per_time"]
    
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
    
    return t_grid_shots, counts_shots