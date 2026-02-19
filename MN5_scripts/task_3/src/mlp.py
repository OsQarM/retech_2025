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

from model_building import xyz_hamiltonian_from_theta, get_theta_shape
from time_evolution import evolve_trajectory, evolve_lindblad, vectorize_density_matrix, unvectorize_density_matrix, apply_rotations

Array = jnp.ndarray


def init_mlp_params(layer_sizes, key, scale=0.1):
    params = []
    keys = random.split(key, len(layer_sizes)-1)
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        W = scale * random.normal(k, (m, n))
        b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params

def mlp_forward(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]

def adam_init(params):
    m = jtu.tree_map(jnp.zeros_like, params)
    v = jtu.tree_map(jnp.zeros_like, params)
    return {"step": 0, "m": m, "v": v}

def adam_update(params, grads, opt_state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    step = opt_state["step"] + 1
    m = jtu.tree_map(lambda m, g: beta1*m + (1-beta1)*g, opt_state["m"], grads)
    v = jtu.tree_map(lambda v, g: beta2*v + (1-beta2)*(g*g), opt_state["v"], grads)
    m_hat = jtu.tree_map(lambda x: x / (1 - beta1**step), m)
    v_hat = jtu.tree_map(lambda x: x / (1 - beta2**step), v)
    params_new = jtu.tree_map(lambda p, mh, vh: p - lr*mh/(jnp.sqrt(vh)+eps), 
                               params, m_hat, v_hat)
    return params_new, {"step": step, "m": m, "v": v}

def get_trainable_mask(params, train_theta, train_nn, train_noise):
    """Create mask for which parameters to train"""
    mask = {}
    mask["theta"] = jnp.ones_like(params["theta"], dtype=bool) if train_theta else jnp.zeros_like(params["theta"], dtype=bool)
    mask["nn"] = jtu.tree_map(lambda p: jnp.ones_like(p, dtype=bool) if train_nn else jnp.zeros_like(p, dtype=bool), 
                              params["nn"])
    if "noise_rates" in params:
        mask["noise_rates"] = jnp.ones_like(params["noise_rates"], dtype=bool) if train_noise else jnp.zeros_like(params["noise_rates"], dtype=bool)
    return mask


def make_step_fn(config, L, OPS_XYZ, NN_MAP_FUN, use_noisy_dynamics, 
                 dephasing_ops=None, damping_ops=None):
    """Create training step function"""
    
    if use_noisy_dynamics:
        loss_fn = lambda params, t_grid, state0, counts: nde_loss_lindblad(
            params, L, OPS_XYZ, NN_MAP_FUN, config["NN_MODEL_TYPE"], 
            config["MODEL_TYPE"], config["hamiltonian_type"],
            config["lambda_reg"], config["noise_reg"], t_grid, state0, counts,
            dephasing_ops, damping_ops, config["noise_model"]
        )
    else:
        loss_fn = lambda params, t_grid, state0, counts: nde_loss_schrodinger(
            params, L, OPS_XYZ, NN_MAP_FUN, config["NN_MODEL_TYPE"],
            config["MODEL_TYPE"], config["hamiltonian_type"],
            config["lambda_reg"], t_grid, state0, counts
        )
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    @jax.jit
    def step_fn(params, opt_state, t_grid, state0, counts, trainable_mask):
        (loss_val, aux), grads = grad_fn(params, t_grid, state0, counts)
        masked_grads = jtu.tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), 
                                    grads, trainable_mask)
        params_new, opt_state_new = adam_update(params, masked_grads, opt_state, 
                                               lr=config["learning_rate"])
        return params_new, opt_state_new, loss_val, aux
    
    return step_fn


def make_rhs_schrodinger(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type):
    """Create RHS for pure Schrödinger dynamics (no noise)"""
    dim = 2**L
    
    def H_phys(params):
        if MODEL_TYPE == "white":
            return xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ, hamiltonian_type)
        else:
            return jnp.zeros((dim, dim), dtype=jnp.complex64)
    
    def H_NN_time_dependent(nn_params, t):
        t_input = jnp.array([[t]])
        coeffs = NN_MAP_FUN(nn_params, t_input)[0]
        num_ham_params = len(OPS_XYZ)
        coeffs = coeffs[:num_ham_params]  # Only use Hamiltonian coefficients
        H_nn = jnp.zeros((dim, dim), dtype=jnp.complex64)
        for k in range(num_ham_params):
            H_nn += coeffs[k] * OPS_XYZ[k]
        return H_nn
    
    def rhs_ode(t: float, psi: Array, params: dict):
        H_total = H_phys(params)
        if MODEL_TYPE != "black":
            H_total = H_total + H_NN_time_dependent(params["nn"], t)
        return -1j * (H_total @ psi)
    
    return rhs_ode


def make_rhs_lindblad(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type,
                      dephasing_ops, damping_ops, noise_model):
    """Create RHS for Lindblad dynamics with learnable noise"""
    dim = 2**L
    
    def H_phys(params):
        if MODEL_TYPE == "white":
            return xyz_hamiltonian_from_theta(L, params["theta"], OPS_XYZ, hamiltonian_type)
        else:
            return jnp.zeros((dim, dim), dtype=jnp.complex64)
    
    def H_NN_time_dependent(nn_params, t):
        t_input = jnp.array([[t]])
        nn_out = NN_MAP_FUN(nn_params, t_input)[0]
        num_ham_params = len(OPS_XYZ)
        coeffs = nn_out[:num_ham_params]
        H_nn = jnp.zeros((dim, dim), dtype=jnp.complex64)
        for k in range(num_ham_params):
            H_nn += coeffs[k] * OPS_XYZ[k]
        return H_nn
    
    def extract_noise_rates(params):
        """Extract noise rates from params (either fixed or learnable)"""
        if "noise_rates" in params:
            # Learnable noise rates
            gamma_deph = jnp.abs(params["noise_rates"][0])
            gamma_damp = jnp.abs(params["noise_rates"][1])
            
            if noise_model == "global":
                # Single rate for all qubits
                gamma_deph_vec = jnp.full((L,), gamma_deph)
                gamma_damp_vec = jnp.full((L,), gamma_damp)
            else:  # local
                # Per-qubit rates
                gamma_deph_vec = jnp.abs(params["noise_rates"][:L])
                gamma_damp_vec = jnp.abs(params["noise_rates"][L:2*L])
        else:
            # No learnable noise (shouldn't happen in Lindblad mode)
            gamma_deph_vec = jnp.zeros((L,))
            gamma_damp_vec = jnp.zeros((L,))
        
        return gamma_deph_vec, gamma_damp_vec
    
    def rhs_ode(t: float, rho_vec: Array, params: dict):
        rho = unvectorize_density_matrix(rho_vec, dim)
        
        # Hamiltonian evolution
        H_total = H_phys(params)
        if MODEL_TYPE != "black":
            H_total = H_total + H_NN_time_dependent(params["nn"], t)
        
        drho = -1j * (H_total @ rho - rho @ H_total)
        
        # Lindblad dissipators
        gamma_deph_vec, gamma_damp_vec = extract_noise_rates(params)
        
        # Vectorized computation
        Z = jnp.stack(dephasing_ops)
        Sm = jnp.stack(damping_ops)
        Sp = jnp.conj(jnp.swapaxes(Sm, -1, -2))
        
        # Dephasing: γ (Z ρ Z - ρ)
        deph = gamma_deph_vec[:, None, None] * (Z @ rho @ Z - rho)
        
        # Damping: γ (σ- ρ σ+ - 0.5{σ+ σ-, ρ})
        jump = Sm @ rho @ Sp
        anticomm = 0.5 * (Sp @ Sm @ rho + rho @ Sp @ Sm)
        damp = gamma_damp_vec[:, None, None] * (jump - anticomm)
        
        diss = jnp.sum(deph + damp, axis=0)
        
        return vectorize_density_matrix(drho + diss)
    
    return rhs_ode



def nde_loss_schrodinger(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE,
                         hamiltonian_type, lambda_reg, t_grid_shots, psi0, counts_shots):
    """Loss for pure state dynamics"""
    rhs_fun = make_rhs_schrodinger(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type)
    psi_traj = evolve_trajectory(psi0, t_grid_shots, rhs_fun, params)
    
    # Negative log-likelihood
    probs = jnp.abs(psi_traj)**2
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = jnp.clip(probs, 1e-9, 1.0)
    logp = jnp.log(probs)

    #In case we only have measurements of the last timestamp
    if counts_shots.ndim == 1:
        logp_final = logp[-1, :]
    else:
        logp_final = logp
    ll = jnp.sum(counts_shots * logp_final)

    #ll = jnp.sum(counts_shots * logp)
    loss_nll = -ll / jnp.sum(counts_shots)
    
    # Regularization
    reg_nn = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))
    
    total_loss = loss_nll + lambda_reg * reg_nn
    
    return total_loss, (loss_nll, reg_nn, 0.0, psi_traj)

def nde_loss_lindblad(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE,
                      hamiltonian_type, lambda_reg, noise_reg, t_grid_shots, rho0,
                      counts_shots, dephasing_ops, damping_ops, noise_model):
    """Loss for mixed state dynamics with learnable noise"""
    rhs_fun = make_rhs_lindblad(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE,
                                hamiltonian_type, dephasing_ops, damping_ops, noise_model)
    rho_traj = evolve_lindblad(rho0, t_grid_shots, rhs_fun, params)
    
    # Negative log-likelihood
    probs = jnp.diagonal(rho_traj, axis1=1, axis2=2).real
    probs = jnp.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    logp = jnp.log(probs)
    ll = jnp.sum(counts_shots * logp)
    loss_nll = -ll / jnp.sum(counts_shots)
    
    # Regularization
    reg_nn = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))
    
    # Noise regularization (penalize large noise rates)
    reg_noise = 0.0
    if "noise_rates" in params:
        reg_noise = jnp.sum(jnp.abs(params["noise_rates"]))
    
    total_loss = loss_nll + lambda_reg * reg_nn + noise_reg * reg_noise
    
    return total_loss, (loss_nll, reg_nn, reg_noise, rho_traj)



def train_phase(params_init, N_epochs, config, step_fn, t_grid_shots, state0, 
                counts_shots, train_theta, train_nn, train_noise, phase_name):
    """Train for one phase"""
    if N_epochs <= 0:
        return params_init, []
    
    train_theta_actual = train_theta and (config["MODEL_TYPE"] == "white")
    
    print(f"\n--- Phase: {phase_name} ---")
    print(f"  θ: {'TRAIN' if train_theta_actual else 'FROZEN'}, "
          f"φ: {'TRAIN' if train_nn else 'FROZEN'}, "
          f"γ: {'TRAIN' if train_noise else 'FROZEN'}")
    
    params = params_init
    opt_state = adam_init(params_init)
    losses = []
    trainable_mask = get_trainable_mask(params, train_theta_actual, train_nn, train_noise)
    
    for epoch in range(1, N_epochs + 1):
        params, opt_state, loss_val, (loss_nll, reg_nn, reg_noise, traj) = step_fn(
            params, opt_state, t_grid_shots, state0, counts_shots, trainable_mask
        )
        
        loss_val = float(jax.device_get(loss_val))
        losses.append(loss_val)
        
        if epoch % config["print_every"] == 0 or epoch == N_epochs:
            loss_nll_val = float(jax.device_get(loss_nll))
            reg_nn_val = float(jax.device_get(reg_nn))
            reg_noise_val = float(jax.device_get(reg_noise))
            print(f"  [{epoch:03d}/{N_epochs}] Loss: {loss_val:.4e} | "
                  f"NLL: {loss_nll_val:.4e} | Reg_NN: {reg_nn_val:.4e} | "
                  f"Reg_Noise: {reg_noise_val:.4e}")
    
    return params, losses