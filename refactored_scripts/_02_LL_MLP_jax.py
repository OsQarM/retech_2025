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

Array = jnp.ndarray

# ============================================================
# 0. CONFIG BLOCK
# ============================================================
CONFIG = {
    # SYSTEM PARAMETERS
    "L": 3,
    "t_max": 1.0,
    "initial_state_kind": "all_plus",
    
    # HAMILTONIAN MODEL
    "hamiltonian_type": "uniform_xyz",  # "uniform_xyz" or "general_local_zz"
    
    # NOISE SETTINGS
    "use_noisy_dynamics": True,  # Set to True to use Lindblad dynamics
    "learn_noise_params": True,   # Whether to learn noise rates
    "noise_model": "local",       # "global" (2 params) or "local" (2L params)
    
    # Initial guesses for noise (used if learn_noise_params=True)
    "gamma_dephasing_init": 0,   # Initial dephasing rate (global or per-qubit)
    "gamma_damping_init": 0,    # Initial damping rate (global or per-qubit)
    
    # TRAINING PARAMETERS
    "dt": 0.01,
    "N_epochs": 500,
    "learning_rate": 1e-2,
    "print_every": 20,
    
    # HAMILTONIAN INITIAL GUESS
    "Jx_init": 0.5, "Jy_init": 0.5, "Jz_init": 0.5,
    "hx_init": 0.0, "hy_init": 0.0, "hz_init": 0.5,
    "hx_list_init": [0]*3,
    "hz_list_init": [0]*3,
    "Jzz_list_init": [0]*2,
    "INIT_PERTURB_SCALE": 0.0,
    
    # NDE ARCHITECTURE
    "MODEL_TYPE": "white",
    "NN_MODEL_TYPE": "time_dependent",
    "NN_hidden_sizes": [64],
    "learn_theta": True,
    
    # CURRICULUM
    "PHASE1_SPLIT": 0.4,
    "PHASE2_SPLIT": 0.4,
    "PHASE3_SPLIT": 0.2,
    "lambda_reg": 1e-1,
    "noise_reg": 1e-4,  # Regularization for noise parameters
    "seed_init": 4321,
    "T_extrapolate_factor": 5.0,
}

# ============================================================
# 1. UTILITIES
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

def get_theta_shape(L: int, hamiltonian_type: str) -> int:
    if hamiltonian_type == "uniform_xyz":
        return 6
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def build_xyz_basis(L: int, hamiltonian_type: str = "uniform_xyz", dtype=jnp.complex64):
    sx, sy, sz, id2 = paulis(dtype)
    dim = 2**L
    
    if hamiltonian_type == "uniform_xyz":
        ops_out = []
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L - 1):
                ops = [id2] * L
                ops[i] = pauli
                ops[i+1] = pauli
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        for pauli in [sx, sy, sz]:
            H_term = jnp.zeros((dim, dim), dtype=dtype)
            for i in range(L):
                ops = [id2] * L
                ops[i] = pauli
                H_term = H_term + kron_n(ops)
            ops_out.append(H_term)
        
        return ops_out
    
    elif hamiltonian_type == "general_local_zz":
        ops_out = []
        for i in range(L):
            ops = [id2] * L
            ops[i] = sx
            ops_out.append(kron_n(ops))
        for i in range(L):
            ops = [id2] * L
            ops[i] = sz
            ops_out.append(kron_n(ops))
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = sz
            ops[i+1] = sz
            ops_out.append(kron_n(ops))
        return ops_out
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list, hamiltonian_type: str = "uniform_xyz") -> Array:
    expected_shape = get_theta_shape(L, hamiltonian_type)
    if len(theta) != expected_shape:
        raise ValueError(f"Expected {expected_shape} parameters, got {len(theta)}")
    
    H = jnp.zeros((2**L, 2**L), dtype=jnp.complex64)
    for i in range(expected_shape):
        H += theta[i] * OPS_XYZ[i]
    return H

def build_lindblad_operators(L: int, dtype=jnp.complex64):
    """Build Lindblad jump operators (without rates)"""
    sx, sy, sz, id2 = paulis(dtype)
    sigma_minus = (sx - 1j * sy) / 2.0
    
    dephasing_ops = []
    damping_ops = []
    
    for i in range(L):
        # Dephasing operator: σ_z / sqrt(2)
        ops_dephase = [id2] * L
        ops_dephase[i] = sz / jnp.sqrt(2.0)
        dephasing_ops.append(kron_n(ops_dephase))
        
        # Damping operator: σ_-
        ops_decay = [id2] * L
        ops_decay[i] = sigma_minus
        damping_ops.append(kron_n(ops_decay))
    
    return dephasing_ops, damping_ops

def prepare_initial_state(L: int, kind: str, loaded_vector: np.ndarray = None, 
                          as_density_matrix: bool = False) -> Array:
    """Prepare initial state as vector or density matrix"""
    if loaded_vector is not None:
        psi = jnp.array(loaded_vector, dtype=jnp.complex64)
    else:
        dim = 2**L
        if kind == "all_zeros":
            psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
        elif kind == "all_plus":
            amp = 1.0 / jnp.sqrt(dim)
            psi = jnp.full((dim,), amp, dtype=jnp.complex64)
        else:
            psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    
    if as_density_matrix:
        psi = psi.reshape(-1, 1)
        rho = psi @ psi.conj().T
        return rho
    else:
        return psi

def vectorize_density_matrix(rho):
    return rho.flatten()

def unvectorize_density_matrix(rho_vec, dim):
    return rho_vec.reshape(dim, dim)

def make_observables(L):
    sx, sy, sz, id2 = paulis()
    obs = {}
    
    for qubit in range(L):
        for name, pauli in [('X', sx), ('Y', sy), ('Z', sz)]:
            ops = [id2] * L
            ops[qubit] = pauli
            obs[f'{name}_{qubit}'] = kron_n(ops)
    
    for i in range(L-1):
        for name, pauli in [('X', sx), ('Y', sy), ('Z', sz)]:
            ops = [id2] * L
            ops[i] = pauli
            ops[i+1] = pauli
            obs[f'{name}_{i} {name}_{i+1}'] = kron_n(ops)
    
    return obs

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

# ============================================================
# 2. NEURAL NETWORK
# ============================================================
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

# ============================================================
# 3. DYNAMICS - SCHRÖDINGER (PURE STATES)
# ============================================================
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
        num_ham_params = get_theta_shape(L, hamiltonian_type)
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

def rk4_step_schrodinger(psi, t, dt, rhs_fun, params):
    """RK4 for pure state"""
    dt_c = jnp.asarray(dt, dtype=psi.dtype)
    k1 = rhs_fun(t, psi, params)
    k2 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, psi + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, psi + dt_c*k3, params)
    psi_next = psi + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return psi_next / (jnp.linalg.norm(psi_next) + 1e-12)

def evolve_schrodinger(psi0, t_grid, rhs_fun, params):
    """Evolve pure state"""
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    @jax.jit
    def scan_fn(psi_prev, t_dt):
        t_prev, dt = t_dt
        psi_next = rk4_step_schrodinger(psi_prev, t_prev, dt, rhs_fun, params)
        return psi_next, psi_next
    
    _, psi_traj_scan = jax.lax.scan(scan_fn, psi0, (t_prev_grid, dt_grid))
    return jnp.concatenate([psi0[None, :], psi_traj_scan], axis=0)

# ============================================================
# 4. DYNAMICS - LINDBLAD (MIXED STATES)
# ============================================================
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
        num_ham_params = get_theta_shape(L, hamiltonian_type)
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

def rk4_step_lindblad(rho_vec, t, dt, rhs_fun, params, dim):
    """RK4 for density matrix"""
    dt_c = jnp.asarray(dt, dtype=rho_vec.dtype)
    k1 = rhs_fun(t, rho_vec, params)
    k2 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, rho_vec + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, rho_vec + dt_c*k3, params)
    
    rho_next_vec = rho_vec + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Enforce trace preservation
    rho_next = unvectorize_density_matrix(rho_next_vec, dim)
    rho_next = 0.5 * (rho_next + rho_next.conj().T)  # Hermiticity
    trace = jnp.trace(rho_next).real
    rho_next = rho_next / (trace + 1e-12)
    
    return vectorize_density_matrix(rho_next)

def evolve_lindblad(rho0, t_grid, rhs_fun, params):
    """Evolve density matrix"""
    dim = rho0.shape[0]
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    rho0_vec = vectorize_density_matrix(rho0)
    
    @jax.jit
    def scan_fn(rho_prev_vec, t_dt):
        t_prev, dt = t_dt
        rho_next_vec = rk4_step_lindblad(rho_prev_vec, t_prev, dt, rhs_fun, params, dim)
        return rho_next_vec, rho_next_vec
    
    _, rho_traj_vec = jax.lax.scan(scan_fn, rho0_vec, (t_prev_grid, dt_grid))
    
    rho_traj = jnp.concatenate([
        rho0[None, ...],
        jnp.array([unvectorize_density_matrix(rv, dim) for rv in rho_traj_vec])
    ])
    
    return rho_traj

# ============================================================
# 5. LOSS FUNCTIONS
# ============================================================
def nde_loss_schrodinger(params, L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE,
                         hamiltonian_type, lambda_reg, t_grid_shots, psi0, counts_shots):
    """Loss for pure state dynamics"""
    rhs_fun = make_rhs_schrodinger(L, OPS_XYZ, NN_MAP_FUN, NN_MODEL_TYPE, MODEL_TYPE, hamiltonian_type)
    psi_traj = evolve_schrodinger(psi0, t_grid_shots, rhs_fun, params)
    
    # Negative log-likelihood
    probs = jnp.abs(psi_traj)**2
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = jnp.clip(probs, 1e-9, 1.0)
    logp = jnp.log(probs)
    ll = jnp.sum(counts_shots * logp)
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

# ============================================================
# 6. TRAINING
# ============================================================
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

# ============================================================
# 7. DATA LOADING
# ============================================================
def load_experimental_data(config):
    """Load experimental/simulated data"""
    L = config["L"]
    T_max = config["t_max"]
    search_pattern = f"experimental_data_L{L}_T{T_max:.2f}_*_config_df.csv"
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No data found for L={L}, T={T_max:.2f}")
    
    config_file = files[0]
    file_core = config_file.replace("_config_df.csv", "").replace("experimental_data_", "")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {file_core}")
    print(f"{'='*60}")
    
    df_counts = pd.read_csv(f"experimental_data_{file_core}_counts.csv", index_col='Time')
    df_config = pd.read_csv(config_file).iloc[0]
    
    try:
        metadata = np.load(f"experimental_data_{file_core}_metadata.npz")
        initial_state_vector = metadata["initial_state"]
        theta_true_array = metadata["theta_true"]
    except:
        initial_state_vector = None
        theta_true_array = None
    
    # Update config from data
    config["N_time_shots"] = int(df_config["N_time_shots"])
    config["N_shots_per_time"] = int(df_config["N_shots_per_time"])
    
    if "hamiltonian_type" in df_config:
        data_ham_type = df_config["hamiltonian_type"]
        if config["hamiltonian_type"] != data_ham_type:
            print(f"⚠️  WARNING: Config hamiltonian_type '{config['hamiltonian_type']}' "
                  f"differs from data '{data_ham_type}'")
            config["hamiltonian_type"] = data_ham_type
    
    # Check if data was generated with noise
    data_has_noise = False
    if "dynamics_type" in df_config:
        data_has_noise = (df_config["dynamics_type"] == "lindblad")
        if data_has_noise:
            print(f"  Data generated with Lindblad dynamics (noisy)")
            if "T1" in df_config and "T2" in df_config:
                T1_true = float(df_config["T1"])
                T2_true = float(df_config["T2"])
                print(f"  True noise: T1={T1_true:.2f}, T2={T2_true:.2f}")
                gamma_deph_true = 1.0/T2_true - 1.0/(2*T1_true)
                gamma_damp_true = 1.0/T1_true
                print(f"  Corresponding rates: γ_deph={gamma_deph_true:.4f}, γ_damp={gamma_damp_true:.4f}")
                config["gamma_deph_true"] = gamma_deph_true
                config["gamma_damp_true"] = gamma_damp_true
        else:
            print(f"  Data generated with Schrödinger dynamics (noiseless)")
    
    if data_has_noise and not config["use_noisy_dynamics"]:
        print(f"  ⚠️  WARNING: Data is noisy but model uses noiseless dynamics!")
    elif not data_has_noise and config["use_noisy_dynamics"]:
        print(f"  ⚠️  WARNING: Data is noiseless but model uses noisy dynamics!")
    
    # Load true Hamiltonian parameters
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    if hamiltonian_type == "uniform_xyz" and "Jx_true" in df_config:
        config["Jx_true"] = float(df_config["Jx_true"])
        config["Jy_true"] = float(df_config["Jy_true"])
        config["Jz_true"] = float(df_config["Jz_true"])
        config["hx_true"] = float(df_config["hx_true"])
        config["hy_true"] = float(df_config["hy_true"])
        config["hz_true"] = float(df_config["hz_true"])
    elif hamiltonian_type == "general_local_zz" and theta_true_array is not None:
        L = config["L"]
        config["hx_list_true"] = list(theta_true_array[:L])
        config["hz_list_true"] = list(theta_true_array[L:2*L])
        config["Jzz_list_true"] = list(theta_true_array[2*L:])
    
    print(f"  Measurement shots: R={config['N_shots_per_time']}, J={config['N_time_shots']}")
    
    t_grid_shots = df_counts.index.values.astype(np.float32)
    counts_shots = df_counts.values.astype(np.int32)
    
    return t_grid_shots, counts_shots, initial_state_vector, theta_true_array

# ============================================================
# 8. DIAGNOSTICS AND VISUALIZATION
# ============================================================
def relative_absolute_error(theta_true, theta_learned):
    """Calculate relative error"""
    error = np.sum(np.abs(np.array(theta_true) - np.array(theta_learned)))
    norm = np.sum(np.abs(np.array(theta_true)))
    return error / (norm + 1e-12)

def plot_hamiltonian_parameters(theta_true, theta_init, theta_final, hamiltonian_type, L):
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
    
    plt.tight_layout()
    return fig

def plot_noise_parameters(learned_rates, true_rates, noise_model, L):
    """Plot learned vs true noise rates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
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
    
    plt.tight_layout()
    return fig

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
    plt.tight_layout()
    return fig

# ============================================================
# 9. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    config = copy.deepcopy(CONFIG)
    
    # Load data
    t_grid_shots, counts_shots, initial_state_vector, theta_true_array = load_experimental_data(config)
    
    L = config["L"]
    dim = 2**L
    hamiltonian_type = config["hamiltonian_type"]
    use_noisy = config["use_noisy_dynamics"]
    
    # Prepare initial state
    state0 = prepare_initial_state(L, config["initial_state_kind"], 
                                   initial_state_vector, 
                                   as_density_matrix=use_noisy)
    
    # Build operators
    OPS_XYZ = build_xyz_basis(L, hamiltonian_type)
    NUM_COEFFICIENTS = get_theta_shape(L, hamiltonian_type)
    
    # Build Lindblad operators if needed
    if use_noisy:
        dephasing_ops, damping_ops = build_lindblad_operators(L)
    else:
        dephasing_ops, damping_ops = None, None
    
    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Dynamics: {'Lindblad (noisy)' if use_noisy else 'Schrödinger (noiseless)'}")
    print(f"  Hamiltonian type: {hamiltonian_type}")
    print(f"  Hamiltonian parameters: {NUM_COEFFICIENTS}")
    print(f"  Model type: {config['MODEL_TYPE']}")
    print(f"  Learn θ: {config['learn_theta']}")
    
    if use_noisy:
        print(f"  Learn noise: {config['learn_noise_params']}")
        print(f"  Noise model: {config['noise_model']}")
        if config['noise_model'] == 'global':
            print(f"    → 2 noise parameters total")
        else:
            print(f"    → {2*L} noise parameters (2 per qubit)")
    
    # Initialize NN
    NN_MAP_FUN = mlp_forward
    NN_INPUT_DIM = 1  # Time-dependent
    NN_OUTPUT_DIM = NUM_COEFFICIENTS  # NN only outputs Hamiltonian corrections
    
    layer_sizes = [NN_INPUT_DIM] + config["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
    key = random.PRNGKey(config["seed_init"])
    key, k_nn, k_th, k_noise = random.split(key, 4)
    nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)
    
    # Initialize Hamiltonian parameters
    if hamiltonian_type == "uniform_xyz":
        theta_init_list = [config["Jx_init"], config["Jy_init"], config["Jz_init"],
                          config["hx_init"], config["hy_init"], config["hz_init"]]
    elif hamiltonian_type == "general_local_zz":
        theta_init_list = (list(config["hx_list_init"]) + 
                          list(config["hz_list_init"]) + 
                          list(config["Jzz_list_init"]))
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
    theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
    if config["INIT_PERTURB_SCALE"] > 0:
        theta_init += config["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_COEFFICIENTS,))
    
    # Initialize noise parameters if needed
    params = {"theta": theta_init, "nn": nn_params}
    
    if use_noisy and config["learn_noise_params"]:
        if config["noise_model"] == "global":
            # Single rate for all qubits
            noise_rates_init = jnp.array([
                config["gamma_dephasing_init"],
                config["gamma_damping_init"]
            ], dtype=jnp.float32)
        else:  # local
            # Per-qubit rates
            noise_rates_init = jnp.concatenate([
                jnp.full((L,), config["gamma_dephasing_init"], dtype=jnp.float32),
                jnp.full((L,), config["gamma_damping_init"], dtype=jnp.float32)
            ])
        params["noise_rates"] = noise_rates_init
        print(f"  Initial noise rates: {noise_rates_init}")
    
    # Create time grids
    t_grid_fine = jnp.arange(0.0, config["t_max"] + config["dt"]/2, config["dt"])
    T_extrap = config["t_max"] * config["T_extrapolate_factor"]
    t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, config["dt"])
    
    # Create step function
    step_fn = make_step_fn(config, L, OPS_XYZ, NN_MAP_FUN, use_noisy,
                          dephasing_ops, damping_ops)
    
    # Training schedule
    N_total = config["N_epochs"]
    P1 = int(N_total * config["PHASE1_SPLIT"])
    P2 = int(N_total * config["PHASE2_SPLIT"])
    P3 = N_total - P1 - P2
    
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    print(f"  Phase 1 (Warm-up): {P1} epochs")
    print(f"  Phase 2 (Distill): {P2} epochs")
    print(f"  Phase 3 (Refine): {P3} epochs")
    
    # Phase 1: Train everything
    learn_noise_p1 = config["learn_noise_params"] if use_noisy else False
    params, l1 = train_phase(params, P1, config, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=config["learn_theta"],
                            train_nn=True, train_noise=learn_noise_p1,
                            phase_name="P1 Warm-up")
    
    # Phase 2: Distill to theta (freeze NN and noise)
    params, l2 = train_phase(params, P2, config, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=config["learn_theta"],
                            train_nn=False, train_noise=False,
                            phase_name="P2 Distill")
    
    # Phase 3: Refine NN (freeze theta, optionally train noise)
    learn_noise_p3 = config["learn_noise_params"] if use_noisy else False
    params, l3 = train_phase(params, P3, config, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=False,
                            train_nn=True, train_noise=learn_noise_p3,
                            phase_name="P3 Refine")
    
    losses = l1 + l2 + l3
    
    # Extract final parameters
    theta_final = np.array(jax.device_get(params["theta"]))
    nn_l2_norm = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")


    
    # Print Hamiltonian parameters
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
    
    # Print noise parameters if learned
    if "noise_rates" in params:
        noise_rates_final = np.array(jax.device_get(params["noise_rates"]))
        print("\nLearned Noise Parameters:")
        if config["noise_model"] == "global":
            print(f"  Dephasing rate (global): {noise_rates_final[0]:.5f}")
            print(f"  Damping rate (global): {noise_rates_final[1]:.5f}")
        else:
            print("  Dephasing rates (per qubit):")
            for i in range(L):
                print(f"    Qubit {i}: {noise_rates_final[i]:.5f}")
            print("  Damping rates (per qubit):")
            for i in range(L):
                print(f"    Qubit {i}: {noise_rates_final[L+i]:.5f}")
    
    # Compare with true parameters
    if theta_true_array is not None:
        theta_true = np.array(theta_true_array)
        rel_error = relative_absolute_error(theta_true, theta_final)
        print(f"\nTrue Hamiltonian Parameters: {np.round(theta_true, 4)}")
        print(f"Relative Error: {rel_error:.4f}")
    else:
        theta_true = None
    
    # Generate trajectories for diagnostics
    print(f"\nGenerating trajectories for diagnostics...")
    
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
        traj_model = evolve_schrodinger(state0, t_grid_long, rhs_model, params)
        
        # Vanilla (theta only)
        params_vanilla = {"theta": params["theta"],
                         "nn": jtu.tree_map(jnp.zeros_like, params["nn"])}
        traj_vanilla = evolve_schrodinger(state0, t_grid_long, rhs_model, params_vanilla)
        
        # True trajectory if available
        if theta_true is not None:
            H_true = xyz_hamiltonian_from_theta(L, jnp.array(theta_true), OPS_XYZ, hamiltonian_type)
            rhs_true = lambda t, psi, p: -1j * H_true @ psi
            traj_true = evolve_schrodinger(state0, t_grid_long, rhs_true, {})
        else:
            traj_true = None
        
        # Calculate observables
        obs_dict = make_observables(L)
        obs_model = calculate_observables_pure(traj_model, obs_dict)
        obs_vanilla = calculate_observables_pure(traj_vanilla, obs_dict)
        obs_true = calculate_observables_pure(traj_true, obs_dict) if traj_true is not None else None
    
    # PLOTTING
    print(f"\nGenerating plots...")
    
    # 1. Hamiltonian parameters
    fig_ham = plot_hamiltonian_parameters(theta_true, np.array(theta_init), 
                                         theta_final, hamiltonian_type, L)
    plt.show()
    
    # 2. Noise parameters (if applicable)
    if "noise_rates" in params:
        true_noise_rates = None
        if "gamma_deph_true" in config and "gamma_damp_true" in config:
            if config["noise_model"] == "global":
                true_noise_rates = [config["gamma_deph_true"], config["gamma_damp_true"]]
            else:
                true_noise_rates = ([config["gamma_deph_true"]] * L + [config["gamma_damp_true"]] * L)

        # Learned noise rates (convert to numpy)
        try:
            learned_noise_rates = np.array(jax.device_get(params["noise_rates"]))
        except Exception:
            learned_noise_rates = None

        try:
            fig_noise = plot_noise_parameters(learned_noise_rates, true_noise_rates, config["noise_model"], L)
            plt.show()
        except Exception as e:
            print(f"  Warning: failed to plot noise parameters: {e}")

    # 3. Fidelity / Purity diagnostics
    print("\nComputing fidelity and purity diagnostics...")
    if use_noisy:
        # Mixed-state fidelity and purity
        try:
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
                fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {hamiltonian_type} (mixed)")
                ax1.set_yscale('log')
                ax1.plot(t_grid_long, 1.0 - fid_model, 'r', label='NDE Infidelity')
                ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1)
                ax1.set_title('Infidelity (linear)'); ax1.legend()
                ax2.set_yscale('log')
                ax2.plot(t_grid_long + 1e-12, inf_model, 'r', label='NDE Infidelity')
                ax2.plot(t_grid_long + 1e-12, inf_van, 'b--', label='Vanilla Infidelity')
                ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1)
                ax2.set_title('Infidelity (log-log)'); ax2.legend()
                plt.tight_layout(); plt.show()
            else:
                print("  No ground-truth trajectory available for fidelity comparison (mixed)")

            # Purity trajectories
            purity_model = calculate_purity(traj_model_np)
            purity_van = calculate_purity(traj_van_np)
            fig = plt.figure(figsize=(6,4))
            plt.plot(t_grid_long, purity_model, 'r', label='NDE Purity')
            plt.plot(t_grid_long, purity_van, 'b--', label='Vanilla Purity')
            plt.axvspan(0, config["t_max"], color='gray', alpha=0.1)
            plt.title('Purity over time'); plt.xlabel('Time'); plt.ylabel('Tr(ρ²)'); plt.legend(); plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"  Warning: failed mixed-state diagnostics: {e}")

    else:
        # Pure-state fidelity (same style as original script)
        try:
            psi_model_np = np.array(jax.device_get(traj_model))
            psi_van_np = np.array(jax.device_get(traj_vanilla))
            if traj_true is not None:
                psi_true_np = np.array(jax.device_get(traj_true))
                fid_nde = 1.0 - np.array([calculate_fidelity_pure(psi_true_np[k], psi_model_np[k]) for k in range(len(psi_true_np))])
                fid_van = 1.0 - np.array([calculate_fidelity_pure(psi_true_np[k], psi_van_np[k]) for k in range(len(psi_true_np))])

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"L = {L} | {config['initial_state_kind']} | {hamiltonian_type}")
                ax1.set_yscale('log')
                ax1.plot(t_grid_long, 1.0 - fid_nde, 'r', label='NDE Fidelity')
                ax1.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax1.legend(); ax1.set_title('Fidelity')
                ax2.set_yscale('log')
                ax2.plot(t_grid_long, fid_nde, 'r', label='NDE Infidelity')
                ax2.plot(t_grid_long, fid_van, 'b--', label='Vanilla Infidelity')
                ax2.axvspan(0, config["t_max"], color='gray', alpha=0.1); ax2.legend(); ax2.set_title('Infidelity')
                plt.tight_layout(); plt.show()
            else:
                print("  No ground-truth trajectory available for fidelity comparison (pure)")
        except Exception as e:
            print(f"  Warning: failed pure-state diagnostics: {e}")

    # 4. Observables
    try:
        fig_obs = plot_observables(t_grid_long, obs_true, obs_model, obs_vanilla, L, hamiltonian_type, config)
        plt.show()
    except Exception as e:
        print(f"  Warning: failed to plot observables: {e}")

    # 5. Training loss
    try:
        plt.figure(figsize=(5,4))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"  Warning: failed to plot training loss: {e}")
