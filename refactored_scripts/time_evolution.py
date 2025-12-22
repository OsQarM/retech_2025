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

def vectorize_density_matrix(rho):
    return rho.flatten()

def unvectorize_density_matrix(rho_vec, dim):
    return rho_vec.reshape(dim, dim)


def rk4_step(state, t, dt, rhs_fun, params):
    dt_c = jnp.asarray(dt, dtype=state.dtype)
    k1 = rhs_fun(t, state, params)
    k2 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k1, params)
    k3 = rhs_fun(t + 0.5*dt_c, state + 0.5*dt_c*k2, params)
    k4 = rhs_fun(t + dt_c, state + dt_c*k3, params)
    state_next = state + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    if state.ndim == 1:  # State vector
        norm = jnp.linalg.norm(state_next)
        return state_next / (norm + 1e-12)
    else:  # Density matrix
        state_next = 0.5 * (state_next + state_next.conj().T)
        trace = jnp.trace(state_next).real
        return state_next / (trace + 1e-12)
    

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


def evolve_trajectory(state0, t_grid, rhs_fun, params):
    '''
    Performs full time evolution of initial state according to specific differential equation (rhs_fun),
    with given variational parameters, and for every time in a given list (t_grid)
    Returns list of all states in the trajectory
    '''
    dt_grid = t_grid[1:] - t_grid[:-1]
    t_prev_grid = t_grid[:-1]
    
    def scan_fn(state_prev, t_dt):
        t_prev, dt = t_dt
        state_next = rk4_step(state_prev, t_prev, dt, rhs_fun, params)
        return state_next, state_next
    
    # Compile the scan function
    scan_fn_jitted = jax.jit(scan_fn)
    
    # Run the scan
    _, state_traj_scan = jax.lax.scan(scan_fn_jitted, state0, (t_prev_grid, dt_grid))
    return jnp.concatenate([state0[None, ...], state_traj_scan], axis=0)



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



