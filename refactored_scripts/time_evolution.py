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



