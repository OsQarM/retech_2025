#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:31:09 2025

@author: marcin
"""

import jax
import jax.numpy as jnp
import sys

Array = jnp.ndarray

sys.path.append('./')

from diagnostics import print_linblad_info

def paulis(dtype=jnp.complex64):
    '''Creates single-qubit basis operators'''
    sx = jnp.array([[0., 1.],[1., 0.]], dtype=dtype)
    sy = jnp.array([[0., -1j],[1j, 0.]], dtype=dtype)
    sz = jnp.array([[1., 0.],[0., -1.]], dtype=dtype)
    id2 = jnp.eye(2, dtype=dtype)
    return sx, sy, sz, id2

def kron_n(ops):
    '''Tensor product of a list of operators'''
    out = ops[0]
    for A in ops[1:]: out = jnp.kron(out, A)
    return out

def get_theta_shape(L: int, hamiltonian_type: str) -> int:
    '''Returns size of parameter list depending on the model chosen'''
    if hamiltonian_type == "uniform_xyz":
        return 6
    elif hamiltonian_type == "general_local_zz":
        return 2*L + (L-1)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
def uniform_xyz_basis(L, sx, sy, sz, id2, dtype=jnp.complex64):
    '''Create list of operators for uniform H with local x,y,z and NN xx,yy,zz'''
    ops_out = []
    dim = 2**L
    #Append all interactions first
    for pauli in [sx, sy, sz]:
        H_term = jnp.zeros((dim, dim), dtype=dtype)
        for i in range(L - 1):
            ops = [id2] * L
            ops[i] = pauli
            ops[i+1] = pauli
            H_term = H_term + kron_n(ops)
        ops_out.append(H_term)

    #Append local fields second
    for pauli in [sx, sy, sz]:
        H_term = jnp.zeros((dim, dim), dtype=dtype)
        for i in range(L):
            ops = [id2] * L
            ops[i] = pauli
            H_term = H_term + kron_n(ops)
        ops_out.append(H_term)
    
    return ops_out

def general_local_zz_basis(L, sx, sy, sz, id2, dtype=jnp.complex64):
    '''Creates operator list for inhomogeneous H with local x,z and NN zz'''
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


def build_xyz_basis(L: int, hamiltonian_type: str = "uniform_xyz", dtype=jnp.complex64):
    '''Generates list of basis operators that will be present in the Hamiltonian.
        It chooses between the available models. It's written like this so that it
        Can be expanded to accomodate more models'''
    sx, sy, sz, id2 = paulis(dtype)
    dim = 2**L

    if hamiltonian_type == "uniform_xyz":
        ops_out = uniform_xyz_basis(L, sx, sy, sz, id2, dtype)
    
    elif hamiltonian_type == "general_local_zz":
        ops_out = general_local_zz_basis(L, sx, sy, sz, id2, dtype)
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

    return ops_out

def get_theta_true_from_config(config: dict) -> Array:
    '''Returns Hamiltonian parameters defined in the configuration file.
        It chooses between the different available models'''
    
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    
    if hamiltonian_type == "uniform_xyz":
        theta_true = jnp.array([
            config["Jx_true"], config["Jy_true"], config["Jz_true"],
            config["hx_true"], config["hy_true"], config["hz_true"]
        ], dtype=jnp.float32)
    
    elif hamiltonian_type == "general_local_zz":
        L = config["L"]
        hx_list = config.get("hx_list_true", [0.0] * L)
        hz_list = config.get("hz_list_true", [0.0] * L)
        Jzz_list = config.get("Jzz_list_true", [0.0] * (L-1))
        
        if len(hx_list) != L or len(hz_list) != L or len(Jzz_list) != L-1:
            raise ValueError("Parameter list lengths don't match L")
        
        theta_true = jnp.array(
            list(hx_list) + list(hz_list) + list(Jzz_list),
            dtype=jnp.float32
        )
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")
    
    return theta_true


def xyz_hamiltonian_from_theta(L: int, theta: Array, OPS_XYZ: list, 
                               hamiltonian_type: str = "uniform_xyz") -> Array:
    '''Creates Hamiltonian from list of operators and corresponding weights'''

    expected_shape = get_theta_shape(L, hamiltonian_type)
    
    if len(theta) != expected_shape or len(OPS_XYZ) != expected_shape:
        raise ValueError(f"Parameter/operator count mismatch")
    
    H = jnp.zeros((2**L, 2**L), dtype=jnp.complex64)
    for i in range(expected_shape):
        H += theta[i] * OPS_XYZ[i]
    
    return H

def build_lindblad_operators(L: int, T1_list: list, T2_list: list, 
                                       dtype=jnp.complex64):
    """
    Build Lindblad operators with per-qubit noise rates.
    
    Args:
        L: Number of qubits
        T1_list: List of T1 times (length L)
        T2_list: List of T2 times (length L)
    
    Returns:
        operators: List of jump operators
        rates: List of corresponding rates
    """
    if len(T1_list) != L or len(T2_list) != L:
        raise ValueError(f"T1_list and T2_list must have length L={L}")
    
    sx, sy, sz, id2 = paulis(dtype)
    sigma_minus = (sx - 1j * sy) / 2.0
    
    operators = []
    rates = []
    
    for i in range(L):
        T1 = T1_list[i]
        T2 = T2_list[i]
        
        # Decay rate for qubit i
        gamma_decay = 1.0 / T1 if T1 > 0 else 0.0
        
        # Dephasing rate for qubit i
        gamma_dephase = max(1.0 / T2 - 1.0 / (2.0 * T1), 0.0) if T1 > 0 and T2 > 0 else 0.0
        
        # Decay operator
        if gamma_decay > 0:
            ops_decay = [id2] * L
            ops_decay[i] = sigma_minus
            operators.append(kron_n(ops_decay))
            rates.append(gamma_decay)
        
        # Dephasing operator
        if gamma_dephase > 0:
            ops_dephase = [id2] * L
            ops_dephase[i] = sz / jnp.sqrt(2.0)
            operators.append(kron_n(ops_dephase))
            rates.append(gamma_dephase)
    
    return operators, rates


def prepare_initial_state(L: int, kind: str, as_density_matrix=False) -> Array:
    dim = 2**L
    if kind == "all_zeros":
        psi = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    elif kind == "all_plus":
        amp = 1.0 / jnp.sqrt(dim)
        psi = jnp.full((dim,), amp, dtype=jnp.complex64)
    else:
        raise ValueError(f"Unknown initial state kind: {kind}")
    
    if as_density_matrix:
        psi = psi.reshape(-1, 1)
        rho = psi @ psi.conj().T
        return rho
    else:
        return psi
    

def schrodinger_rhs(t, psi, params):
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    return -1j * (H @ psi)

def lindblad_rhs(t, rho, params):
    L = params["L"]
    hamiltonian_type = params.get("hamiltonian_type", "uniform_xyz")
    H = xyz_hamiltonian_from_theta(L, params["theta"], params["ops_xyz"], hamiltonian_type)
    
    # Hamiltonian evolution
    drho = -1j * (H @ rho - rho @ H)
    
    # Lindblad dissipators
    jump_ops = params["jump_operators"]
    rates = params["jump_rates"]
    
    for L_op, gamma in zip(jump_ops, rates):
        if gamma > 0:
            L_dag = L_op.conj().T
            L_dag_L = L_dag @ L_op
            drho += gamma * (L_op @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))
    
    return drho
    

def define_dynamics(config, theta_true, params_true):

    L = config["L"]
    dynamics_type = config.get("dynamics_type", "schrodinger")
    hamiltonian_type = config.get("hamiltonian_type", "uniform_xyz")
    noise_model = config.get("noise_model", "global")

    if dynamics_type == "schrodinger":

        rhs_fun = schrodinger_rhs
        print(f"Using Schrödinger dynamics (noiseless)")
        print(f"Hamiltonian: {hamiltonian_type} ({len(theta_true)} params)")
        
    elif dynamics_type == "lindblad":

        # Build Lindblad operators based on noise model
        if noise_model == "global":
            T1_list = [config.get("T1_global", 10.0)]*L
            T2_list = [config.get("T2_global", 5.0)]*L

        elif noise_model == "local":
            T1_list = config.get("T1_list", [10.0] * L)
            T2_list = config.get("T2_list", [5.0] * L)

        else:
            raise ValueError(f"Unknown noise_model: {noise_model}")
            
        if len(T1_list) != L or len(T2_list) != L:
            raise ValueError(f"T1_list and T2_list must have length L={L}")
            
        jump_ops, jump_rates = build_lindblad_operators(L, T1_list, T2_list)
        print_linblad_info(L, T1_list, T2_list, noise_model)

        
        params_true["jump_operators"] = jump_ops
        params_true["jump_rates"] = jump_rates
        rhs_fun = lindblad_rhs
        print(f"Hamiltonian: {hamiltonian_type} ({len(theta_true)} params)")
        print(f"Jump operators: {len(jump_ops)}")

    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
    
    return rhs_fun, params_true