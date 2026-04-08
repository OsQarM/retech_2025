

import jax
import jax.numpy as jnp
import numpy as np
import sys
from typing import Optional

Array = jnp.ndarray

sys.path.append('./')

from model_building import paulis, kron_n


class OperatorClass:
    '''Will class that contains a list of all the operator types the Hamiltonian will have
       The operators will be applied to each qubit, and we will allow for the construction of any
       combination of Pauli strings 
    '''

    def __init__(self, L, dtype=jnp.complex64):

        self.L = L
        self.dim = 2**L
        self.pauli_basis = {}
        self.pauli_basis['X'], self.pauli_basis['Y'], self.pauli_basis['Z'], self.pauli_basis['I'] = paulis(dtype)
        self.operators = []
    
    def __len__(self):
        return len(self.operators)
    
    def add_operators(self, pauli_string:str):
        #e.g. 'X','Y','ZZ'
        '''Adds one type of operator at a time. It loops through all the qubits, 
        and for each position does the tensor product of the whole chain, with the 
        required qubits substituted by the operators of the string'''
        #Note: I could refactor it to take a list of strings and make add all the operators together
        for i in range(self.L - len(pauli_string) + 1):
                #Create identity operators for each qubit
                ops = [self.pauli_basis['I']]*self.L
                for j, char in enumerate(pauli_string):
                     #Build string
                     ops[i+j] = self.pauli_basis[char]
                self.operators.append(kron_n(ops))
        print(f"{pauli_string} terms added to the Hamiltonian")
        
                     
                    

                







