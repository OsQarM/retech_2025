import torch

class OperatorClass:
    '''Class that contains a list of all the operator types the Hamiltonian will have
       The operators will be applied to each qubit, and we will allow for the construction of any
       combination of Pauli strings 
    '''
    def __init__(self, L, topology = "linear", dtype=torch.complex64):

        self.L = L
        self.topology = topology # "linear" or "star"
        self.dim = 2**L
        self.pauli_basis = {}
        self.pauli_basis['X'], self.pauli_basis['Y'], self.pauli_basis['Z'], self.pauli_basis['I'] = paulis(dtype)
        self.operators = []
    
    def __len__(self):
        return len(self.operators)
    
    def __getitem__(self, idx):
        return self.operators[idx]
    
    def add_operators(self, pauli_string:str):
        #e.g. 'X','Y','ZZ'
        '''Adds one type of operator at a time. It loops through all the qubits, 
        and for each position does the tensor product of the whole chain, with the 
        required qubits substituted by the operators of the string'''

        if len(pauli_string) > self.L:
            raise ValueError(f"Pauli string '{pauli_string}' longer than system size {self.L}")
        
        if not all(char in 'XYZI' for char in pauli_string):
            raise ValueError(f"Invalid character in '{pauli_string}'. Use only X, Y, Z, I")
        
        for i in range(self.L - len(pauli_string) + 1):
                #Create identity operators for each qubit
                ops = [self.pauli_basis['I']]*self.L

    
                if self.topology == "linear":
                    for j, char in enumerate(pauli_string):
                           #Build string starting at i.
                        #At each iterations the string displaces one site starting at i.
                        #At each iterations the string displaces one site
                           ops[i+j] = self.pauli_basis[char]

                elif self.topology == "star":
                    if len(pauli_string) == 1:
                        # Single-qubit: place on qubit i (same as linear)
                        ops[i] = self.pauli_basis[pauli_string[0]]
                    else:
                        #Puts first operator in center
                        ops[0] = self.pauli_basis[pauli_string[0]]
                        #Starts putting in qubits around. 
                        # As i increases it skips the qubits where operators have been placed
                        for j, char in enumerate(pauli_string[1:]):
                            ops[i+1+j] = self.pauli_basis[char]

                else:
                    raise ValueError(f"Topology {self.topology} invalid. Choose 'linear' or 'star'.")


                self.operators.append(kron_n(ops))
        print(f"{pauli_string} terms added to the Hamiltonian")


def prepare_initial_state(L, kind, dtype=torch.complex64):
    """Prepare initial quantum states for L qubits."""
    if kind == 'all_zeros':
        psi0 = torch.zeros(2**L, dtype=dtype)
        psi0[0] = 1.0
        
    elif kind == 'all_plus':
        plus = torch.ones(2, dtype=dtype) / np.sqrt(2)
        psi0 = plus
        for _ in range(L - 1):
            psi0 = torch.kron(psi0, plus)
            
    else:
        raise ValueError(f"Initial state '{kind}' not recognized. "
                        f"Use 'all_zeros' or 'all_plus'")
    return psi0

def paulis(dtype=torch.complex64, requires_grad=False):
    '''Creates single-qubit basis operators'''
    sx = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype, requires_grad=requires_grad)
    sy = torch.tensor([[0., -1j], [1j, 0.]], dtype=dtype, requires_grad=requires_grad)
    sz = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, requires_grad=requires_grad)
    id2 = torch.eye(2, dtype=dtype, requires_grad=requires_grad)
    return sx, sy, sz, id2

def kron_n(ops):
    '''Tensor product of a list of operators'''
    out = ops[0]
    for A in ops[1:]:
        out = torch.kron(out, A)
    return out

def rk4_step(state, H, t, dt, rhs_fun):
    dt_c = torch.asarray(dt, dtype=state.dtype)
    k1 = rhs_fun(H, t, state)
    k2 = rhs_fun(H, t + 0.5*dt_c, state + 0.5*dt_c*k1)
    k3 = rhs_fun(H, t + 0.5*dt_c, state + 0.5*dt_c*k2)
    k4 = rhs_fun(H, t + dt_c, state + dt_c*k3)
    state_next = state + (dt_c/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    
    if state.ndim == 1:  # State vector
        norm = torch.linalg.norm(state_next)
        return state_next / (norm + 1e-12)
    else:  # Density matrix
        state_next = 0.5 * (state_next + state_next.conj().T)
        trace = torch.trace(state_next).real
        return state_next / (trace + 1e-12)
    

def build_hamiltonian(L, theta, OPS_LIST):
    '''Creates Hamiltonian from list of operators and corresponding weights'''

    expected_shape = len(OPS_LIST)
    
    if len(theta) != expected_shape or len(OPS_LIST) != expected_shape:
        raise ValueError(f"Parameter/operator count mismatch")
    
    theta_complex = theta.to(torch.complex64)
    
    H = torch.zeros((2**L, 2**L), dtype=torch.complex64)
    for i in range(expected_shape):
        H += theta_complex[i] * OPS_LIST.operators[i]
    
    return H


def schrodinger_rhs(H, t, psi):
    return -1j * (H @ psi)


def evolve_state(psi_t, H, t_grid):
    rhs_fun = schrodinger_rhs
    dt = t_grid[1] - t_grid[0]
    for i,t in enumerate(t_grid[0:-1]):
        dt = t_grid[i+1] - t_grid[i]
        psi_t = rk4_step(psi_t, H, t, dt, rhs_fun)
 
    return psi_t


def time_evolution(psi, theta, OPS_LIST, L, t_grid):
    H = build_hamiltonian(L, theta, OPS_LIST)
    psi_t = evolve_state(psi, H, t_grid)

    return psi_t


def physics_computation(params, psi0, OPS_LIST, CONFIG, t_grid):

    psi_t = time_evolution(psi0, params['theta'], OPS_LIST, CONFIG['L'], t_grid)
    
    return psi_t