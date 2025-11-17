import qutip as qt
import numpy as np
from qilisdk.analog import Schedule, X, Y, Z


sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

class QiliSDKHamiltonian():
    
    def __init__(self,size):
        self.size = size
        self.H = 0
    
    def add_local_field(self, qubit, field, weight=1.):

        assert qubit <= self.size-1, f"Index {qubit} out of range. Size of the system is {self.size}"
        
        self.H += weight*field(qubit)

    def add_ZZ_term(self, qubit1, qubit2, weight=1.):

        assert qubit1<= self.size-1, f"Index {qubit1} out of range. Size of the system is {self.size}"
        assert qubit2 <= self.size-1, f"Index {qubit2} out of range. Size of the system is {self.size}"
        assert qubit1 != qubit2,    f"Qubits 1 and 2 are the same, cannot apply self-interaction term"

        self.H += weight*Z(qubit1)*Z(qubit2)




class QutipHamiltonian():

    def __init__(self,size):
        self.size = size
        self.H = 0 
        self.sx_list, self.sy_list, self.sz_list = self._initialize_operators()

    def _initialize_operators(self):
        '''Setup operators for individual qubits
           for each value of i it puts the paulis in different positions of the list, 
           then does IxIxI...sigma_ixIxI...xI
        '''
        sx_list, sy_list, sz_list = [], [], []
        for i in range(self.size):
            #list of 2x2 identity matrices
            op_list = [qt.qeye(2)] * self.size
            #replace i-th element with sigma_x
            op_list[i] = sx
            #create matrices of 2^Nx2^N
            sx_list.append(qt.tensor(op_list))
            #do the same for sigma_y and sigma_z
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list 
    
    
    def add_x_field(self, qubit, weight=1.):

        assert qubit <= self.size-1, f"Index {qubit} out of range. Size of the system is {self.size}"
        self.H += weight*self.sx_list[qubit]

    def add_z_field(self, qubit, weight=1.):

        assert qubit <= self.size-1, f"Index {qubit} out of range. Size of the system is {self.size}"
        self.H += weight*self.sz_list[qubit]

    def add_ZZ_term(self, qubit1, qubit2, weight=1.):

        assert qubit1<= self.size-1, f"Index {qubit1} out of range. Size of the system is {self.size}"
        assert qubit2 <= self.size-1, f"Index {qubit2} out of range. Size of the system is {self.size}"
        assert qubit1 != qubit2,    f"Qubits 1 and 2 are the same, cannot apply self-interaction term"

        self.H += weight*self.sz_list[qubit1]*self.sz_list[qubit2]

    


def create_random_hamiltonian(Nqubits, min_weight, max_weight):
    H_out = QutipHamiltonian(Nqubits)
    single_x_weights = np.random.uniform(min_weight, max_weight, size=Nqubits)
    single_z_weights = np.random.uniform(min_weight, max_weight, size=Nqubits)

    interaction_weights = np.random.uniform(min_weight, max_weight, size=int(Nqubits*(Nqubits-1)/2))
    interaction_counter = 0
    for i in range(Nqubits):
        H_out.add_x_field(i, single_x_weights[i])
        H_out.add_z_field(i, single_z_weights[i])

        for j in range(i+1, Nqubits):
            H_out.add_ZZ_term(i, j, interaction_weights[interaction_counter])
            interaction_counter+=1

    all_weights = np.concatenate([single_x_weights,single_z_weights,interaction_weights])
    return H_out, all_weights
    

    
