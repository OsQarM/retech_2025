from qilisdk.backends import QutipBackend
from qilisdk.digital import Circuit, M, U1, CNOT, U2, U3, CZ, RX, RZ, H
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.optimizers import SciPyOptimizer
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.functionals.sampling import Sampling, SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from scipy.optimize import minimize

def generate_connectivity_list(size, mode = 'ATA', boundary = 'open'):
    connectivity = []
    if mode == 'ATA':
        for i in range(size):
            for j in range(i+1, size):    
                connectivity.append([i,j])

    elif mode == 'NN':
        for i in range(size-1):
            connectivity.append([i, i+1])

    if boundary == 'closed' and size > 2:
        connectivity.append([size-1, 0])

    return connectivity

# Build a parameterized circuit ansatz:
def build_ansatz_circuit(params, nqubits, layers, connectivity):
    """
    params layout:
      - per layer:
         - for each qubit: RX angle, RZ angle  -> 2*nqubits
         - for each entangler pair: CRZ angle -> len(connectivity)
      total params per layer = 2*nqubits + len(connectivity)
    params should be a flat 1D array of length layers * (2*nqubits + len(connectivity))
    """
    per_layer = 2 * nqubits + len(connectivity)
    assert params.size == layers * per_layer, "params size mismatch"
    c = Circuit(nqubits=nqubits)
    p = params.reshape((layers, per_layer))
    for L in range(layers):
        # single-qubit rotations
        for q in range(nqubits):
            rx_angle = p[L, 2*q]
            rz_angle = p[L, 2*q + 1]
            c.add(RX(q, theta=rx_angle))
            c.add(RZ(q, phi=rz_angle))
        # entanglers: controlled-RZ via decomposition
        ent_start = 2*nqubits
        for k, (control, target) in enumerate(connectivity):
            lam = p[L, ent_start + k]
            c.add(RZ(target, phi=lam).controlled(control))

    return c