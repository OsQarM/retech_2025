import sys
import numpy as np
import qutip as qt

sys.path.append('/Users/omichel/Desktop/qilimanjaro/projects/retech/retech_2025/src')

import estimator
import hamiltonian
import dynamics

from qilisdk.backends import QutipBackend
from qilisdk.digital import Circuit, M, U1, CNOT, U2, U3, CZ, RX, RZ, H
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.optimizers import SciPyOptimizer
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.functionals.sampling import Sampling, SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from scipy.optimize import minimize

from qilisdk.analog import Schedule, X, Z, Y
from qilisdk.core import ket, tensor_prod
from qilisdk.backends import QutipBackend, CudaBackend
from qilisdk.functionals import TimeEvolution


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



def simulate_annealing(T, dt, Hx, Ht, initial_state):
    # Build a time‑dependent schedule
    schedule = Schedule(T, dt)
    
    # Add hx with a time‐dependent coefficient function
    schedule.add_hamiltonian(label="hx", hamiltonian=Hx, schedule = lambda t: 1 - t / T)
    schedule.add_hamiltonian(label="ht", hamiltonian=Ht.H, schedule = lambda t: t / T)
    
    
    # Create the TimeEvolution functional
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=initial_state,
        observables=[Z(0), X(0), Y(0)],
        nshots=0,
        store_intermediate_results=False,
    )
    
    # Execute on Qutip backend and inspect results
    backend = QutipBackend()
    results = backend.execute(time_evolution)
    return results


def fidelity_cost(params, nqubits, layers, connectivity, true_probabilities, backend=QutipBackend):
    circuit = build_ansatz_circuit(np.array(params), nqubits, layers, connectivity)
    circuit_simulation = backend.execute(functional=Sampling(circuit, nshots = 10000))
    loss = 1 - estimator.classical_fidelity(true_probabilities, circuit_simulation.probabilities)
    return float(loss)


def nll_cost(params, nqubits, psi_0, ti, tf_list, nsteps, timestamp_measurements):
    H_ansatz = hamiltonian.create_hamiltonian_from_weights(nqubits, np.array(params), backend='qutip')
    loss = 0
    for i, time in enumerate(timestamp_measurements):
        sim = dynamics.time_evolution(H_ansatz, psi_0, ti, tf_list[i], nsteps)
        loss += estimator.nll(sim.states[-1], timestamp_measurements[i])/len(timestamp_measurements)
    return float(loss)


def annealing_cost(params, times, dt, Hx, nqubits, initial_state, target_state_list):

    Ht = hamiltonian.create_hamiltonian_from_weights(nqubits, params) #target
    loss = 0
    for i, T in enumerate(times):
        sim = simulate_annealing(T, dt, Hx, Ht, initial_state)

        final_state_data = sim.final_state.data
        final_state_array = final_state_data.toarray()
        final_qutip_state = qt.Qobj(final_state_array, dims=[[2]*nqubits, [1]])

        loss += (1 - qt.fidelity(target_state_list[i], final_qutip_state))/len(target_state_list)

    return float(loss)

