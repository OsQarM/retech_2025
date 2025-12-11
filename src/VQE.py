import sys
import numpy as np
import qutip as qt
from scipy.optimize import minimize

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


def simulate_annealing(T, dt, tf, steps, Hx, Ht, initial_state):
    # Build a time‑dependent schedule
    schedule = Schedule(
        T=T,
        dt=dt,
        hamiltonians={"driver": Hx, "problem": Ht},
        schedule={i: {"driver": 1 - t /tf, "problem": t /tf} for i, t in enumerate(steps)},
    )
    
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


def learn_hamiltonian(nqubits, weights_i, H_driver, initial_state, target_states, times, connectivity = "all-to-all", alpha = 1., observables = None, optimizer_options = None, mode = "complete"):
    if mode == "complete":
        result = minimize_cost(nqubits, weights_i, H_driver, initial_state, target_states, times, connectivity, alpha, observables, optimizer_options)

    elif mode == "segmented":
        for i,t in enumerate(times):
            tf = [t]
            target_state = [target_states[i]]
            result = minimize_cost(nqubits, weights_i, H_driver, initial_state, target_state, tf, alpha, observables, optimizer_options, connectivity)
            weights_i = result.x

    return result


def minimize_cost(nqubits, initial_weights, H_driver, initial_state, target_states, times, alpha, observables, optimizer_options, connectivity = 'all-to-all'):
    res = minimize(
    fun=lambda p: annealing_cost(p, times, H_driver, nqubits, initial_state, target_states, alpha, observables, connectivity),
    x0=initial_weights,
    method='L-BFGS-B',
    jac=lambda p: estimator.parameter_shift_grad(
        lambda x: annealing_cost(x, times, H_driver, nqubits, initial_state, target_states, alpha, observables, connectivity),
        p
    ),
    options=optimizer_options
)
    return res

def annealing_cost(params, times, Hx, nqubits, initial_state, target_state_list, alpha, target_observables, connectivity = 'all-to-all'):

    Ht = hamiltonian.create_hamiltonian_from_weights(nqubits, params, connectivity) #target
    loss = 0

    target_x = target_observables[0]
    target_y = target_observables[1]
    target_z = target_observables[2]
    #final time of annealing schedule where H = Hp (sometimes we want to simulate only part of the annealing)
    T_final = times[-1]
    for i, T in enumerate(times):
        dt = T/100
        steps = np.linspace(0, T, int(T / dt))
        sim = simulate_annealing(T, dt, T_final, steps, Hx, Ht.H, initial_state)

        final_state_data = sim.final_state.data
        final_state_array = final_state_data.toarray()
        final_qutip_state = qt.Qobj(final_state_array, dims=[[2]*nqubits, [1]])

        final_x = sim.final_expected_values[0]
        final_y = sim.final_expected_values[1]
        final_z = sim.final_expected_values[2]

        fidelity_loss = 1 - qt.fidelity(target_state_list[i], final_qutip_state)
        observable_loss = (abs((final_x - target_x[i])) +
                           abs((final_y - target_y[i])) +
                           abs((final_z - target_z[i]))
                           )/3.0 

        # divide by 2 because it can take a maximum of 2
        loss += alpha*fidelity_loss + (1-alpha)*observable_loss
    return float(loss)


def nll_cost(params, nqubits, psi_0, ti, tf_list, nsteps, timestamp_measurements):
    H_ansatz = hamiltonian.create_hamiltonian_from_weights(nqubits, np.array(params), backend='qutip')
    loss = 0
    for i, time in enumerate(timestamp_measurements):
        sim = dynamics.time_evolution(H_ansatz, psi_0, ti, tf_list[i], nsteps)
        loss += estimator.nll(sim.states[-1], timestamp_measurements[i][0], basis='z')/len(timestamp_measurements)
        loss += estimator.nll(sim.states[-1], timestamp_measurements[i][1], basis='x')/len(timestamp_measurements)
    return float(loss)

def inverse_nll_cost(params, nqubits, psi_0, ti, tf_list, nsteps, nshots, target_states):
    #build ansatz H from variational parameters
    H_ansatz = hamiltonian.create_hamiltonian_from_weights(nqubits, np.array(params), backend='qutip')
    loss = 0

    for i, state in enumerate(target_states):
        #simulate dynamics
        sim = dynamics.time_evolution(H_ansatz, psi_0, ti, tf_list[i], nsteps)
        #sampling
        measurements_z = dynamics.sample_from_state(sim.states[-1], nqubits, nshots, basis='z')
        measurements_x = dynamics.sample_from_state(sim.states[-1], nqubits, nshots, basis='x')
        #loss calculation
        loss += estimator.nll(state, measurements_z, basis='z')/len(target_states)
        loss += estimator.nll(state, measurements_x, basis='x')/len(target_states)
    return float(loss)






