import qutip as qt
import numpy as np

def initialize_down_state(size):

    '''Initialize product state of all qubits down'''

    register = [qt.basis(2, 0)]*size
    prod_state = qt.tensor(register)
    return prod_state

def time_evolution(Ham, initial_state, initial_time, final_time, timesteps, progessbar = None):
    '''
    Time evolution of a state under a given Hamiltonian

    :Ham:(Hamiltonian class object) Hamiltonian under which we evolve the system
    :initial_state:(qt.state) state of the chain at time 0
    :initial_time, final_time:(float) self-explanatory
    :timesteps:(float) number of simulation steps between the two times 
    '''
    hamiltonian_object = Ham.H
    times = np.linspace(initial_time, final_time, timesteps)
    #apply hamiltonian to initial state and don't track any observables

    options = {'method': 'adams'}

    if progessbar:
        options = {
        'method': 'adams', 
        'progress_bar': 'tqdm'
        }

    simulation_results = qt.sesolve(hamiltonian_object, initial_state, times, options = options)

    return simulation_results



def calculate_expectation_values(state_evolution, Hamiltonian):
    """
    Calculate expectation value of spin X,Y,Z on time evolution of chain

    Args:
        state_evolution: result of qutip.sesolve of length Nsteps
        Hamiltonian: qutip object containing properties of transport Hamiltonian

    Returns:
        magn_t: Dictionary containing result of each observable for each qubit and each timestep
    """   

    sigma_x_list = Hamiltonian.sx_list
    sigma_y_list = Hamiltonian.sy_list
    sigma_z_list = Hamiltonian.sz_list

    magn_t = {}

    #calculate expectation value of sz for each spin
    magn_t["Sx"] = calculate_observable_along_chain(state_evolution, sigma_x_list)
    magn_t["Sy"] = calculate_observable_along_chain(state_evolution, sigma_y_list)
    magn_t["Sz"] = calculate_observable_along_chain(state_evolution, sigma_z_list)

    return magn_t


def calculate_observable_along_chain(state_evolution, observable):
    """
    Calculates a given observable over every spin of the chain and every timestep of simulation

    Args:
    state_evolution: result of qutip.sesolve of length Nsteps
    observable: list of operators where each index applies an observable on a chain qubit

    Returns: 
    2D array of the results for each timestep and qubit
    """
    return np.array([[qt.expect(op, state) 
                        for op in observable] 
                       for state in state_evolution.states])



def sample_measurements(state, measurement_basis, num_shots=1000):
    """
    Simulate multiple measurements of a quantum state in a given basis
    
    Args:
        state: Quantum state (ket or density matrix)
        measurement_basis: List of projection operators for the basis
        num_shots: Number of measurement shots
    
    Returns:
        outcomes: Array of measurement outcomes (0, 1, 2, ...)
        probabilities: Theoretical probabilities for each outcome
    """
    # Calculate probabilities for each measurement outcome
    probabilities = [qt.expect(proj, state) for proj in measurement_basis]
    
    # Sample from multinomial distribution
    outcomes = np.random.choice(len(measurement_basis), size=num_shots, p=probabilities)
    
    return outcomes, probabilities


def measure_qubit(state, qubit_index, basis='Z', num_shots=1000):
    """
    Measure a specific qubit in a quantum system (works for both single and multi-qubit)
    """
    # Define basis operators
    if basis == 'Z':
        proj0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # |0⟩⟨0|
        proj1 = qt.basis(2, 1) * qt.basis(2, 1).dag()  # |1⟩⟨1|
    elif basis == 'X':
        plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
        proj0 = plus * plus.dag()  # |+⟩⟨+|
        proj1 = minus * minus.dag()  # |-⟩⟨-|
    elif basis == 'Y':
        plus_i = (qt.basis(2, 0) + 1j*qt.basis(2, 1)).unit()
        minus_i = (qt.basis(2, 0) - 1j*qt.basis(2, 1)).unit()
        proj0 = plus_i * plus_i.dag()  # |+i⟩⟨+i|
        proj1 = minus_i * minus_i.dag()  # |-i⟩⟨-i|
    
    # Handle single qubit case
    if state.dims[0][0] == 2 and len(state.dims[0]) == 1:
        # Single qubit system
        measurement_basis = [proj0, proj1]
    else:
        # Multi-qubit system
        identity_ops = [qt.qeye(2) for _ in range(state.dims[0][0])]
        identity_ops[qubit_index] = proj0
        projector0 = qt.tensor(identity_ops)
        
        identity_ops[qubit_index] = proj1
        projector1 = qt.tensor(identity_ops)
        
        measurement_basis = [projector0, projector1]
    
    return sample_measurements(state, measurement_basis, num_shots)