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