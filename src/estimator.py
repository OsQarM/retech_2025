import numpy as np
import qutip as qt
from typing import List, Tuple



def classical_fidelity(prob_dict1, prob_dict2):
    """Classical fidelity between two probability distributions stored as dictionaries"""
    # Get all unique keys from both dictionaries (the ones missing have value 0)
    all_keys = set(prob_dict1.keys()) | set(prob_dict2.keys())
    
    # Extract values for all keys, using 0 for missing keys
    prob_array1 = np.array([prob_dict1.get(key, 0.0) for key in all_keys])
    prob_array2 = np.array([prob_dict2.get(key, 0.0) for key in all_keys])
    
    return np.sum(np.sqrt(prob_array1 * prob_array2))



class NegativeLogLikelihoodEstimator:
    def __init__(self):
        pass
    
    #first 2 functions just generate dataset of exact state

    def generate_measurement_dataset(self, true_state: qt.Qobj, 
                                   pauli_basis: str = 'Z',
                                   M: int = 1000) -> List[int]:
        """
        Generate measurement dataset by sampling from true state marginals
        
        Args:
            true_state: The true time-evolved state |ψ_T(t)⟩
            pauli_basis: Pauli basis to measure in ('X', 'Y', 'Z')
            M: Number of measurement repeats
        
        Returns:
            List of binary measurement outcomes
        """
        # Get measurement probabilities for the chosen basis
        probs = self.get_measurement_probabilities(true_state, pauli_basis)
        
        # Sample M measurement outcomes from the probability distribution
        n_qubits = len(true_state.dims[0])
        possible_bitstrings = list(range(2**n_qubits))
        outcomes = np.random.choice(possible_bitstrings, size=M, p=probs)
        
        return outcomes.tolist()
    
    def get_measurement_probabilities(self, state: qt.Qobj, basis: str) -> List[float]:
        """
        Get measurement probabilities for a given state and Pauli basis
        """
        if basis == 'Z':
            proj0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # |0⟩⟨0|
            proj1 = qt.basis(2, 1) * qt.basis(2, 1).dag()  # |1⟩⟨1|
        elif basis == 'X':
            plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
            proj0 = plus * plus.dag()  # |+⟩⟨+|
            proj1 = minus * minus.dag()  # |-⟩⟨-|
        elif basis == 'Y':
            plus_i = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
            minus_i = (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit()
            proj0 = plus_i * plus_i.dag()  # |+i⟩⟨+i|
            proj1 = minus_i * minus_i.dag()  # |-i⟩⟨-i|
        
        # Handle multi-qubit states
        if len(state.dims[0]) > 1:
            # For simplicity, measure all qubits in the same basis
            # You can modify this to measure specific qubits
            n_qubits = len(state.dims[0])
            probabilities = []
            
            # Calculate probability for each possible bitstring
            #loops over number of bitstrings
            for i in range(2**n_qubits): 
                # Create computational basis state |i⟩
                #creates binary format of index
                bitstring = format(i, f'0{n_qubits}b')
                #creates qutip state with this bitstring
                basis_state = qt.tensor([qt.basis(2, int(bit)) for bit in bitstring])
                
                # Transform to measurement basis if needed
                if basis != 'Z':
                    if basis == 'X':
                        H = (1/np.sqrt(2)) * qt.Qobj([[1, 1], [1, -1]])
                    elif basis == 'Y':
                        H = (1/np.sqrt(2)) * qt.Qobj([[1, 1], [1j, -1j]])
                    
                    # Apply basis change to all qubits
                    basis_change = qt.tensor([H] * n_qubits)
                    basis_state = basis_change.dag() * basis_state
                
                # Calculate probability using Born rule
                #compare bitstring with supplied state (important to note, this does not sample, it gives the exact probability)
                prob = abs((basis_state.dag() * state)) ** 2
                probabilities.append(prob)
            
            return probabilities
        else:
            # Single qubit case simples as there are only 2 possible bitstrings
            p0 = qt.expect(proj0, state)
            p1 = qt.expect(proj1, state)
            return [p0, p1]
        
    
    #from here on we have the actual loss function
    
    def compute_bitstring_probability(self, estimated_state: qt.Qobj, 
                                    bitstring: int, 
                                    basis: str = 'Z') -> float:
        """
        Compute probability of observing a bitstring from the estimated state
        
        Args:
            estimated_state: The estimated state |ψ(t; θ, φ)⟩
            bitstring: Measurement outcome (as integer or binary string)
            basis: Pauli basis in which measurement was performed
        
        Returns:
            Probability p(b|ψ_θ) = |⟨b|ψ_θ⟩|²
        """
        n_qubits = len(estimated_state.dims[0])
        
        # Convert bitstring to proper format
        if isinstance(bitstring, int):
            bitstring_bin = format(bitstring, f'0{n_qubits}b')
        else:
            bitstring_bin = bitstring
        
        # Create computational basis state corresponding to the bitstring
        basis_vectors = []
        for bit in bitstring_bin:
            basis_vectors.append(qt.basis(2, int(bit)))
        computational_basis_state = qt.tensor(basis_vectors)
        
        # Transform to measurement basis if needed
        if basis != 'Z':
            if basis == 'X':
                # Hadamard basis change
                H = (1/np.sqrt(2)) * qt.Qobj([[1, 1], [1, -1]])
                basis_change = qt.tensor([H] * n_qubits)
                measurement_basis_state = basis_change.dag() * computational_basis_state
            elif basis == 'Y':
                # Y-basis change
                Y_basis = (1/np.sqrt(2)) * qt.Qobj([[1, 1], [1j, -1j]])
                basis_change = qt.tensor([Y_basis] * n_qubits)
                measurement_basis_state = basis_change.dag() * computational_basis_state
        else:
            measurement_basis_state = computational_basis_state
        
        # Calculate probability using Born rule: |⟨b|ψ_θ⟩|²
        amplitude = (measurement_basis_state.dag() * estimated_state)
        probability = abs(amplitude) ** 2
        
        return probability
    
    def negative_log_likelihood(self, estimated_state: qt.Qobj, 
                              dataset: List[int],
                              basis: str = 'Z') -> float:
        """
        Compute negative log-likelihood loss function
        
        L(|ψ_θ⟩; D) = -Σ_{b∈D} log |⟨b|ψ_θ⟩|²
        
        Args:
            estimated_state: The estimated state |ψ(t; θ, φ)⟩
            dataset: List of measurement outcomes (bitstrings)
            basis: Pauli basis in which measurements were performed
        
        Returns:
            Negative log-likelihood value
        """
        total_loss = 0.0
        epsilon = 1e-10  # Small value to avoid log(0)
        
        for bitstring in dataset:
            # Compute probability of observing this bitstring
            prob = self.compute_bitstring_probability(estimated_state, bitstring, basis)
            
            # Add to negative log-likelihood
            total_loss -= np.log(prob + epsilon)
        
        # Average over the dataset
        average_loss = total_loss / len(dataset)
        
        return average_loss
    

