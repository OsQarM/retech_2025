import numpy as np
import torch
from torchvision import transforms, datasets
import tensorkrowch as tk

import jax
import jax.numpy as jnp

import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import glob
import yaml
import matplotlib.pyplot as plt
import torchtt as tntt 

import sys
import os

sys.path.append('../src/')

from run_inference import run_inference
from plots import bar_plot_strings_comparison, plot_training_loss


'''Script to learn distribution of Multiverse model

This program takes a trained MLP model and an example input. The MLP has one hidden layer that can be compressed into an MPO
The structure of the MLP is:

Input (784)  
→ Dense  
Hidden Layer 1 (1000)  
→ MPO (TenPy)  
Hidden Layer 2 (1000)  
→ Dense  
Output (10)

The program first performs inference with or without the tensorized layer and treats the output parameters as a quantum state,
that is, it encodes the 10 classes in a 4-qubit state. The objective of the Hamiltonian learning is to find which Hamiltonian
Generates this state from the input. 

The Hamiltonian needs to be able to process the input, so we encode the 784 parameters in a 10-qubit quantum state. Then the Hamiltonian
Evolves this state into the output state where only 10 parameters matter (even though this makes most of the information of the output state unnecessary)

These are the steps of the program:

1. Load the MLP and the input
2. Perform inference
3. Create 4-qubit state from output
4. At this point we chan chose between two options:
    a) Train a 4-qubit Hamiltonian from an input of |0000> (made for testing or alternative result in case option b) fails)
    b) Train a 10-qubit Hamiltonian from an input encoded by the MLP example
5. Build input state from list of input parameters (option a) or list of zeros (option b)
6. Create NN model for HL learning with tensorkrowch, taking a random input since it doesn't matter much
7. Train the Network
8. Save results
9. Validate results

'''


##########################################
#1. UTILS
##########################################

def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_filename_core(config, N):
    chi_nn = config['bond_dimension_learning']
    kind = config['learning_mode']
    chi_mpo = config['MAX_MPO_CHI']
    mpo_on = config['MPO_ON']
    
    filename_core = f"L{N}_chi_nn-{chi_nn}_kind-{kind}_Chimpo-{chi_mpo}_MPO-{mpo_on}_class-7"

    return filename_core

def generate_bitstring_list(nqubits):
    '''Create list containing all possible bitstrings of the N-qubit chain'''
    bitstrings = []
    decimal_bitstrings = range(0, 2**nqubits)
    int_bitstrings = [bin(i)[2:].zfill(nqubits) for i in decimal_bitstrings]
    bitstrings =  [str(bit) for bit in int_bitstrings]

    return bitstrings


##########################################
#2. PHYSICS
##########################################

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


def physics_computation(params, psi0, OPS_LIST, L, t_grid):
    #psi_rot = compute_rotations_corrected(psi0, params, L)
    psi_t = time_evolution(psi0, params['theta'], OPS_LIST, L, t_grid)
    return psi_t



#################################################################
#3. INITIAL STATE AND OPERATORS
#################################################################

def min_power_of_2(n):
    """
    Alternative method using bit manipulation (more efficient)
    """
    if n <= 0:
        raise ValueError("Number must be positive")
    
    # If n is already a power of 2, return its exponent
    if (n & (n - 1)) == 0:
        return n.bit_length() - 1
    
    # Otherwise, return the bit length
    return n.bit_length()


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


def initial_state_from_input(L, data, dtype=torch.complex64):

    if 2**L < len(data):
        raise ValueError("Not enough qubits to encode input")

    #Create empty state
    psi0 = torch.zeros(2**L, dtype=dtype)

    #Fill entries until loading all data, leave the rest as zero
    for i in range(len(data)):
        psi0[i] = data[i]
    
    #Normalize and return
    return psi0 / torch.norm(psi0)



class OperatorClass:
    '''Class that contains a list of all the operator types the Hamiltonian will have
       The operators will be applied to each qubit, and we will allow for the construction of any
       combination of Pauli strings 
    '''
    def __init__(self, L, dtype=torch.complex64):

        self.L = L
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
                for j, char in enumerate(pauli_string):
                     #Build string
                     ops[i+j] = self.pauli_basis[char]
                self.operators.append(kron_n(ops))
        print(f"{pauli_string} terms added to the Hamiltonian")


##########################################
#4. NEURAL NETWORK
##########################################


class MPS_MLP(nn.Module):
    def __init__(self, L, chi, num_params, num_dims = []):
        super().__init__()
        self.layers = nn.ModuleList()

        layer_sizes = num_dims + [num_params]

        # 1. MPS input layer: processes L×2 features → first hidden size (will be output size if no middle layers)
        mps = tk.models.MPSLayer(
            n_features=L,
            in_dim=2,
            out_dim=layer_sizes[0],  
            bond_dim=chi
        )
        self.layers.append(mps)
        
        # 2. Middle layers (optional)
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(
                layer_sizes[i],
                layer_sizes[i+1]
            ))
        
        if len(layer_sizes) > 1:
        # 3. Final output layer: hidden → num_params
            self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
        # Custom initialization (similar to your function)
        self._initialize_parameters()
    
    def _initialize_parameters(self, scale=0.1):
        """Initialize weights with normal distribution and biases to zero."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear): #MPSLayer initializes itself
                nn.init.normal_(layer.weight, mean=0.0, std=scale)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Apply all but last layer with tanh activation
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        
        # Last layer - linear only (no activation)
        x = self.layers[-1](x)
        return x
    



#################################################################
#5. TRAINING
#################################################################
    

def nll(psi, counts):
    #Propper format of data
    counts_torch = torch.from_numpy(counts)

    #Normalized probabilities
    probs = torch.abs(psi)**2
    probs = probs / probs.sum()

    #Avoid log(0) by clipping
    probs = torch.clip(probs, 1e-9, 1.0)

    #negative log likelihood (normalized)
    logp = torch.log(probs)
    ll = torch.sum(counts_torch * logp)
    loss_nll = -ll / torch.sum(counts_torch)

    return loss_nll


def train_model(model, n_epochs, input_data, psi0, OPS_LIST, L, CONFIG, 
                        t_grid_fine, learning_rate, counts_shots, print_every=50):
    """
    Improved training function with better convergence.
    """
    
    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5
    )
    
    for epoch_i in range(n_epochs):

        optimizer.zero_grad()
        
        # Forward pass
        output_params = model(input_data)
        predicted_params = create_parameter_dict(output_params, OPS_LIST, L, CONFIG)
        
        # Physics computation
        psi_t = physics_computation(predicted_params, psi0, OPS_LIST, L, t_grid_fine)
        
        # Compute loss
        loss = nll(psi_t, counts_shots)
        
        # Add regularization to prevent parameter explosion
        if CONFIG.get('lambda_reg', 0) > 0:
            param_norm = sum(p.norm() for p in predicted_params.values() if isinstance(p, torch.Tensor))
            loss = loss + CONFIG['lambda_reg'] * param_norm
        
        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        # Record loss
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Learning rate scheduling
        scheduler.step(loss_val)
        
        # Print progress
        if epoch_i % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_i:4d} | Loss: {loss_val:.6f} | LR: {current_lr:.2e}")
    
    return model, predicted_params, psi_t, loss_history


def create_parameter_dict(params, OPS_LIST, L, CONFIG):
    """
    Create parameter dictionary from model output.
    FIXED: Better handling of parameter shapes.
    """
    predicted_params = {}
    
    # Ensure params is 1D
    if params.ndim > 1:
        params = params.squeeze()
    
    if params.ndim > 1:
        raise ValueError(f"params should be 1D after squeeze, got shape {params.shape}")
    
    idx = 0
    
    # 1. Hamiltonian parameters
    n_hamiltonian = len(OPS_LIST)
    predicted_params['theta'] = params[idx:idx + n_hamiltonian]
    idx += n_hamiltonian
    
    #Not used
    for field in ['x', 'y', 'z']:
        key = f'{field}_fields'
        if CONFIG.get(key, False):
            predicted_params[f'rot_{field}'] = params[idx:idx + L]
            idx += L
    
    # Verify we used all parameters
    if idx != params.shape[0]:
        raise ValueError(
            f"Parameter count mismatch. Used {idx} parameters, "
            f"but model output has {params.shape[0]}"
        )
    
    return predicted_params




#################################################################
#6. MAIN PROGRAM
#################################################################


if __name__ == "__main__":
    
    config_file = "../config/MLP_to_H_configuration.yaml"
    
    # Load configuration
    print(f"Loading config: {config_file}")
    CONFIG = load_config(config_file)

    # Main parameters
    MLP_input_size = CONFIG['MLP_input_size']
    MLP_output_size = CONFIG['MLP_output_size']
    CHI = CONFIG['bond_dimension_learning']
    learning_mode = CONFIG['learning_mode']
    MLP_input_folder = CONFIG['MLP_input_folder']
    MLP_input_file = CONFIG['MLP_input_file']
    use_mpo = CONFIG['MPO_ON']
    max_mpo_chi = CONFIG['MAX_MPO_CHI']


    #Load input
    input_data = np.load(f"{MLP_input_file}")

    #Run inference with MLP model
    bitstring_probs, prediction = run_inference(input_data, MLP_input_folder, use_mpo, max_mpo_chi)

    #Prepare initial state depending on the learning mode selected
    if learning_mode == 'output_only':
        initial_state_kind = CONFIG['initial_state_kind']
        L = min_power_of_2(MLP_output_size)
        psi0 = prepare_initial_state(L, initial_state_kind)

        print(f"Learning only output of size {MLP_output_size} with {L} qubits")

    elif learning_mode == 'output_from_input':
        L = min_power_of_2(MLP_input_size)
        psi0 = initial_state_from_input(L, input_data.flatten())

        print(f"Learning output from input of size {MLP_input_size} with {L} qubits")

    dim = 2**L
    #Input of training NN
    mps_input_probs = torch.ones((1, L, 2))

    #Add missing zeros to bistring probs
    if len(bitstring_probs) < dim:
        # Create padded array with zeros
        padded_probs = np.zeros(dim)
        padded_probs[:len(bitstring_probs)] = bitstring_probs
        bitstring_probs = padded_probs


    # Initialize Hamiltonian operators
    OPS_LIST = OperatorClass(L)
    OPS_LIST.add_operators('ZZ')
    OPS_LIST.add_operators('X')
    OPS_LIST.add_operators('Z')
    NUM_COEFFICIENTS = len(OPS_LIST)

    # Set random seed
    torch.manual_seed(CONFIG["seed_init"])
    np.random.seed(CONFIG["seed_init"])
    
    # Calculate output dimension
    NN_OUTPUT_DIM = NUM_COEFFICIENTS + sum(
        CONFIG.get(f'{axis}_fields', False) for axis in ['x', 'y', 'z']
    ) * L
    
    #Print run infor
    print(f"\n{'='*60}")
    print(f"Network Configuration:")
    print(f"  Input dim: {L} qubits × 2 = {L*2}")
    print(f"  Output dim: {NN_OUTPUT_DIM} parameters")
    print(f"  - Hamiltonian: {NUM_COEFFICIENTS}")
    print(f"  - Rotations: {NN_OUTPUT_DIM - NUM_COEFFICIENTS}")
    print(f"{'='*60}\n")
    
    # Training parameters
    n_epochs = CONFIG['N_epochs']
    t_grid_fine = torch.arange(0.0, CONFIG["t_max"] + CONFIG["dt"]/2, CONFIG["dt"])
    learning_rate = CONFIG['learning_rate']

    #Create model to train
    NNmodel = MPS_MLP(
        L=L, 
        chi=CHI, 
        num_params=NN_OUTPUT_DIM, 
        num_dims=[]
    )

    print(f"{'='*60}\n")
    print("TRAINING WITH MPS NETWORK")
    print(f"{'='*60}\n")
    print(f"Starting training for {n_epochs} epochs...")

    #Train model
    NNmodel, final_params, psi_final, loss_history = train_model(
        NNmodel, 
        n_epochs, 
        mps_input_probs, 
        psi0, 
        OPS_LIST, 
        L,
        CONFIG, 
        t_grid_fine, 
        learning_rate, 
        bitstring_probs, 
        CONFIG['print_every']
    )

    # Save model

    file_core = create_filename_core(CONFIG, L)
    save_path = f"../saved_models/MPS-model_{file_core}.pt"
    os.makedirs("../saved_models", exist_ok=True)
    
    torch.save({
        'model_state_dict': NNmodel.state_dict(),
        'config': CONFIG,
        'loss_history': loss_history,
        'final_params': final_params
    }, save_path)
    
    print(f"\n✅ Model saved to: {save_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    # Compute final probabilities
    probs_final = torch.abs(psi_final)**2
    probs_np = probs_final.detach().numpy()
    probs_np = probs_np / probs_np.sum()
    
    # Normalize counts
    normalized_counts = bitstring_probs / bitstring_probs.sum()

    
    # Calculate divergence
    total_divergence = np.sum(np.abs(normalized_counts - probs_np))
    print(f"Total probability divergence: {total_divergence:.6f}")
    
    # Calculate KL divergence
    epsilon = 1e-10
    kl_div = np.sum(
        normalized_counts * np.log((normalized_counts + epsilon) / (probs_np + epsilon))
    )
    print(f"KL divergence: {kl_div:.6f}")
    
    # Print learned parameters
    print("\nLearned parameters:")
    for key, val in final_params.items():
        if isinstance(val, torch.Tensor):
            val_np = val.detach().numpy()
            print(f"  {key}: {val_np}")


    bitstrings = generate_bitstring_list(L)

    #Save results

    filename = f'../results/learned_hamiltonian_{file_core}'
    np.savez(filename, np.array(final_params))

    filename = f'../results/loss_history_{file_core}.npy'
    np.save(filename, np.array(loss_history))

    filename = f'../results/learned_distribution_{file_core}.npy'
    data = np.array(list(zip(bitstrings, probs_np)))
    
    # Plot results
    limit = int(MLP_output_size)
    bar_plot_strings_comparison(bitstrings[0:limit], normalized_counts[0:limit], probs_np[0:limit], file_core)
    plot_training_loss(loss_history, file_core)
    
    print("\nPlots saved!")