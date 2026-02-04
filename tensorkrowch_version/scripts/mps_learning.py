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



#################################################################
#PLOTS
#################################################################

def bar_plot_strings_comparison(strings, values1, values2, labels=None, 
                                title="Learned bitstring probabilities", xlabel="Bitstrings", 
                                ylabel="Probability", colors=None, edgecolor='black', 
                                figsize=(12, 7), style='grouped', alpha=0.8):
    """
    Bar plot comparing two sets of data with string labels.
    
    Parameters:
    -----------
    strings : list of str
        String labels for x-axis
    values1, values2 : arrays
        Two sets of values to compare
    labels : tuple of str, optional
        Labels for the two data sets (default: ('Set 1', 'Set 2'))
    colors : tuple of str, optional
        Colors for the two data sets (default: ('skyblue', 'salmon'))
    style : str
        'grouped' for side-by-side bars, 'stacked' for stacked bars,
        'overlap' for overlapping transparent bars
    """
    
    if labels is None:
        labels = ('True', 'Learned')
    if colors is None:
        colors = ('skyblue', 'salmon')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    n = len(strings)
    x_pos = np.arange(n)
    width = 0.35  # Width of bars
    
    if style == 'grouped':
        # Side-by-side bars
        bars1 = ax.bar(x_pos - width/2, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=alpha)
        bars2 = ax.bar(x_pos + width/2, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=alpha)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.01:  # Only label if significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strings, rotation=45, ha='right', fontsize=10)
        
    elif style == 'stacked':
        # Stacked bars
        bars1 = ax.bar(x_pos, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=alpha)
        bars2 = ax.bar(x_pos, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=alpha,
                      bottom=values1)
        
        # Add value labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            total = v1 + v2
            if total > 0.01:
                ax.text(i, v1/2, f'{v1:.2f}', ha='center', va='center', fontsize=9, color='white')
                ax.text(i, v1 + v2/2, f'{v2:.2f}', ha='center', va='center', fontsize=9, color='white')
                ax.text(i, total, f'{total:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strings, rotation=45, ha='right', fontsize=10)
        
    elif style == 'overlap':
        # Overlapping transparent bars
        bars1 = ax.bar(x_pos, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=0.6)
        bars2 = ax.bar(x_pos, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=0.6)
        
        # Add value labels
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            if v1 > 0.01:
                ax.text(i - 0.1, v1, f'{v1:.2f}', ha='center', va='bottom', 
                       fontsize=9, color=colors[0])
            if v2 > 0.01:
                ax.text(i + 0.1, v2, f'{v2:.2f}', ha='center', va='bottom', 
                       fontsize=9, color=colors[1])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strings, rotation=45, ha='right', fontsize=10)
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add legend
    ax.legend(fontsize=11, framealpha=0.9)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return fig, ax, (bars1, bars2) if style != 'stacked' else (bars1, bars2)


def plot_training_loss(n_epoch, losses):
       plt.figure(figsize=(5,4))
       plt.plot(list(range(1, n_epoch+1)), losses)
       plt.title("Training Loss")
       plt.xlabel("Epoch")
       plt.ylabel("Loss")
       plt.grid(True)
       plt.show()


def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_experimental_data(config):
    """Load experimental/simulated data"""
    N = config["L"]
    chi = config['bond_dimension_data']
    T_max = config["t_max"]
    search_pattern = f"../data/experimental_data_quantum_sampling_L{N}_Chi_{chi}_*_counts.csv"
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No data found for L={N}")

    config_file = files[0]
    file_core = config_file.replace(".csv", "").replace("../data/experimental_data_quantum_sampling_", "")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {file_core}")
    print(f"{'='*60}")
    
    df_counts = pd.read_csv(f"../data/experimental_data_quantum_sampling_{file_core}.csv")
        
    # Remove leading single quote if present
    if df_counts['bitstring'].astype(str).str.startswith("'").all():
        df_counts['bitstring'] = df_counts['bitstring'].str[1:]
    
    # Now extract values
    bitstrings = df_counts['bitstring'].values.astype(str)
    counts_shots = df_counts['count'].values.astype(np.int32)
    
    return bitstrings, counts_shots


def local_probability_tensor(strings, counts):
    '''Calculates probabilities of each qubit, returning vector of size L
    containing the probs of each qubit being 0 or 1'''
    L = len(strings[0])
    total_counts = sum(counts)
    
    prob_matrix = torch.zeros((1, L, 2)) #First index is batch. Needed for feeding into NN
    
    for bitstring, count in zip(strings, counts):
        for qubit in range(L):
            bit_value = int(bitstring[qubit])
            prob_matrix[0, qubit, bit_value] += count
    
    # Normalize by total counts
    prob_matrix /= total_counts
    
    return prob_matrix


#################################################################
#UTILS
#################################################################



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


def x_rotation(theta, dtype=torch.complex64):
    sx = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype, requires_grad=False)
    return torch.matrix_exp(-1j * theta / 2 * sx)

def y_rotation(theta, dtype=torch.complex64):
    sy = torch.tensor([[0., -1j], [1j, 0.]], dtype=dtype, requires_grad=False)
    return torch.matrix_exp(-1j * theta / 2 * sy)

def z_rotation(theta, dtype=torch.complex64):
    sz = torch.tensor([[1., 0.], [0., -1.]], dtype=dtype, requires_grad=False)
    return torch.matrix_exp(-1j * theta / 2 * sz)


#################################################################
#NN
#################################################################

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
#INITIAL STATE AND OPERATORS
#################################################################


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




#################################################################
#PHYSICS
#################################################################


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
    

def compute_rotations_corrected(psi, params, L):
    """Apply rotations more efficiently and correctly."""
    sx, sy, sz, id2 = paulis(dtype=psi.dtype)
    
    # Build per-qubit rotation matrices
    qubit_rots = [torch.eye(2, dtype=psi.dtype, device=psi.device) for _ in range(L)]
    
    # Apply rotations in XYZ order (common convention)
    for rot_type in ['x', 'y', 'z']:
        key = f'rot_{rot_type}'
        if key in params:
            angles = params[key]
            for i in range(min(L, len(angles))):
                if rot_type == 'x':
                    R = x_rotation(angles[i], dtype=psi.dtype)
                elif rot_type == 'y':
                    R = y_rotation(angles[i], dtype=psi.dtype)
                elif rot_type == 'z':
                    R = z_rotation(angles[i], dtype=psi.dtype)
                qubit_rots[i] = R @ qubit_rots[i]
    
    # Apply all rotations at once via tensor product
    full_rotation = kron_n(qubit_rots)
    return full_rotation @ psi
    

def compute_rotations(psi, params, L):
    sx, sy, sz, id2 = paulis()

    rot_funcs = {
    'rot_x': lambda theta: x_rotation(theta, dtype=psi.dtype),
    'rot_y': lambda theta: y_rotation(theta, dtype=psi.dtype),
    'rot_z': lambda theta: z_rotation(theta, dtype=psi.dtype)
    }
    
    for key, rot_func in rot_funcs.items():
        if key in params:
            for i in range(L):
                rot = kron_n([id2]*i + [rot_func(params[key][i])] + [id2]*(L-i-1))
                #print(f'rot {key} in qubit {i} of angle {params[key][i]}')
                psi = rot@psi
                         
    return psi  


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

    psi_rot = compute_rotations_corrected(psi0, params, CONFIG['L'])

    psi_t = time_evolution(psi_rot, params['theta'], OPS_LIST, CONFIG['L'], t_grid)
    
    return psi_t



#################################################################
#TRAINING
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


def create_parameter_dict(params, OPS_LIST, CONFIG):
    """
    Create parameter dictionary from model output based on configuration.
    
    Args:
        params: Tensor of shape (total_params,)
        OPS_LIST: List of Hamiltonian operators
        CONFIG: Dictionary with keys:
            - 'L': number of qubits
            - 'x_fields': bool (whether to include X rotations)
            - 'y_fields': bool (whether to include Y rotations)
            - 'z_fields': bool (whether to include Z rotations)
    
    Returns:
        Dictionary with keys: 'theta', 'rot_x', 'rot_y', 'rot_z' (only if active)
    """
    predicted_params = {}

    if params.ndim > 1:
        params = params.squeeze() 
    
    # Start index for slicing params
    idx = 0
    
    # 1. Hamiltonian parameters (theta)
    n_hamiltonian = len(OPS_LIST)
    predicted_params['theta'] = params[idx:idx + n_hamiltonian]
    idx += n_hamiltonian
    
    # 2. Rotation parameters based on configuration
    L = CONFIG['L']
    
    if CONFIG.get('x_fields', False):
        predicted_params['rot_x'] = params[idx:idx + L]
        idx += L
    
    if CONFIG.get('y_fields', False):
        predicted_params['rot_y'] = params[idx:idx + L]
        idx += L
    
    if CONFIG.get('z_fields', False):
        predicted_params['rot_z'] = params[idx:idx + L]
        idx += L
    
    # Verify we used all parameters
    if idx != params.shape[0]:
        raise ValueError(f"Parameter count mismatch. Expected {idx} parameters, "
                        f"but model output has {params.shape[0]}")
    
    return predicted_params



def train_model(model, n_epochs, single_qubit_probs, psi0, OPS_LIST, CONFIG, t_grid_fine, learning_rate, counts_shots):

    #initialization
    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_i in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass: NN predicts Hamiltonian parameters
        predicted_params = {}
        output_params = model(single_qubit_probs)

        predicted_params = create_parameter_dict(output_params, OPS_LIST, CONFIG)

        #Dynamics of obtained parameters
        psi_t = physics_computation(predicted_params, psi0, OPS_LIST, CONFIG, t_grid_fine)

        #compute loss
        loss = nll(psi_t, counts_shots)

        #backpropagate and update optimizer
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch_i % 50 == 0:
            print(f"Epoch {epoch_i}, Loss: {loss.item()}")

         
    return model, predicted_params, psi_t, loss_history


if __name__ == "__main__":

    config_file = "/Users/omichel/Desktop/qilimanjaro/projects/retech/retech_2025/tensorkrowch_version/config/MPS_learning_configuration.yaml"

    #load configuration
    print(config_file)
    CONFIG = load_config(config_file)

    # Load data
    bitstrings, counts_shots = load_experimental_data(CONFIG)

    #Main parameters
    L = CONFIG['L']
    CHI = CONFIG['bond_dimension_learning']
    inital_state_kind = CONFIG['initial_state_kind']
    dim = 2**L

    #Reshape data into local probabilities
    single_qubit_probs = local_probability_tensor(bitstrings, counts_shots)

    psi0 = prepare_initial_state(L, inital_state_kind)

    #Initialize and onfigure Hamiltonian Ansatz
    OPS_LIST = OperatorClass(L)

    OPS_LIST.add_operators('ZZ')
    OPS_LIST.add_operators('X')
    OPS_LIST.add_operators('Z')

    NUM_COEFFICIENTS = len(OPS_LIST)

    #Initialize parameters
    torch.manual_seed(CONFIG["seed_init"])

    theta_init = torch.rand(NUM_COEFFICIENTS, dtype=torch.float32, requires_grad=True)
    # Initialize NN
    NN_INPUT_DIM = L

    params = {"theta": theta_init}

    # Add rotation parameters for each enabled field type
    if CONFIG['x_fields']:
        params["rot_x"] = torch.rand(L, dtype=torch.float32, requires_grad=True)
    if CONFIG['y_fields']:
        params["rot_y"] = torch.rand(L, dtype=torch.float32, requires_grad=True)
    if CONFIG['z_fields']:
        params["rot_z"] = torch.rand(L, dtype=torch.float32, requires_grad=True)

    # Update NN output dimension
    NN_OUTPUT_DIM = NUM_COEFFICIENTS + sum(CONFIG[f'{axis}_fields'] for axis in ['x', 'y', 'z']) * L

    model = MPS_MLP(NN_INPUT_DIM, CHI, NN_OUTPUT_DIM, num_dims = []) #num_dims is for optional intermediate layers
    n_epochs = CONFIG['N_epochs']

    t_grid_fine = torch.arange(0.0, CONFIG["t_max"] + CONFIG["dt"]/2, CONFIG["dt"])
    learning_rate = CONFIG['learning_rate']

    model, final_params, psi_final, loss_history = train_model(model, n_epochs, single_qubit_probs, psi0, OPS_LIST, CONFIG, t_grid_fine, learning_rate, counts_shots)



    probs_final = torch.abs(psi_final)**2
    probs_np = probs_final.detach().numpy()
    probs_np = probs_np / probs_np.sum()

    normalized_counts = counts_shots / counts_shots.sum()

    print(probs_np)
    print(normalized_counts)

    diff = 0
    for i,j in zip(normalized_counts, probs_np):
        diff+= abs(i-j)
    print("Total probability divergenge:", diff)

    bar_plot_strings_comparison(bitstrings, normalized_counts, probs_np)

    plot_training_loss(n_epochs, loss_history)

    print(final_params)
    



