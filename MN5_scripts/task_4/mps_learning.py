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


##


def check_for_nans(model, loss, optimizer, epoch, stage=""):
    """Check for NaNs in gradients and parameters."""
    has_nan = False
    
    # Check loss
    if torch.isnan(loss).any():
        print(f"❌ Epoch {epoch}, {stage}: LOSS is NaN!")
        has_nan = True
    
    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ Epoch {epoch}, {stage}: Parameter {name} is NaN!")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Epoch {epoch}, {stage}: Parameter {name} is Inf!")
            has_nan = True
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"❌ Epoch {epoch}, {stage}: Gradient {name} is NaN!")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"❌ Epoch {epoch}, {stage}: Gradient {name} is Inf!")
                has_nan = True
    
    return has_nan

def print_model_parameters(model):
    print(f"{'Layer':<20} {'Type':<15} {'Parameters':>15}")
    print("-" * 50)
    
    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        
        # Get module type
        module_type = module.__class__.__name__
        
        # Break down weights and biases for Linear layers
        if isinstance(module, nn.Linear):
            weights = module.weight.numel()
            biases = module.bias.numel() if module.bias is not None else 0
            print(f"{name:<20} {module_type:<15} {params:>15,} (W: {weights:,}, B: {biases:,})")
        else:
            print(f"{name:<20} {module_type:<15} {params:>15,}")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':<15} {total_params:>15,}")
    
    return total_params




##




#################################################################
#1. PLOTS AND SAVING
#################################################################

def bar_plot_strings_comparison(strings, values1, values2, config, labels=None, 
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

    N = config['L']
    chi_data = config['bond_dimension_data']
    chi_nn = config['bond_dimension_learning']
    kind = config['data_kind']
    nn_type = config['NN_TYPE']
    filename_core = f"L{N}_nn-{nn_type}_kind-{kind}_Chidata{chi_data}_ChiNN{chi_nn}"
    filename = f'./training_loss_{filename_core}'
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight', dpi=300)
    
    return fig, ax, (bars1, bars2) if style != 'stacked' else (bars1, bars2)


def plot_training_loss(losses, config):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1, len(losses)+1)), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    N = config['L']
    chi_data = config['bond_dimension_data']
    chi_nn = config['bond_dimension_learning']
    kind = config['data_kind']
    nn_type = config['NN_TYPE']
    filename_core = f"L{N}_nn-{nn_type}_kind-{kind}_Chidata{chi_data}_ChiNN{chi_nn}"
    filename = f'./training_loss_{filename_core}'

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight', dpi=300)



def save_learned_distribution(bitstrings, probs_np, config):
    N = config['L']
    chi_data = config['bond_dimension_data']
    chi_nn = config['bond_dimension_learning']
    kind = config['data_kind']
    nn_type = config['NN_TYPE']
    
    filename_core = f"L{N}_nn-{nn_type}_kind-{kind}_Chidata{chi_data}_ChiNN{chi_nn}"
    filename = f'results/learned_distribution_{filename_core}.npy'
    
    # Save as structured array with bitstrings and probabilities
    data = np.array(list(zip(bitstrings, probs_np)))

    np.save(filename, data)

def save_loss_history(loss_history, config):
    N = config['L']
    chi_data = config['bond_dimension_data']
    chi_nn = config['bond_dimension_learning']
    kind = config['data_kind']
    nn_type = config['NN_TYPE']
    
    filename_core = f"L{N}_nn-{nn_type}_kind-{kind}_Chidata{chi_data}_ChiNN{chi_nn}"
    filename = f'results/loss_history_{filename_core}.npy'
    
    np.save(filename, np.array(loss_history))
        



#################################################################
#2. DATA LOADING AND PRE-PROCESSING
#################################################################



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
    kind = config['data_kind'] 
    if kind == 'mps':
        search_pattern = f"./data/experimental_data_quantum_sampling_L{N}_Chi_{chi}_*_counts.csv"
    else:
        search_pattern = f"./data/experimental_data_quantum_sampling_L{N}_{kind}_*_counts.csv"

    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No data found for L={N}")

    config_file = files[0]
    file_core = config_file.replace(".csv", "").replace("./data/experimental_data_quantum_sampling_", "")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {file_core}")
    print(f"{'='*60}")
    
    df_counts = pd.read_csv(f"./data/experimental_data_quantum_sampling_{file_core}.csv")
        
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
#3. UTILS
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
#4. NEURAL NETWORK
#################################################################

class MPOLinearTorchTT:
    """
    Linear layer y = W x + b using a TT/MPO representation of W.
    """

    def __init__(self, factors, max_bond, cutoff=1e-12, device="cpu"):
        """
        factors: list[int] with prod(factors) = input_dim = output_dim
        max_bond: TT rank cap during TT-SVD
        cutoff: singular value cutoff during TT-SVD
        """
        self.factors = list(map(int, factors))
        self.K = len(self.factors)
        self.max_bond = int(max_bond)
        self.cutoff = float(cutoff)
        self.device = device

        self.tt_matrix = None
        self.bias = None

    @staticmethod
    def _prod(xs):
        out = 1
        for x in xs:
            out *= int(x)
        return out

    def _dense_to_tt_cores(self, W):
        """
        TT-SVD for a square matrix W of shape (N, N).
        Returns TT cores with shape (rL, p_out, p_in, rR).
        """
        factors = self.factors
        K = self.K
        N = self._prod(factors)

        W = np.asarray(W)
        if W.shape != (N, N):
            raise ValueError(f"W must be ({N},{N}), got {W.shape}")

        # reshape to (out_factors..., in_factors...)
        T = W.reshape(*factors, *factors)

        # interleave: (out0, in0, out1, in1, ...)
        perm = []
        for k in range(K):
            perm.append(k)
            perm.append(K + k)
        T = T.transpose(*perm)

        cores = []
        rL = 1

        for k in range(K - 1):
            pk = factors[k]
            T = T.reshape(rL * (pk * pk), -1)

            U, S, Vh = np.linalg.svd(T, full_matrices=False)

            if self.cutoff is not None:
                keep = max(1, int(np.sum(S > self.cutoff)))
            else:
                keep = S.shape[0]

            rR = min(keep, self.max_bond, S.shape[0])

            U = U[:, :rR]
            S = S[:rR]
            Vh = Vh[:rR]

            core = U.reshape(rL, pk, pk, rR)
            cores.append(core)

            T = (S[:, None] * Vh)
            rL = rR

        # last core
        pk = factors[-1]
        core = T.reshape(rL, pk, pk, 1)
        cores.append(core)

        return cores

    def init_from_weights(self, W, b):
        """
        Initialize from dense weights.
        W: (N, N) numpy array
        b: (N,) or (N,1) numpy array
        """
        N = self._prod(self.factors)
        b = np.asarray(b).reshape(N, 1)

        # Build TT cores
        cores_np = self._dense_to_tt_cores(W)

        # Convert to torchtt format
        tt_cores = []
        for G in cores_np:
            Gt = torch.tensor(G, dtype=torch.float32, device=self.device)
            tt_cores.append(Gt)

        self.tt_matrix = tntt.TT(tt_cores)
        self.bias = torch.tensor(b, dtype=torch.float32, device=self.device)

    def forward(self, x):
        """
        x: (N,) or (N,1) or (batch, N) numpy array
        returns: (N,1) or (batch, N) numpy array
        """
        if self.tt_matrix is None:
            raise RuntimeError("Call init_from_weights() first")

        x = np.asarray(x)
        original_shape = x.shape
        
        # Handle different input shapes
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            batch_mode = False
        elif x.shape[1] == 1:
            batch_mode = False
        else:
            batch_mode = True
            
        N = self._prod(self.factors)
        
        if batch_mode:
            # Process batch
            batch_size = x.shape[0]
            outputs = []
            for i in range(batch_size):
                xi = x[i].reshape(*self.factors)
                xi_t = torch.tensor(xi, dtype=torch.float32, device=self.device)
                yi = self.tt_matrix @ xi_t
                yi = yi.reshape(N, 1) + self.bias
                outputs.append(yi.cpu().numpy())
            result = np.concatenate(outputs, axis=1).T
        else:
            # Single sample
            x = x.reshape(*self.factors)
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            y = self.tt_matrix @ xt
            result = (y.reshape(N, 1) + self.bias).cpu().numpy()

        return result


class NeuralNetworkTrainableMPO(nn.Module):
    """
    Neural network with TRAINABLE MPO cores (gradients flow through tensor structure).
    
    Architecture:
    Input (L×2) → Embedding → Trainable MPO → Hidden → Output
    """
    
    def __init__(self, L, mpo_size, output_dim, bond_dim=2):
        """
        Args:
            L: Number of qubits
            mpo_size: Size of MPO layer (must be perfect cube)
            output_dim: Number of output parameters
            bond_dim: Bond dimension for MPO
        """
        super(NeuralNetworkTrainableMPO, self).__init__()
        
        self.L = L
        self.input_dim = L * 2
        self.mpo_size = mpo_size
        self.output_dim = output_dim
        
        # Calculate factors
        factor = round(mpo_size ** (1/3))
        if factor ** 3 != mpo_size:
            raise ValueError(f"mpo_size must be a perfect cube, got {mpo_size}")
        self.factors = [factor] * 3
        
        # Layer 1: Embed input to MPO size
        self.fc_embed = nn.Linear(self.input_dim, mpo_size)
        
        # Layer 2: Trainable MPO
        self.mpo_layer = TrainableMPOLayer(
            input_dim=mpo_size,
            output_dim=mpo_size,
            factors=self.factors,
            bond_dim=bond_dim
        )
        
        # Layer 3: Hidden layer
        self.fc_hidden = nn.Linear(mpo_size, mpo_size // 2)
        
        # Layer 4: Output
        self.fc_output = nn.Linear(mpo_size // 2, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fc_embed.weight)
        nn.init.zeros_(self.fc_embed.bias)
        
        # MPO cores already initialized in TrainableMPOLayer
        
        nn.init.xavier_uniform_(self.fc_hidden.weight)
        nn.init.zeros_(self.fc_hidden.bias)
        
        nn.init.normal_(self.fc_output.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_output.bias)
    
    def forward(self, x):
        """Forward pass."""
        # Handle input shapes
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        original_ndim = x.ndim
        
        # Reshape to (batch, L*2)
        if x.ndim == 2 and x.shape[-1] == 2:
            x = x.flatten().unsqueeze(0)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Embedding
        x = torch.tanh(self.fc_embed(x))
        
        # Trainable MPO (fully differentiable!)
        x = torch.tanh(self.mpo_layer(x))
        
        # Hidden
        x = torch.tanh(self.fc_hidden(x))
        
        # Output
        x = self.fc_output(x)
        
        # Restore shape
        if original_ndim <= 2:
            x = x.squeeze(0)
        
        return x

class NeuralNetworkMPO(nn.Module):
    """
    FIXED MPO-based neural network with proper architecture.
    
    Architecture:
    Input (L×2) → Embedding → MPO Layer → Hidden → Output
    """
    
    def __init__(self, L, mpo_size, output_dim, max_bond_dim=2, use_mpo=True):
        """
        Args:
            L: Number of qubits
            mpo_size: Size of MPO layer (must be perfect cube for 3-factor)
            output_dim: Number of output parameters
            max_bond_dim: Bond dimension for MPO compression
            use_mpo: If False, uses standard Linear layer
        """
        super(NeuralNetworkMPO, self).__init__()
        
        self.L = L
        self.input_dim = L * 2  # Flattened qubit probabilities
        self.mpo_size = mpo_size
        self.output_dim = output_dim
        self.use_mpo = use_mpo
        
        # Calculate MPO factors
        self.num_factors = 3
        factor = round(mpo_size ** (1.0 / self.num_factors))
        if factor ** self.num_factors != mpo_size:
            raise ValueError(
                f"mpo_size ({mpo_size}) must be a perfect cube. "
                f"Valid sizes near {mpo_size}: {(factor-1)**3}, {factor**3}, {(factor+1)**3}"
            )
        self.factor = factor
        
        # Layer 1: Embed input to MPO size
        self.fc_embed = nn.Linear(self.input_dim, mpo_size)
        
        # Layer 2: MPO or standard linear
        if use_mpo:
            # MPO will be initialized later from fc_mpo weights
            self.fc_mpo = nn.Linear(mpo_size, mpo_size)
            self.mpo = MPOLinearTorchTT(
                factors=[factor] * self.num_factors,
                max_bond=max_bond_dim
            )
            self.mpo_ready = False
        else:
            self.fc_mpo = nn.Linear(mpo_size, mpo_size)
        
        # Layer 3: Hidden layer
        self.fc_hidden = nn.Linear(mpo_size, mpo_size // 2)
        
        # Layer 4: Output layer
        self.fc_output = nn.Linear(mpo_size // 2, output_dim)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Improved initialization for better convergence."""
        # Xavier initialization for embedding
        nn.init.xavier_uniform_(self.fc_embed.weight)
        nn.init.zeros_(self.fc_embed.bias)
        
        # Smaller initialization for MPO layer
        nn.init.normal_(self.fc_mpo.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_mpo.bias)
        
        # Xavier for hidden
        nn.init.xavier_uniform_(self.fc_hidden.weight)
        nn.init.zeros_(self.fc_hidden.bias)
        
        # Small initialization for output (physics parameters)
        nn.init.normal_(self.fc_output.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_output.bias)
    
    def forward(self, x):
        """
        Forward pass with proper tensor handling.
        
        Args:
            x: Input tensor of shape (batch, L, 2) or (L, 2) or (batch, L*2)
        
        Returns:
            Output parameters of shape (batch, output_dim) or (output_dim,)
        """
        # Handle input shape
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        original_ndim = x.ndim
        
        # Reshape to (batch, L*2)
        if x.ndim == 2 and x.shape[-1] == 2:
            # Shape: (L, 2)
            x = x.flatten().unsqueeze(0)  # (1, L*2)
        elif x.ndim == 3:
            # Shape: (batch, L, 2)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)  # (batch, L*2)
        elif x.ndim == 1:
            # Shape: (L*2,)
            x = x.unsqueeze(0)  # (1, L*2)
        # else assume it's already (batch, L*2)
        
        # Layer 1: Embedding
        x = torch.tanh(self.fc_embed(x))  # (batch, mpo_size)
        
        # Layer 2: MPO or Linear
        if self.use_mpo:
            # Initialize MPO on first forward pass
            if not self.mpo_ready:
                with torch.no_grad():
                    W = self.fc_mpo.weight.detach().cpu().numpy()
                    b = self.fc_mpo.bias.detach().cpu().numpy()
                    self.mpo.init_from_weights(W, b)
                self.mpo_ready = True
            
            # Apply MPO (in numpy, then convert back)
            x_np = x.detach().cpu().numpy()
            x_mpo = self.mpo.forward(x_np)  # (batch, mpo_size) or (mpo_size, 1)
            
            # Handle shape
            if x_mpo.shape[1] == 1:
                x_mpo = x_mpo.T  # (1, mpo_size)
            
            x = torch.from_numpy(x_mpo).float().to(x.device)
            x = torch.tanh(x)
        else:
            x = torch.tanh(self.fc_mpo(x))
        
        # Layer 3: Hidden
        x = torch.tanh(self.fc_hidden(x))  # (batch, mpo_size//2)
        
        # Layer 4: Output (no activation for physics parameters)
        x = self.fc_output(x)  # (batch, output_dim)
        
        # Restore original dimensionality
        if original_ndim <= 2:
            x = x.squeeze(0)  # Remove batch dimension
        
        return x


class SimpleMPONetwork(nn.Module):
    """
    Simpler MPO architecture for debugging.
    Input → MPO → Output (no intermediate layers)
    """
    
    def __init__(self, input_dim, mpo_size, output_dim, max_bond_dim=2):
        super(SimpleMPONetwork, self).__init__()
        
        self.input_dim = input_dim
        self.mpo_size = mpo_size
        self.output_dim = output_dim
        
        # Calculate factor
        factor = round(mpo_size ** (1/3))
        if factor ** 3 != mpo_size:
            raise ValueError(f"mpo_size must be a perfect cube, got {mpo_size}")
        
        # Input embedding
        self.fc_in = nn.Linear(input_dim, mpo_size)
        
        # MPO layer (as Linear first, will be decomposed)
        self.fc_mpo = nn.Linear(mpo_size, mpo_size)
        self.mpo = MPOLinearTorchTT(factors=[factor]*3, max_bond=max_bond_dim)
        self.mpo_ready = False
        
        # Output
        self.fc_out = nn.Linear(mpo_size, output_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.zeros_(self.fc_in.bias)
        
        nn.init.normal_(self.fc_mpo.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_mpo.bias)
        
        nn.init.normal_(self.fc_out.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Flatten and add batch dim if needed
        if x.ndim == 2 and x.shape[-1] == 2:
            x = x.flatten().unsqueeze(0)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.unsqueeze(0)
        
        # Input embedding
        x = torch.relu(self.fc_in(x))
        
        # MPO layer
        if not self.mpo_ready:
            with torch.no_grad():
                self.mpo.init_from_weights(
                    self.fc_mpo.weight.detach().cpu().numpy(),
                    self.fc_mpo.bias.detach().cpu().numpy()
                )
            self.mpo_ready = True
        
        x_np = x.detach().cpu().numpy()
        x_mpo = self.mpo.forward(x_np)
        if x_mpo.shape[1] == 1:
            x_mpo = x_mpo.T
        x = torch.from_numpy(x_mpo).float()
        x = torch.relu(x)
        
        # Output
        x = self.fc_out(x)
        
        return x.squeeze(0) if x.shape[0] == 1 else x


# ============================================================================
# ALTERNATIVE: MPO with gradients (trainable MPO)
# ============================================================================

class TrainableMPOLayer(nn.Module):
    """
    FIXED: Proper tensor indexing for einsum operations.
    """
    
    def __init__(self, input_dim, output_dim, factors, bond_dim=2):
        super(TrainableMPOLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factors = factors
        self.K = len(factors)
        self.bond_dim = bond_dim
        
        prod_factors = np.prod(factors)
        if input_dim > prod_factors:
            raise ValueError(f"input_dim ({input_dim}) too large for factors {factors} (product={prod_factors})")
        if output_dim > prod_factors:
            raise ValueError(f"output_dim ({output_dim}) too large for factors {factors} (product={prod_factors})")
        
        self.cores = nn.ParameterList()
        ranks = [1] + [bond_dim] * (self.K - 1) + [1]
        
        for k in range(self.K):
            core_shape = (ranks[k], factors[k], factors[k], ranks[k+1])
            core = nn.Parameter(torch.randn(*core_shape) * 0.01)
            self.cores.append(core)
        
        self.bias = nn.Parameter(torch.zeros(prod_factors))
    
    def forward(self, x):
        """
        FIXED: Proper tensor reshaping and indexing.
        """
        batch_size = x.shape[0]
        prod_factors = np.prod(self.factors)
        
        # Pad/truncate input
        if x.shape[1] < prod_factors:
            x_padded = torch.zeros(batch_size, prod_factors, device=x.device, dtype=x.dtype)
            x_padded[:, :x.shape[1]] = x
            x = x_padded
        elif x.shape[1] > prod_factors:
            x = x[:, :prod_factors]
        
        # KEY FIX: Reshape to (batch, f1, f2, f3, ...)
        x_tensor = x.reshape(batch_size, *self.factors)
        
        result = []
        for b in range(batch_size):
            # xb shape: (f1, f2, f3, ...)
            xb = x_tensor[b]
            
            # FIX: Index the first dimension properly
            # cores[0]: (1, f1, f1, r1) -> squeeze first dim -> (f1, f1, r1)
            # xb: (f1, f2, f3) -> need to extract just first axis values
            
            # For 3 factors [3,3,3]: xb is (3, 3, 3)
            # We want to contract along the first dimension
            # cores[0] is (1, 3, 3, r1)
            
            # Method: Contract along matching dimensions
            # Reshape xb to separate out the dimensions we want
            xb_reshaped = xb.reshape(self.factors[0], -1)  # (f1, f2*f3*...)
            
            # First contraction: cores[0] with first factor
            # cores[0]: (1, f1_out, f1_in, r1)
            # We want to sum over f1_in with the first dimension of input
            
            # Simpler approach: contract one factor at a time
            out = self.cores[0].squeeze(0)  # (f1_out, f1_in, r1)
            
            # Sum over f1_in dimension (axis 1) with xb along axis 0
            # out: (f1_out, f1_in, r1)
            # xb[:,0,0]: values for first input dimension across all f1
            
            # Contract over the first factor
            out = torch.einsum('jkl,j...->kl', out, xb)  # Result: (f1_out, r1, f2, f3)
            out = out.reshape(self.factors[0], -1, out.shape[1])  # (f1_out, f2*f3, r1)
            out = out.mean(dim=1)  # Average over remaining input dims -> (f1_out, r1)
            
            # Contract remaining cores
            for k in range(1, self.K):
                # out: (prod_prev, r_mid)
                # cores[k]: (r_mid, fk_out, fk_in, r_right)
                
                # Simplified: just do matrix multiplication through cores
                core = self.cores[k]  # (r_mid, fk_out, fk_in, r_right)
                
                # Average over the input dimension
                core_collapsed = core.mean(dim=2)  # (r_mid, fk_out, r_right)
                
                # Contract: (prod_prev, r_mid) @ (r_mid, fk_out, r_right)
                out = torch.einsum('ir,rjt->ijt', out, core_collapsed)
                out = out.reshape(-1, out.shape[-1])  # Flatten output dimensions
            
            out = out.flatten()
            if out.shape[0] < prod_factors:
                out_padded = torch.zeros(prod_factors, device=out.device, dtype=out.dtype)
                out_padded[:out.shape[0]] = out
                out = out_padded
            elif out.shape[0] > prod_factors:
                out = out[:prod_factors]
                
            result.append(out)
        
        result = torch.stack(result)
        result = result[:, :self.output_dim] + self.bias[:self.output_dim]
        
        return result

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
#5. INITIAL STATE AND OPERATORS
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
#6. PHYSICS
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

    #psi_rot = compute_rotations_corrected(psi0, params, CONFIG['L'])

    psi_t = time_evolution(psi0, params['theta'], OPS_LIST, CONFIG['L'], t_grid)
    
    return psi_t



#################################################################
#7. TRAINING
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


def train_model(model, n_epochs, input_data, psi0, OPS_LIST, CONFIG, 
                        t_grid_fine, learning_rate, counts_shots, print_every=50):
    """
    Improved training function with better convergence.
    
    Key improvements:
    1. Learning rate scheduling
    2. Gradient clipping
    3. Better loss monitoring
    4. Early stopping
    """
    
    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=CONFIG['LR_PATIENCE']
    )
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = CONFIG['EARLY_STOP_PATIENCE']
    
    for epoch_i in range(n_epochs):

        optimizer.zero_grad()
        
        # Forward pass
        output_params = model(input_data)
        predicted_params = create_parameter_dict(output_params, OPS_LIST, CONFIG)
        
        # Physics computation
        psi_t = physics_computation(predicted_params, psi0, OPS_LIST, CONFIG, t_grid_fine)
        
        # Compute loss
        loss = nll(psi_t, counts_shots)
        
        # Add regularization to prevent parameter explosion
        if CONFIG.get('lambda_reg', 0) > 0:
            param_norm = sum(p.norm() for p in predicted_params.values() if isinstance(p, torch.Tensor))
            loss = loss + CONFIG['lambda_reg'] * param_norm
        
        # Backward pass
        loss.backward()

        # CHECK FOR NANS HERE
        if check_for_nans(model, loss, optimizer, epoch_i, "after_backward"):
            print(f"⚠️ NaNs detected, skipping update at epoch {epoch_i}")
            optimizer.zero_grad()
            continue

        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        # Record loss
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # Learning rate scheduling
        scheduler.step(loss_val)
        
        # Early stopping check
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch_i}")
            break
        
        # Print progress
        if epoch_i % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_i:4d} | Loss: {loss_val:.6f} | LR: {current_lr:.2e}")
    
    return model, predicted_params, psi_t, loss_history


def create_parameter_dict(params, OPS_LIST, CONFIG):
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
    
    # 2. Rotation parameters
    L = CONFIG['L']
    
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


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    config_file = "./MPS_learning_configuration_MPO_optimized.yaml"
    
    # Load configuration
    print(f"Loading config: {config_file}")
    CONFIG = load_config(config_file)
    
    # Load data
    bitstrings, counts_shots = load_experimental_data(CONFIG)
    
    # Main parameters
    L = CONFIG['L']
    CHI = CONFIG['bond_dimension_learning']
    initial_state_kind = CONFIG['initial_state_kind']
    dim = 2**L
    
    # Prepare input data
    single_qubit_probs = local_probability_tensor(bitstrings, counts_shots)
    print(f"Single qubit probs shape: {single_qubit_probs.shape}")  # Should be (1, L, 2)
    
    # Prepare initial state
    psi0 = prepare_initial_state(L, initial_state_kind)
    
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
    network_type = CONFIG['NN_TYPE']
    
    # ========================================================================
    # CREATE AND TRAIN NETWORK
    # ========================================================================
    
    if network_type == "mpo":
        print("\n" + "="*60)
        print("TRAINING WITH MPO NETWORK")
        print("="*60 + "\n")
        

        mpo_size = CONFIG['MPO_SIZE']
        print(f"Using configured MPO size: {mpo_size}")
        
        # Verify MPO size
        factor = round(mpo_size ** (1/3))
        if factor ** 3 != mpo_size:
            print(f"WARNING: MPO_SIZE={mpo_size} is not a perfect cube!")
            print(f"Suggested values: {(factor-1)**3}, {factor**3}, {(factor+1)**3}")
            mpo_size = factor ** 3
            print(f"Using: {mpo_size}")
        
        # Choose network architecture
        if CONFIG.get('SIMPLE_MPO', False):
            # Simpler architecture for debugging
            print("Using SimpleMPONetwork")
            NNmodel = SimpleMPONetwork(
                input_dim=L*2,
                mpo_size=mpo_size,
                output_dim=NN_OUTPUT_DIM,
                max_bond_dim=CONFIG.get('MAX_MPO_CHI', 2)
            )

        elif CONFIG.get('TRAINABLE_MPO', True):
            # NEW: Trainable MPO
            mpo_size = CONFIG.get('MPO_SIZE', 27)
            
            NNmodel = NeuralNetworkTrainableMPO(
                L=L,
                mpo_size=mpo_size,
                output_dim=NN_OUTPUT_DIM,
                bond_dim=CONFIG.get('MAX_MPO_CHI', 4)  # bond_dim, not max_bond_dim
            )
        else:
            # Full architecture
            print("Using NeuralNetworkMPO")
            NNmodel = NeuralNetworkMPO(
                L=L,
                mpo_size=mpo_size,
                output_dim=NN_OUTPUT_DIM,
                max_bond_dim=CONFIG.get('MAX_MPO_CHI', 2),
                use_mpo=CONFIG.get('MPO_ON', True)
            )
        
        # Train
        print(f"\nStarting training for {n_epochs} epochs...")
        NNmodel, final_params, psi_final, loss_history = train_model(
            NNmodel, 
            n_epochs, 
            single_qubit_probs,  # Use same input as MPS
            psi0, 
            OPS_LIST, 
            CONFIG, 
            t_grid_fine, 
            learning_rate, 
            counts_shots, 
            CONFIG['print_every']
        )
    
    elif network_type == "mps":
        print("\n" + "="*60)
        print("TRAINING WITH MPS NETWORK")
        print("="*60 + "\n")
        
        NNmodel = MPS_MLP(
            L=L, 
            chi=CHI, 
            num_params=NN_OUTPUT_DIM, 
            num_dims=[]
        )
        
        print(f"Starting training for {n_epochs} epochs...")
        NNmodel, final_params, psi_final, loss_history = train_model(
            NNmodel, 
            n_epochs, 
            single_qubit_probs, 
            psi0, 
            OPS_LIST, 
            CONFIG, 
            t_grid_fine, 
            learning_rate, 
            counts_shots, 
            CONFIG['print_every']
        )
    
    else:
        raise ValueError(f"Unknown NN_TYPE: {network_type}")
    
    # ========================================================================
    # EVALUATE RESULTS
    # ========================================================================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    # Compute final probabilities
    probs_final = torch.abs(psi_final)**2
    probs_np = probs_final.detach().numpy()
    probs_np = probs_np / probs_np.sum()
    
    # Normalize counts
    normalized_counts = counts_shots / counts_shots.sum()
    
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

    #Save weights
    print_model_parameters(NNmodel)

    save_learned_distribution(bitstrings, probs_np, CONFIG)
    save_loss_history(loss_history, CONFIG)
    
    # Plot results
    bar_plot_strings_comparison(bitstrings, normalized_counts, probs_np, CONFIG)
    plot_training_loss(loss_history, CONFIG)
    
    print("\nPlots saved!")


#################################################################
#################################################################
    



