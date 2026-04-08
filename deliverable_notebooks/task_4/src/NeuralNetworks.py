import numpy as np
import torch
from torchvision import transforms, datasets
import tensorkrowch as tk
import torch.nn as nn
import torchtt as tntt 


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
    
    def load_dense_weights(self, dense_weights, dense_bias):
        """Load pre-trained dense weights into MPO format."""
        with torch.no_grad():
            # Decompose dense weights to TT format
            W_np = dense_weights.cpu().numpy()
            b_np = dense_bias.cpu().numpy()
            self.mpo.init_from_weights(W_np, b_np)
            self.mpo_ready = True
            
            # Also store in fc_mpo for reference
            self.fc_mpo.weight.data = dense_weights
            self.fc_mpo.bias.data = dense_bias


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