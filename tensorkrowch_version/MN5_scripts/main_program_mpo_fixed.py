"""
UPDATED MAIN PROGRAM SECTION FOR MPO

Replace your __main__ section with this improved version.
"""

import numpy as np
import torch
import torch.nn as nn

# Import the fixed MPO classes
from mpo_fixed_complete import (
    NeuralNetworkMPO, 
    SimpleMPONetwork,
    calculate_optimal_mpo_size
)

# Your existing MPS_MLP import
# from your_module import MPS_MLP


def train_model_improved(model, n_epochs, input_data, psi0, OPS_LIST, CONFIG, 
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
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
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
    
    config_file = "./MPS_learning_configuration.yaml"
    
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
        
        # Option 1: Use automatic MPO size calculation
        if CONFIG.get('AUTO_MPO_SIZE', True):
            mpo_size, factor = calculate_optimal_mpo_size(
                input_dim=L*2,
                output_dim=NN_OUTPUT_DIM,
                num_factors=3
            )
            print(f"Auto-calculated MPO size: {mpo_size} (factor={factor})")
        else:
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
        NNmodel, final_params, psi_final, loss_history = train_model_improved(
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
        NNmodel, final_params, psi_final, loss_history = train_model_improved(
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
    
    # Plot results
    bar_plot_strings_comparison(bitstrings, normalized_counts, probs_np, CONFIG)
    plot_training_loss(n_epochs, loss_history, CONFIG)
    
    print("\nPlots saved!")


# ============================================================================
# ADDITIONAL DEBUGGING UTILITIES
# ============================================================================

def test_mpo_gradient_flow(model, input_data):
    """
    Test if gradients flow through the MPO network.
    """
    print("\n=== Testing MPO Gradient Flow ===")
    
    # Forward pass
    output = model(input_data)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = []
    no_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_grad.append((name, grad_norm))
        else:
            no_grad.append(name)
    
    print("\nParameters with gradients:")
    for name, norm in has_grad:
        print(f"  {name:30s}: {norm:.6e}")
    
    if no_grad:
        print("\nWARNING: Parameters without gradients:")
        for name in no_grad:
            print(f"  {name}")
    else:
        print("\n✓ All parameters have gradients!")
    
    return len(has_grad) > 0


def compare_mpo_vs_linear(L, mpo_size, output_dim, input_data):
    """
    Compare MPO network vs standard linear network.
    """
    print("\n=== Comparing MPO vs Linear ===")
    
    # Create both models
    model_mpo = NeuralNetworkMPO(L, mpo_size, output_dim, use_mpo=True)
    model_linear = NeuralNetworkMPO(L, mpo_size, output_dim, use_mpo=False)
    
    # Count parameters
    params_mpo = sum(p.numel() for p in model_mpo.parameters())
    params_linear = sum(p.numel() for p in model_linear.parameters())
    
    print(f"MPO network parameters: {params_mpo:,}")
    print(f"Linear network parameters: {params_linear:,}")
    print(f"Compression ratio: {params_linear / params_mpo:.2f}x")
    
    # Test forward pass
    out_mpo = model_mpo(input_data)
    out_linear = model_linear(input_data)
    
    print(f"\nOutput shapes:")
    print(f"  MPO: {out_mpo.shape}")
    print(f"  Linear: {out_linear.shape}")
    
    return model_mpo, model_linear


if __name__ == "__main__":
    # Run main program
    pass
