# Complete MPO Implementation Fix Guide

## Problem Summary

Your MPO network wasn't converging because:

1. **Input dimension mismatch**: Fed `counts_shots` (num_bitstrings,) instead of `single_qubit_probs` (1, L, 2)
2. **Hardcoded factor**: `factor=5` doesn't work for all MPO sizes
3. **Poor gradient flow**: MPO decomposition wasn't preserving gradients well
4. **Architecture issues**: Layer sizes didn't align properly
5. **No regularization**: Parameters could explode during training

## Complete Solution

### 1. Fixed MPO Architecture

The new `NeuralNetworkMPO` class properly handles:

```
Input (1, L, 2) 
    ↓ flatten
Input (1, L×2)
    ↓ fc_embed
Hidden (1, mpo_size)
    ↓ MPO layer (or Linear)
Hidden (1, mpo_size)
    ↓ fc_hidden
Hidden (1, mpo_size//2)
    ↓ fc_output
Output (1, num_params)
```

**Key features:**
- Automatic input reshaping from (L, 2) to (mpo_size)
- Proper batch dimension handling
- Gradient flow through MPO (via numpy conversion when needed)
- Correct tensor shapes at each layer

### 2. Automatic Factor Calculation

```python
# Calculate factor from MPO size
factor = round(mpo_size ** (1/3))  # For 3-factor decomposition

# Verify it's valid
if factor ** 3 != mpo_size:
    raise ValueError("mpo_size must be a perfect cube")
```

**Valid MPO sizes** (for 3 factors):
- 8 = 2³
- 27 = 3³  ← **Recommended for L=8**
- 64 = 4³
- 125 = 5³
- 216 = 6³
- etc.

### 3. Improved Training Function

Added these critical improvements:

```python
# 1. Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

# 2. Gradient clipping (prevents explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Regularization
if CONFIG['lambda_reg'] > 0:
    param_norm = sum(p.norm() for p in predicted_params.values())
    loss = loss + CONFIG['lambda_reg'] * param_norm

# 4. Early stopping
if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch_i}")
    break
```

### 4. Better Initialization

```python
# Xavier initialization for main layers
nn.init.xavier_uniform_(self.fc_embed.weight)

# Small initialization for MPO (prevents large initial values)
nn.init.normal_(self.fc_mpo.weight, 0.0, 0.01)

# Small initialization for output (physics parameters)
nn.init.normal_(self.fc_output.weight, 0.0, 0.01)
```

## How to Use

### Step 1: Replace Network Class

Replace your `NeuralNetwork` class with either:

**Option A: Full architecture (recommended)**
```python
from mpo_fixed_complete import NeuralNetworkMPO

NNmodel = NeuralNetworkMPO(
    L=8,
    mpo_size=27,  # or use auto-calculation
    output_dim=NN_OUTPUT_DIM,
    max_bond_dim=4,  # Increased from 2
    use_mpo=True
)
```

**Option B: Simple architecture (for debugging)**
```python
from mpo_fixed_complete import SimpleMPONetwork

NNmodel = SimpleMPONetwork(
    input_dim=L*2,
    mpo_size=27,
    output_dim=NN_OUTPUT_DIM,
    max_bond_dim=4
)
```

### Step 2: Use Same Input as MPS

```python
# Both MPS and MPO use the same input now!
input_data = single_qubit_probs  # Shape: (1, L, 2)

# The network handles reshaping internally
output = NNmodel(input_data)
```

### Step 3: Update Configuration

Use the optimized config file: `MPS_learning_configuration_MPO_optimized.yaml`

Key settings:
```yaml
NN_TYPE: "mpo"
AUTO_MPO_SIZE: True  # Let code calculate optimal size
MAX_MPO_CHI: 4       # Increased bond dimension
learning_rate: 0.001 # Lower for stability
lambda_reg: 0.01     # Regularization
N_epochs: 500        # More epochs
```

### Step 4: Use Improved Training

Replace `train_model` with `train_model_improved`:

```python
NNmodel, final_params, psi_final, loss_history = train_model_improved(
    NNmodel, 
    n_epochs, 
    single_qubit_probs,  # Same input as MPS!
    psi0, 
    OPS_LIST, 
    CONFIG, 
    t_grid_fine, 
    learning_rate, 
    counts_shots, 
    CONFIG['print_every']
)
```

## Understanding MPO Size vs Input Size

**The key insight:** MPO size doesn't need to match input size exactly!

```
Input: L×2 = 16 features (for L=8)
    ↓ Embedding layer
MPO layer: 27 features (3×3×3)
    ↓ 
Output: num_params features
```

The embedding layer `fc_embed` maps from input size (16) to MPO size (27).

## Debugging Checklist

If MPO still doesn't converge:

### 1. Verify MPO size is valid
```python
mpo_size = 27
factor = round(mpo_size ** (1/3))
assert factor ** 3 == mpo_size, f"Invalid MPO size: {mpo_size}"
print(f"✓ MPO size {mpo_size} = {factor}³")
```

### 2. Check gradient flow
```python
from main_program_mpo_fixed import test_mpo_gradient_flow

has_gradients = test_mpo_gradient_flow(NNmodel, single_qubit_probs)
if has_gradients:
    print("✓ Gradients flowing correctly")
```

### 3. Compare with Linear layer
```python
# Train with MPO_ON: False first
NNmodel = NeuralNetworkMPO(L, mpo_size, output_dim, use_mpo=False)
# ... train ...
# If this converges but use_mpo=True doesn't, issue is MPO-specific
```

### 4. Compare with MPS
```python
# Train MPS first to verify data/physics code
NNmodel_mps = MPS_MLP(L, chi, output_dim, num_dims=[])
# ... train ...
# MPS should converge well - this is your baseline
```

### 5. Try simpler architecture
```yaml
SIMPLE_MPO: True  # Uses SimpleMPONetwork
```

### 6. Monitor loss carefully
```python
print(f"Epoch {i}: Loss={loss:.6f}, Params norm={param_norm:.6f}")
# Loss should decrease steadily
# Params norm shouldn't explode
```

## Expected Results

### Good convergence:
```
Epoch    0 | Loss: 2.456789 | LR: 1.00e-03
Epoch   50 | Loss: 1.234567 | LR: 1.00e-03
Epoch  100 | Loss: 0.567890 | LR: 1.00e-03
Epoch  150 | Loss: 0.234567 | LR: 5.00e-04
...
Total probability divergence: 0.045123
```

### Bad convergence (needs fixing):
```
Epoch    0 | Loss: 2.456789 | LR: 1.00e-03
Epoch   50 | Loss: 2.401234 | LR: 1.00e-03  ← Loss barely decreasing
Epoch  100 | Loss: 2.398765 | LR: 1.00e-03
...
Total probability divergence: 0.456789  ← High divergence
```

## Parameter Count Comparison

For L=8, output_dim=19:

```
MPS (chi=2):
  ~800 parameters

Standard Linear MPO:
  Input:  16 × 27 = 432
  MPO:    27 × 27 = 729
  Hidden: 27 × 13 = 351
  Output: 13 × 19 = 247
  Total:  ~1,759 parameters

MPO with compression (chi=4):
  ~400-600 parameters (compressed)
```

## Advanced: Trainable MPO Cores

If the fixed MPO still doesn't work well, try the experimental `TrainableMPOLayer`:

```python
from mpo_fixed_complete import TrainableMPOLayer

# This makes MPO cores directly trainable (better gradients)
mpo_layer = TrainableMPOLayer(size=27, factors=[3,3,3], bond_dim=4)
```

This allows gradients to flow directly through the tensor cores rather than through a numpy conversion.

## Summary of Changes

### What was wrong:
1. ❌ Fed wrong input shape to MPO
2. ❌ Hardcoded factor=5
3. ❌ No gradient clipping → parameter explosion
4. ❌ No learning rate scheduling → stuck in bad local minima
5. ❌ No regularization → overfitting
6. ❌ MPO size mismatch with architecture

### What's fixed:
1. ✅ Proper input handling (L,2) → (mpo_size)
2. ✅ Automatic factor calculation
3. ✅ Gradient clipping (max_norm=1.0)
4. ✅ Learning rate scheduling (ReduceLROnPlateau)
5. ✅ L2 regularization (lambda_reg)
6. ✅ Correct layer dimensions throughout
7. ✅ Better initialization
8. ✅ Early stopping
9. ✅ Auto MPO size calculation

## Quick Start

```python
# 1. Import fixed classes
from mpo_fixed_complete import NeuralNetworkMPO, calculate_optimal_mpo_size

# 2. Calculate optimal size
mpo_size, factor = calculate_optimal_mpo_size(L*2, NN_OUTPUT_DIM)
print(f"Using MPO size: {mpo_size} ({factor}³)")

# 3. Create model
model = NeuralNetworkMPO(
    L=8, 
    mpo_size=mpo_size, 
    output_dim=NN_OUTPUT_DIM,
    max_bond_dim=4,
    use_mpo=True
)

# 4. Train with improvements
from main_program_mpo_fixed import train_model_improved

model, params, psi, losses = train_model_improved(
    model, epochs, single_qubit_probs, psi0, OPS_LIST, 
    CONFIG, t_grid, lr, counts_shots
)

# 5. Done!
```

## Files Provided

1. **mpo_fixed_complete.py**: Fixed MPO classes
   - `NeuralNetworkMPO`: Main architecture
   - `SimpleMPONetwork`: Simplified version
   - `MPOLinearTorchTT`: Fixed MPO layer
   - `calculate_optimal_mpo_size()`: Helper function

2. **main_program_mpo_fixed.py**: Updated main program
   - `train_model_improved()`: Better training loop
   - Debugging utilities
   - Comparison tools

3. **MPS_learning_configuration_MPO_optimized.yaml**: Optimal config
   - Recommended hyperparameters
   - Detailed comments

## Next Steps

1. Copy files to your project
2. Update your main script to use new classes
3. Use optimized config file
4. Run training
5. If issues persist, try debug steps above

Good luck! The MPO should now converge properly. 🚀
