

import torch
import torch.nn as nn



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