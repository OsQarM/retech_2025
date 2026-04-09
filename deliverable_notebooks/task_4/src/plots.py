import matplotlib.pyplot as plt
import numpy as np

def bar_plot_strings_comparison(strings, values1, values2, config, L=10, labels=None, 
                                title="Distribución de probabilidad, L=6", xlabel="Bitstrings", 
                                ylabel="Probabilidad", colors=None, edgecolor='black', 
                                figsize=(12, 7), style='grouped', alpha=0.8,
                                y_tick_fontsize=14, x_tick_fontsize=14):
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
    y_tick_fontsize : int
        Fontsize for y-axis tick labels
    x_tick_fontsize : int
        Fontsize for x-axis tick labels
    """
    
    if labels is None:
        labels = ('Real', 'Aprendida')
    if colors is None:
        colors = ('skyblue', 'salmon')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    n = len(strings)
    x_pos = np.arange(n)
    width = 0.35  # Width of bars
    
    # Determine number of x-tick labels based on number of bins (strings)
    # Goal: have approximately n/4 labels for n=2^L, but ensure at least 2 labels
    # For 4 qubits (16 bins): ~8 labels, for 6 qubits (64 bins): ~12-16 labels
    n_bins = len(strings)
    
    # Calculate desired number of labels (aim for ~n_bins/2 but with max reasonable)
    if n_bins <= 16:
        # For small systems, show all labels
        n_labels = n_bins
        tick_indices = np.arange(n_bins)
        tick_labels = strings
    else:
        # For larger systems, show approximately sqrt(n_bins) to n_bins/4 labels
        # Aim for 8-16 labels depending on size
        if n_bins <= 32:  # 5 qubits
            n_labels = 8
        elif n_bins <= 64:  # 6 qubits
            n_labels = 12
        elif n_bins <= 128:  # 7 qubits
            n_labels = 16
        elif n_bins <= 256:  # 8 qubits
            n_labels = 20
        else:  # 9+ qubits
            n_labels = 24
        
        # Ensure we don't try to show more labels than bins
        n_labels = min(n_labels, n_bins)
        
        # Calculate equally spaced indices
        tick_indices = np.linspace(0, n_bins - 1, n_labels, dtype=int)
        tick_labels = [strings[i] for i in tick_indices]
    
    if style == 'grouped':
        # Side-by-side bars
        bars1 = ax.bar(x_pos - width/2, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=alpha)
        bars2 = ax.bar(x_pos + width/2, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=alpha)
        
        ax.set_xticks(x_pos[tick_indices])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=x_tick_fontsize)
        
    elif style == 'stacked':
        # Stacked bars
        bars1 = ax.bar(x_pos, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=alpha)
        bars2 = ax.bar(x_pos, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=alpha,
                      bottom=values1)
        
        ax.set_xticks(x_pos[tick_indices])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=x_tick_fontsize)
        
    elif style == 'overlap':
        # Overlapping transparent bars
        bars1 = ax.bar(x_pos, values1, width, 
                      label=labels[0], color=colors[0], 
                      edgecolor=edgecolor, alpha=0.6)
        bars2 = ax.bar(x_pos, values2, width, 
                      label=labels[1], color=colors[1], 
                      edgecolor=edgecolor, alpha=0.6)
        
        ax.set_xticks(x_pos[tick_indices])
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=x_tick_fontsize)
    
    # Customize plot
    ax.set_title(f"Distribución de probabilidad, L={L}", fontsize=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    # Set y-tick label fontsize
    ax.tick_params(axis='y', labelsize=y_tick_fontsize)
    
    # Add legend
    ax.legend(fontsize=18, framealpha=0.9)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    N = config['L']
    chi_data = config['bond_dimension_data']
    chi_nn = config['bond_dimension_learning']
    kind = config['data_kind']
    nn_type = config['NN_TYPE']
    filename_core = f"L{N}_nn-{nn_type}_kind-{kind}_Chidata{chi_data}_ChiNN{chi_nn}"
    filename = f'./bitstring_comparison_{filename_core}'
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'../plots/{filename}.png', bbox_inches='tight', dpi=300)
    
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
    plt.savefig(f'../plots/{filename}.png', bbox_inches='tight', dpi=300)