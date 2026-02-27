import matplotlib.pyplot as plt

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