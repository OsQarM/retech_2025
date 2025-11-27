import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from IPython.display import Image
import numpy as np
import os


def plot_expectations(z_data, N, filepath=None):
    # Create simplified figure with only line plot
    fig = plt.figure(figsize=(8, 6), dpi=100)

    # Custom colors
    first_color = '#1f77b4'  # Blue
    last_color = '#d62728'   # Red
    middle_color = '#4a4a4a' # Dark gray

    # ========== SIMPLIFIED LINE PLOT ==========
    ax = fig.add_subplot(111)

    for i in range(N):
        magn = z_data[:,i]
        norm_time = np.linspace(0, 1, len(magn))

        lineprops = {
            'color': first_color if i == 0 else (last_color if i == N-1 else middle_color),
            'lw': 2.5 if i in [0, N-1] else 1.0,
            'alpha': 1.0 if i in [0, N-1] else 0.6,
            'label': r'First spin $(n=0)$' if i == 0 else (r'Last spin $(n={})$'.format(N-1) if i == N-1 else None)
        }

        ax.plot(norm_time, magn, **lineprops)

    # Formatting
    ax.set_xlabel(r'Normalized time $t/\tau_{\mathrm{transfer}}$', fontsize=12)
    ax.set_ylabel(r'Magnetization $\langle Z \rangle$', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=10)

    # Clean grid
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')

    # Simplified legend - only show first and last if many spins
    if N > 10:
        # Only show first and last for clarity
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], 
                  fontsize=10, framealpha=0.9, loc='best')
    else:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

    # Title
    plt.title(r'Spin Magnetization Dynamics ($N={}$)'.format(N), fontsize=14, pad=15)

    # Final layout adjustment
    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
        print(f'Figure saved to {filepath}')

    plt.show()
    return



# Quick plot if you just want to see the comparison
def plot_weights(target, initial, final):
    """Quick bar plot for weight comparison"""
    x = np.arange(len(target))
    width = 0.25
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, target, width, label='Target', alpha=0.7)
    plt.bar(x, initial, width, label='Initial', alpha=0.7)
    plt.bar(x + width, final, width, label='Final', alpha=0.7)
    
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Weights Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_2_expectations(z_list, x_list, z_list_2, x_list_2):
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: z lists
    ax1.plot(z_list, 'b-', label='z_list', linewidth=2)
    ax1.plot(z_list_2, 'r--', label='z_list_2', linewidth=2)
    ax1.set_title('Z Lists')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1,1)
    # Right plot: x lists  
    ax2.plot(x_list, 'g-', label='x_list', linewidth=2)
    ax2.plot(x_list_2, 'm--', label='x_list_2', linewidth=2)
    ax2.set_title('X Lists')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_4_expectations(z_list, x_list, z_list_2, x_list_2, z_list_3, x_list_3, z_list_4, x_list_4):
    # Create side-by-side subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Left plot: z lists
    ax1.plot(z_list, 'b-', label='z_list', linewidth=2)
    ax1.plot(z_list_3, 'r--', label='z_list_2', linewidth=2)
    ax1.set_title('Z Lists')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1,1)
    # Right plot: x lists  
    ax2.plot(x_list, 'g-', label='x_list', linewidth=2)
    ax2.plot(x_list_3, 'm--', label='x_list_2', linewidth=2)
    ax2.set_title('X Lists')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1,1)

    # Left plot: z lists
    ax3.plot(z_list_2, 'b-', label='z_list', linewidth=2)
    ax3.plot(z_list_4, 'r--', label='z_list_2', linewidth=2)
    ax3.set_title('Z Lists')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1,1)
    # Right plot: x lists  
    ax4.plot(x_list_2, 'g-', label='x_list', linewidth=2)
    ax4.plot(x_list_4, 'm--', label='x_list_2', linewidth=2)
    ax4.set_title('X Lists')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1,1)
    plt.tight_layout()
    plt.show()
