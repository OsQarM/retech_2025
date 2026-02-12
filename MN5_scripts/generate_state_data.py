import numpy as np
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml


#########################################
#1. UTILS
#########################################

def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
  

def print_run_info(config):
    print(f"Generating data for MPS of size {config['L']}, " \
          f"with bond dimension {config['bond_dimension']} and values " \
          f"between {config['min_val']} and {config['max_val']} " \
          f"and {config['N_shots']} shots")
    return


#########################################
#2 STATE GENERATION
#########################################


def create_state(L, kind, dtype=np.complex64):
    """Prepare initial quantum states for L qubits."""

    if kind == 'zeros':
        psi = np.zeros(2**L, dtype=dtype)
        psi[0]  = 1.0

    elif kind == 'plus':
        plus = np.ones(2, dtype=dtype) / np.sqrt(2)
        psi = plus
        for _ in range(L - 1):
            psi = np.kron(psi, plus)
    
    elif kind == 'ghz':
        psi = np.zeros(2**L, dtype=dtype)
        psi[0]  = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
            
    else:
        raise ValueError(f"Initial state '{kind}' not recognized. "
                        f"Use 'all_zeros' or 'all_plus'")
    return psi

def generate_bitstring_list(nqubits):
    '''Create list containing all possible bitstrings of the N-qubit chain'''
    bitstrings = []
    decimal_bitstrings = range(0, 2**nqubits)
    int_bitstrings = [bin(i)[2:].zfill(nqubits) for i in decimal_bitstrings]
    bitstrings =  [str(bit) for bit in int_bitstrings]

    return bitstrings

def get_amplitude(psi, string, N, dtype = np.complex64):
    '''For each bitstring, calculate the amplitude of the MPS'''
    psi_string = np.zeros(2, dtype=dtype)
    psi_string[0 if string[0] == '0' else 1] = 1.0

    for bit in string[1:]:
        psi_bit = np.zeros(2, dtype=dtype)
        psi_bit[0 if bit == '0' else 1] = 1.0
        psi_string = np.kron(psi_string, psi_bit)
    
    # Final contraction gives scalar
    amplitude = float(np.vdot(psi_string, psi))
    
    return amplitude

def extract_amplitudes(psi, N):
    num_qubits = N
    bitstrings = generate_bitstring_list(num_qubits)
    probs = []
    for string in bitstrings:
        amp = get_amplitude(psi, string, num_qubits)
        probs.append(amp**2)

    total_prob = sum(probs)
    normalized_probs = [p/total_prob for p in probs]
    return bitstrings, normalized_probs


def sample_from_probs(nqubits, N_shots, probs, seed):
    '''Generates finite sample from list of bistring probabilities'''
    dim = 2**nqubits
    counts_shots = []
    rng = np.random.default_rng(seed)
    samples = rng.choice(dim, size=N_shots, p=probs)
    hist = np.bincount(samples, minlength=dim)
    counts_shots.append(hist)

    return counts_shots


#########################################
#3. SAVING AND PLOTTING
#########################################


def save_sample_bitstrings(bitstrings, counts, kind, prefix = None):
    #Create config file and save here L, chi, Nshots
    filename_core = f"L{N}_{kind}_R{N_shots}"

    filename = f'./data/experimental_data_{prefix}_{filename_core}_counts.csv'

    bitstrings_with_quotes = ["'" + bs for bs in bitstrings]
    
    df = pd.DataFrame({'bitstring': bitstrings_with_quotes, 'count': counts[0]})
    df.to_csv(filename, index=False)

    print(f"Saved to {filename}")
    return filename


def bar_plot_strings(strings, values, title="Bar Plot", xlabel="Bitstrings", ylabel="Probability", 
                     color='skyblue', edgecolor='black', figsize=(10, 6)):
    """
    Bar plot with string labels on x-axis.
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    x_pos = np.arange(len(strings))
    bars = ax.bar(x_pos, values, color=color, edgecolor=edgecolor, alpha=0.8)
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-ticks to string labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strings, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    filename_core = f"L{N}_Chi_{kind}_R{N_shots}"

    filename = f'./bitstring_histogram_{filename_core}'
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', bbox_inches='tight', dpi=300)
    
    return fig, ax, bars



#########################################
#4. MAIN PROGRAM
#########################################


if __name__ == "__main__":

    #choose configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "./MPS_data_generation.yaml"

    #load configuration
    print(config_file)
    CONFIG = load_config(config_file)

    # Test
    seed = CONFIG['seed_data']
    N = CONFIG['L']
    N_shots = CONFIG['N_shots']
    kind = CONFIG['state_type']
    print_run_info(CONFIG)

    psi = create_state(N, kind)

    print(len(psi))

    bitstrings, probs = extract_amplitudes(psi, N)

    counts_shots = sample_from_probs(N, N_shots, probs, seed)

    print("Bitstring counts:", counts_shots[0])

    files = save_sample_bitstrings(
        bitstrings=bitstrings,
        counts=counts_shots,
        kind = kind,
        prefix="quantum_sampling"
        )
    
    bar_plot_strings(bitstrings, probs, "Bitstring probabilities")


#########################################
#########################################