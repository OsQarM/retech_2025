import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import copy 
import pandas as pd
import glob
import sys
import yaml

sys.path.append('../src/')

from model_building import get_theta_shape, build_xyz_basis, build_lindblad_operators, prepare_initial_state
from diagnostics import print_training_info, print_hamiltonian_parameters, print_noise_parameters, extract_learned_mps_ham_probs, print_relative_error, generate_diagnostic_trajectories
from mlp import init_mlp_params, mlp_forward, make_step_fn, train_phase
from figures import plot_noise_parameters, plot_hamiltonian_parameters, plot_mixed_state_fidelity, plot_purity, plot_pure_state_fidelity, plot_observables, plot_training_loss, plot_final_probabilities

Array = jnp.ndarray



def load_config(config_path):
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_experimental_data(config):
    """Load experimental/simulated data"""
    N = config["L"]
    chi = config['bond_dimension']
    T_max = config["t_max"]
    search_pattern = f"experimental_data_quantum_sampling_L{N}_Chi_{chi}_*_counts.csv"
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No data found for L={N}")

    config_file = files[0]
    file_core = config_file.replace(".csv", "").replace("experimental_data_quantum_sampling_", "")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {file_core}")
    print(f"{'='*60}")
    
    df_counts = pd.read_csv(f"experimental_data_quantum_sampling_{file_core}.csv")
        
    # Remove leading single quote if present
    if df_counts['bitstring'].astype(str).str.startswith("'").all():
        df_counts['bitstring'] = df_counts['bitstring'].str[1:]
    
    # Now extract values
    bitstrings = df_counts['bitstring'].values.astype(str)
    counts_shots = df_counts['count'].values.astype(np.int32)
    
    return bitstrings, counts_shots



if __name__ == "__main__":

    #choose configuration file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "/Users/omichel/Desktop/qilimanjaro/projects/retech/retech_2025/config_files/data_generation_configuration.yaml"

    #load configuration
    print(config_file)
    CONFIG = load_config(config_file)

