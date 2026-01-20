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
from figures import plot_noise_parameters, plot_hamiltonian_parameters, plot_mixed_state_fidelity
from figures import plot_purity, plot_pure_state_fidelity, plot_observables, plot_training_loss, plot_final_probabilities
from operators import OperatorClass

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
    search_pattern = f"../data/experimental_data_quantum_sampling_L{N}_Chi_{chi}_*_counts.csv"
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No data found for L={N}")

    config_file = files[0]
    file_core = config_file.replace(".csv", "").replace("../data/experimental_data_quantum_sampling_", "")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {file_core}")
    print(f"{'='*60}")
    
    df_counts = pd.read_csv(f"../data/experimental_data_quantum_sampling_{file_core}.csv")
        
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
        config_file = "/Users/omichel/Desktop/qilimanjaro/projects/retech/retech_2025/refactored_scripts/config/MPS_learning_configuration.yaml"

    #load configuration
    print(config_file)
    CONFIG = load_config(config_file)

    # Load data
    bitstrings, counts_shots = load_experimental_data(CONFIG)

    L = CONFIG["L"]
    dim = 2**L
    hamiltonian_type = CONFIG["hamiltonian_type"] #Probably not necessary
    use_noisy = CONFIG["use_noisy_dynamics"]

    # Prepare initial state
    state0 = prepare_initial_state(L, CONFIG["initial_state_kind"], 
                                    as_density_matrix=use_noisy)
    

    
    OPS_XYZ = OperatorClass(L)
    if hamiltonian_type == 'uniform_xyz' or hamiltonian_type == 'general_local_zz':
        # Build operators
        OPS_XYZ.operators = build_xyz_basis(L, hamiltonian_type)           
        #Then call a series of functions to add basis terms (the ones we desire for our anatz)
        #Then extract num coefficients from the H we have created
        NUM_COEFFICIENTS = get_theta_shape(L, hamiltonian_type)

    elif hamiltonian_type == 'custom':
        #Optional:Create custom H with the desired terms
        #then we have to create a theta list with all the initial parameters
        OPS_XYZ.add_operators('X')
        OPS_XYZ.add_operators('Z')
        OPS_XYZ.add_operators('ZZ')

        NUM_COEFFICIENTS = len(OPS_XYZ)
        print(f"Working with {NUM_COEFFICIENTS} Hamiltonian parameters")
    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

    # Build Lindblad operators if needed
    if use_noisy:
        dephasing_ops, damping_ops = build_lindblad_operators(L)
    else:
        dephasing_ops, damping_ops = None, None


    print_training_info(CONFIG, NUM_COEFFICIENTS)

    t_max = CONFIG['t_max']
    dt = CONFIG['dt']
    t_grid_shots = np.linspace(0., t_max, int(t_max/dt) +1)


    # Initialize NN
    NN_MAP_FUN = mlp_forward
    NN_INPUT_DIM = 1  # Time-dependent
    NN_OUTPUT_DIM = NUM_COEFFICIENTS  # NN only outputs Hamiltonian corrections

    layer_sizes = [NN_INPUT_DIM] + CONFIG["NN_hidden_sizes"] + [NN_OUTPUT_DIM]
    key = random.PRNGKey(CONFIG["seed_init"])
    key, k_nn, k_th, k_noise, k_init = random.split(key, 5)
    nn_params = init_mlp_params(layer_sizes, k_nn, scale=0.1)

    # Initialize Hamiltonian parameters
    if hamiltonian_type == "uniform_xyz":
        theta_init_list = [CONFIG["Jx_init"], CONFIG["Jy_init"], CONFIG["Jz_init"],
                            CONFIG["hx_init"], CONFIG["hy_init"], CONFIG["hz_init"]]
    elif hamiltonian_type == "general_local_zz":
        theta_init_list = (list(CONFIG["hx_list_init"]) + 
                            list(CONFIG["hz_list_init"]) + 
                            list(CONFIG["Jzz_list_init"]))
    elif hamiltonian_type == "custom":
        theta_init_list = jax.random.uniform(k_init, (NUM_COEFFICIENTS), dtype=jnp.float32)

    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

    theta_init = jnp.array(theta_init_list, dtype=jnp.float32)
    if CONFIG["INIT_PERTURB_SCALE"] > 0:
        theta_init += CONFIG["INIT_PERTURB_SCALE"] * random.normal(k_th, (NUM_COEFFICIENTS,))

    # Initialize noise parameters if needed
    params = {"theta": theta_init, "nn": nn_params}

    if use_noisy and CONFIG["learn_noise_params"]:
        if CONFIG["noise_model"] == "global":
            # Single rate for all qubits
            noise_rates_init = jnp.array([
                CONFIG["gamma_dephasing_init"],
                CONFIG["gamma_damping_init"]
            ], dtype=jnp.float32)
        else:  # local
            # Per-qubit rates
            noise_rates_init = jnp.concatenate([
                jnp.full((L,), CONFIG["gamma_dephasing_init"], dtype=jnp.float32),
                jnp.full((L,), CONFIG["gamma_damping_init"], dtype=jnp.float32)
            ])
        params["noise_rates"] = noise_rates_init
        print(f"  Initial noise rates: {noise_rates_init}")

    # Create time grids
    t_grid_fine = jnp.arange(0.0, CONFIG["t_max"] + CONFIG["dt"]/2, CONFIG["dt"])
    T_extrap = CONFIG["t_max"] * CONFIG["T_extrapolate_factor"]
    t_grid_long = jnp.arange(0.0, T_extrap + 1e-12, CONFIG["dt"])

    # Create step function (here we need to examine the shape of OPS)
    #1. when createing xyz_hamiltonian from theta:
        #params is a dictionary with one key that is theta
        #theta is a list of lists that contains the starting parameters for all the qubits
        #OPS_XYZ is a single list with all the operators for every qubit
        #we build H by multiplying theta by OPS_XYZ
    #2. When building NN part it doesn't care about structure, just number of coeffs

    step_fn = make_step_fn(CONFIG, L, OPS_XYZ, NN_MAP_FUN, use_noisy,
                            dephasing_ops, damping_ops)

    # Training schedule
    N_total = CONFIG["N_epochs"]
    P1 = int(N_total * CONFIG["PHASE1_SPLIT"])
    P2 = int(N_total * CONFIG["PHASE2_SPLIT"])
    P3 = N_total - P1 - P2

    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    print(f"  Phase 1 (Warm-up): {P1} epochs")
    print(f"  Phase 2 (Distill): {P2} epochs")
    print(f"  Phase 3 (Refine): {P3} epochs")


    print("\n=== DEBUGGING PARAMETER MISMATCH ===")
    print(f"L (system size): {L}")
    print(f"Hamiltonian type: {hamiltonian_type}")
    print(f"Expected theta shape: {NUM_COEFFICIENTS}")
    print(f"Actual theta init length: {len(theta_init)}")
    print(f"OPS_XYZ length: {len(OPS_XYZ)}")

    # Phase 1: Train everything
    learn_noise_p1 = CONFIG["learn_noise_params"] if use_noisy else False
    params, l1 = train_phase(params, P1, CONFIG, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=CONFIG["learn_theta"],
                            train_nn=True, train_noise=learn_noise_p1,
                            phase_name="P1 Warm-up")

    # Phase 2: Distill to theta (freeze NN and noise)
    params, l2 = train_phase(params, P2, CONFIG, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=CONFIG["learn_theta"],
                            train_nn=False, train_noise=False,
                            phase_name="P2 Distill")

    # Phase 3: Refine NN (freeze theta, optionally train noise)
    learn_noise_p3 = CONFIG["learn_noise_params"] if use_noisy else False
    params, l3 = train_phase(params, P3, CONFIG, step_fn, t_grid_shots, state0,
                            counts_shots, train_theta=False,
                            train_nn=True, train_noise=learn_noise_p3,
                            phase_name="P3 Refine")

    losses = l1 + l2 + l3

    # Extract final parameters
    theta_final = np.array(jax.device_get(params["theta"]))
    nn_l2_norm = sum(jnp.sum(p**2) for p in jtu.tree_leaves(params["nn"]))


    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")

    print_hamiltonian_parameters(CONFIG, hamiltonian_type, theta_final, nn_l2_norm)

    # Print noise parameters if learned
    if "noise_rates" in params:
        print_noise_parameters(CONFIG, params)

    probs_model, probs_vanilla = extract_learned_mps_ham_probs(CONFIG, OPS_XYZ, NN_MAP_FUN, state0, t_grid_fine, params)

    plot_final_probabilities(bitstrings, counts_shots, probs_model, probs_vanilla, labels = ['Data counts', 'NN', 'Vanilla'])

    # 5. Training loss
    try:
        plot_training_loss(losses)
    except Exception as e:
        print(f"  Warning: failed to plot training loss: {e}")
