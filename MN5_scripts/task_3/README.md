## Hamiltonian learning in HPC


This directory contains the scripts necessary to generate simulated data of a time evolution, and a ML algorithm to learn the Hamiltonian that generates de trajectories of the simulated data.

First, the required libraries are:

- numpy
- matplotlib
- pandas
- pyyaml
- jax

The folder 'scripts' contains the two .py scripts for the data generation and Hamiltonian learning tasks. They each use a configuration file located in the 'config' folder. The first script: '_01_generate_data.py' stores the data file and its metadata in the folder 'data'. The second script: '_02_LL_MLP_refactored.py' reads from these files and stores the output in the 'results' and 'plots' folder.

Important: Both scripts print information on the terminal that is also relevant (when running on HPC it would be good to save the .out files of the job executions)

Here are the instructions to run a simple validation of the program:

1. To generate data, all the relevant parameters are in the configuration file 'data_generation_configuration.yaml'. The important ones are:
    - L: System size (relevant for a scalability test in HPC)
    - N_shots_per_time: measurement sample size. If using large L, it needs to be increased from the default value
    - hamiltonian_type: We tested two kinds of Hamiltonian. The parameters under allow to configure the parameters of the chosen Hamiltonian
    - dynamics_type: Allows to run model with or without noise. The rest of the parameters allow for the configuration of the noise model

    For a simple test we recommend leaving the hamiltonian_type as 'uniform_xyz' and dynamics_type as 'schrodinger'. After finishing with the configuration, the only thing needed is to run the first script with 'python _01_generate_data.py'

2. The ML script contains some more configuration options in the file 'lindbladian_learning_configuration.yaml'. These are the most important:
    - L: Again, this is the system size. But you have to make sure that there is an existing data file with that size already
    - After that, comes the hamiltonian_type (again, make sure data has been generated before), and configuration of noise parameters (if noisy data has been generated)
    - The rest of the parameters have to do with initialization and training parameters, which in principle work well for all tests performed until now

    The second script is ran with 'python _02_LL_MLP_refactored.py', and it should store all the output properly

Note: In both scripts, the paths to other folders are specified relatively to the location of the script. This should in principle simplify the execution in any environment, as long as all the files can be located.


'Advanced' tips:

- When running with the hamiltonian_type: 'general_local_zz', the length of the lists of initial parameters and data generation parameters must match the size of the system being built L. (This is specified also in the configuration files)

- Same for list of T1 and T2 when running noisy model with 'local' parameters, which means each qubit can have different T1 and T2.

- Local learning of noise parameters doesn’t work very well. It needs strong regularization set by the parameter 'noise_reg'




