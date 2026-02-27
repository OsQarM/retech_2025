import numpy as np
import torch
from torchvision import transforms, datasets
import tensorkrowch as tk

import jax
import jax.numpy as jnp

import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import glob
import yaml
import matplotlib.pyplot as plt
import torchtt as tntt 

import os


'''Script to learn distribution of Multiverse model

This program takes a trained MLP model and an example input. The MLP has one hidden layer that can be compressed into an MPO
The structure of the MLP is:

Input (784)  
→ Dense  
Hidden Layer 1 (1000)  
→ MPO (TenPy)  
Hidden Layer 2 (1000)  
→ Dense  
Output (10)

The program first performs inference with or without the tensorized layer and treats the output parameters as a quantum state,
that is, it encodes the 10 classes in a 4-qubit state. The objective of the Hamiltonian learning is to find which Hamiltonian
Generates this state from the input. 

The Hamiltonian needs to be able to process the input, so we encode the 784 parameters in a 10-qubit quantum state. Then the Hamiltonian
Evolves this state into the output state where only 10 parameters matter (even though this makes most of the information of the output state unnecessary)

These are the steps of the program:

1. Load the MLP and the input
2. Perform inference
3. Create 4-qubit state from output
4. At this point we chan chose between two options:
    a) Train a 4-qubit Hamiltonian from an input of |0000> (made for testing or alternative result in case option b) fails)
    b) Train a 10-qubit Hamiltonian from an input encoded by the MLP example
5. Build input state from list of input parameters (option a) or list of zeros (option b)
6. Create NN model for HL learning with tensorkrowch, taking a random input since it doesn't matter much
7. Train the Network
8. Save results
9. Validate results

'''





