# infer.py
import numpy as np
from model import NeuralNetwork

def run_inference(x, weights_path, use_mpo = True, max_bond = 128):
    # network dimensions (must match training)
    nn = NeuralNetwork(784, 1000, 1000, 10, max_bond)

    # load trained weights
    nn.weights_ih1 = np.load(f"{weights_path}weights_ih1.npy")
    nn.bias_ih1    = np.load(f"{weights_path}bias_ih1.npy")

    nn.weights_h1h2 = np.load(f"{weights_path}weights_h1h2.npy")
    nn.bias_h1h2    = np.load(f"{weights_path}bias_h1h2.npy")

    nn.weights_h2o = np.load(f"{weights_path}weights_h2o.npy")
    nn.bias_h2o    = np.load(f"{weights_path}bias_h2o.npy")

    # enable MPO inference (second layer only)
    nn.use_mpo = use_mpo
    nn.mpo_ready = False   # force fresh MPO build

    # run inference
    out = nn.feedforward(x)
    pred = np.argmax(out)

    print("Predicted class:", pred)

    return out.flatten(), pred


