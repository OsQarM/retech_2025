# infer.py
import numpy as np
from model import NeuralNetwork

def main():
    # network dimensions (must match training)
    nn = NeuralNetwork(784, 1000, 1000, 10)

    # load trained weights
    nn.weights_ih1 = np.load("weights/weights_ih1.npy")
    nn.bias_ih1    = np.load("weights/bias_ih1.npy")

    nn.weights_h1h2 = np.load("weights/weights_h1h2.npy")
    nn.bias_h1h2    = np.load("weights/bias_h1h2.npy")

    nn.weights_h2o = np.load("weights/weights_h2o.npy")
    nn.bias_h2o    = np.load("weights/bias_h2o.npy")

    # enable MPO inference (second layer only)
    nn.use_mpo = True
    nn.mpo_ready = False   # force fresh MPO build

    # load ONE input (saved during training)
    x = np.load("weights/example_input.npy")

    # run inference
    out = nn.feedforward(x)
    pred = np.argmax(out)

    print("Predicted class:", pred)

if __name__ == "__main__":
    main()


