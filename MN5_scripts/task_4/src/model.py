# model.py
import numpy as np
from mpo_tenpy import MPOLinearTorchTT


class NeuralNetwork:
    def __init__(self, num_inputs, num_hiddenNodes1, num_hiddenNodes2, num_outputs):
        self.num_inputs = num_inputs
        self.num_hiddenNodes1 = num_hiddenNodes1
        self.num_hiddenNodes2 = num_hiddenNodes2
        self.num_outputs = num_outputs

        self.weights_ih1 = np.zeros((num_hiddenNodes1, num_inputs))
        self.weights_h1h2 = np.zeros((num_hiddenNodes2, num_hiddenNodes1))
        self.weights_h2o = np.zeros((num_outputs, num_hiddenNodes2))

        self.bias_ih1 = np.zeros((num_hiddenNodes1, 1))
        self.bias_h1h2 = np.zeros((num_hiddenNodes2, 1))
        self.bias_h2o = np.zeros((num_outputs, 1))

        self.mpo = MPOLinearTorchTT(factors=[10, 10, 10], max_bond=128)
        self.use_mpo = False
        self.mpo_ready = False

    def setup(self):
        # input -> hidden1
        self.weights_ih1 = np.random.normal(
            0.0, self.num_inputs ** -0.5,
            (self.num_hiddenNodes1, self.num_inputs)
        )
        self.bias_ih1 = np.random.normal(
            0.0, self.num_inputs ** -0.5,
            (self.num_hiddenNodes1, 1)
        )

        # hidden1 -> hidden2
        self.weights_h1h2 = np.random.normal(
            0.0, self.num_hiddenNodes1 ** -0.5,
            (self.num_hiddenNodes2, self.num_hiddenNodes1)
        )
        self.bias_h1h2 = np.random.normal(
            0.0, self.num_hiddenNodes1 ** -0.5,
            (self.num_hiddenNodes2, 1)
        )

        # hidden2 -> output
        self.weights_h2o = np.random.normal(
            0.0, self.num_hiddenNodes2 ** -0.5,
            (self.num_outputs, self.num_hiddenNodes2)
        )
        self.bias_h2o = np.random.normal(
            0.0, self.num_hiddenNodes2 ** -0.5,
            (self.num_outputs, 1)
        )

        self.mpo_ready = False

    # ---------- forward ----------
    def feedforward(self, inputs):
        inputs = np.asarray(inputs).reshape(self.num_inputs, 1)

        self.hidden1 = self.sigmoid(self.weights_ih1 @ inputs + self.bias_ih1)

        if self.use_mpo:
            if not self.mpo_ready:
                self.mpo.init_from_weights(self.weights_h1h2, self.bias_h1h2)
                self.mpo_ready = True
            self.hidden2 = self.mpo.forward(self.hidden1)
        else:
            self.hidden2 = self.weights_h1h2 @ self.hidden1 + self.bias_h1h2

        self.hidden2 = self.sigmoid(self.hidden2)
        self.outputs = self.sigmoid(self.weights_h2o @ self.hidden2 + self.bias_h2o)
        return self.outputs

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1 - x)

