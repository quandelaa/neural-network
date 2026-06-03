import numpy as np
import math
import sys

class NeuralNetwork:
    def __init__(self, hidden_layers, len_output, neurons, inputs: list):
        self.inp = np.array(inputs.copy())
        self.len_inp = len(inputs)

        self.outputs_count = len_output
        self.weights = []

        self.overall_weights = []
        self.overall_biases = []

        self.layers = hidden_layers
        self.neuron_per_layer = neurons

        self.set_params()

    def set_params(self):
        # np.random.seed(1)
        
        if self.layers != 0:
            hidden_weights = [np.random.rand(self.neuron_per_layer, self.neuron_per_layer) - 0.5 for _ in range(1, self.layers)]
            
            self.overall_biases = [np.random.rand(self.neuron_per_layer, 1) - 0.5 for _ in range(1, self.layers)]
            self.overall_weights = [weight for weight in hidden_weights]

        self.overall_weights.insert(0, np.random.rand(self.neuron_per_layer, self.len_inp) - 0.5)
        self.overall_weights.append(np.random.rand(self.outputs_count, self.neuron_per_layer) - 0.5)

        self.overall_biases.insert(0, np.random.rand(self.neuron_per_layer, 1) - 0.5)
        self.overall_biases.append(np.random.rand(self.outputs_count, 1) - 0.5)

    def forward_propagation(self):
        cur_Z = np.dot(self.overall_weights[0], self.inp) + self.overall_biases[0]
        cur_A = self.sigmoid(cur_Z)

        for weightset, bias in zip(self.overall_weights[1:-1], self.overall_biases[1:-1]):
            cur_Z = np.dot(weightset, cur_A) + bias
            cur_A = self.sigmoid(cur_Z)

        end_Z = np.dot(self.overall_weights[-1], cur_A) + self.overall_biases[-1]
        end_A = self.softmax(end_Z)

        return end_A

    @classmethod
    def sigmoid(cls, Z):
        val = 1 / (1 + np.exp(-Z))

        return val

    @classmethod
    def softmax(cls, Z):
        z_sum = 0

        for num in Z:
            z_sum += math.exp(num[0])

        s_max = np.array([[math.exp(num[0]) / z_sum] for num in Z])

        return s_max

def main():
    if len(sys.argv) != 5:
        print("usage: python neural_network.py <amount_of_hidden_layers> <amount_of_output_values> <neurons_per_hidden_layer> <inputs>")
        return

    inputs = [[float(item)] for item in sys.argv[4].split(",")]

    nn = NeuralNetwork(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), inputs)

    X = nn.forward_propagation()

    print(X)

if __name__ == "__main__":
    main()
