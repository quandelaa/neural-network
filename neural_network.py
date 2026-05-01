import numpy as np
import sys

class NeuralNetwork:
    def __init__(self, hidden_layers, len_output, neurons, inputs: list):
        self.inputs = np.array(inputs.copy())
        self.inputs_count = len(inputs)

        self.outputs_count = len_output
        self.weights = []

        self.overall_weights = []
        self.overall_biases = []

        self.layers = hidden_layers
        self.neuron_per_layer = neurons

        self.set_params()

    def set_params(self):
        np.random.seed(1)
        
        if self.layers != 0:
            hidden_weights = [np.random.rand(self.neuron_per_layer, self.neuron_per_layer) - 0.5 for _ in range(1, self.layers)]
            
            self.overall_biases = [np.random.rand(self.neuron_per_layer, 1) - 0.5 for _ in range(1, self.layers)]
            self.overall_weights = [weight for weight in hidden_weights]

        self.overall_biases.insert(0, np.random.rand(self.neuron_per_layer, 1) - 0.5)
        self.overall_biases.append(np.random.rand(self.outputs_count, 1) - 0.5)

        self.overall_weights.insert(0, np.random.rand(self.neuron_per_layer, self.inputs_count) - 0.5)
        self.overall_weights.append(np.random.rand(self.outputs_count, self.neuron_per_layer) - 0.5)

    def forward_propagation(self):
        cur_Z = np.dot(self.overall_weights[0], self.inputs) + self.overall_biases[0]
        cur_A = self.sigmoid(cur_Z)

        for weightset, bias in zip(self.overall_weights[1:-1], self.overall_biases[1:-1]):
            cur_Z = np.dot(weightset, cur_A) + bias
            cur_A = self.sigmoid(cur_Z)

        end_Z = np.dot(self.overall_weights[-1], cur_A) + self.overall_biases[-1]
        end_A = self.sigmoid(end_Z)

        return end_A

    @classmethod
    def sigmoid(cls, Z):
        val = 1 / (1 + np.exp(-Z))

        return val

def main():
    if len(sys.argv) != 5:
        print("Usage: python nn.py <amount_of_hidden_layers> <output_values> <neurons_per_layer> <inputs=x,y,z>")
        return

    inputs = sys.argv[4].split(",")
    polished = [[float(_input)] for _input in inputs]

    nn = NeuralNetwork(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), polished)

    X = nn.forward_propagation()

    print(X)

if __name__ == "__main__":
    main()