import numpy as np
import pickle

class MLP:
    def __init__(self, input_count, hidden_layers_count, output_count=1):
        self.model = {'W1': np.random.randn(hidden_layers_count, input_count) / np.sqrt(input_count),
                      'W2': np.random.randn(hidden_layers_count) / np.sqrt(hidden_layers_count)}

        # update buffers that add up gradients over a batch
        self.gradient_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()} # rmsprop memory

    def forward_pass(self, input):
        hidden_layer = np.dot(self.model['W1'], input)
        hidden_layer[hidden_layer < 0] = 0  # ReLU non-linearity
        logp = np.dot(self.model['W2'], hidden_layer)
        output = self.__sigmoid(logp)
        return output, hidden_layer  # return probability of taking action 2, and hidden state

    def backward_pass(self, eph, epdlogp, epx):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # back-prop ReLU
        dW1 = np.dot(dh.T, epx)
        current_gradient = {'W1': dW1, 'W2': dW2}

        for k in self.model:
            self.gradient_buffer[k] += current_gradient[k]  # accumulate grad over batch

    def load_network(self, file_name='native_mlp.p'):
        self.model = pickle.load(open(file_name, 'rb'))

    def save_network(self, file_name='native_mlp.p'):
        pickle.dump(self.model, open(file_name, 'wb'))

    def train(self, learning_rate, decay_rate):
        for k, v in self.model.items():
            g = self.gradient_buffer[k]
            self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            self.model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            self.gradient_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

    def __sigmoid(self, number):
        return 1.0 / (1.0 + np.exp(-number))  # sigmoid "squashing" function to interval [0,1]
