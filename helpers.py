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
        output = Helpers.sigmoid(logp)
        return output, hidden_layer  # return probability of taking action 2, and hidden state

    def backward_pass(self, eph, epdlogp, epx):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # back-prop ReLU
        dW1 = np.dot(dh.T, epx)
        current_gradient = {'W1': dW1, 'W2': dW2}

        for k in self.model:
            self.gradient_buffer[k] += current_gradient[k]  # accumulate grad over batch

    def load_network(self, file_name='save.p'):
        self.model = pickle.load(open(file_name, 'rb'))

    def save_network(self, file_name='save.p'):
        pickle.dump(self.model, open(file_name, 'wb'))

    def train(self, learning_rate, decay_rate):
        for k, v in self.model.items():
            g = self.gradient_buffer[k]  # gradient
            self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            self.model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            self.gradient_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer


class Helpers:
    @staticmethod
    def sigmoid(number):
        return 1.0 / (1.0 + np.exp(-number))  # sigmoid "squashing" function to interval [0,1]

    @staticmethod
    def preprocess_frame(image_frame):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        image_frame = image_frame[35:195]  # crop
        image_frame = image_frame[::2, ::2, 0]  # downsample by factor of 2
        image_frame[image_frame == 144] = 0  # erase background (background type 1)
        image_frame[image_frame == 109] = 0  # erase background (background type 2)
        image_frame[image_frame != 0] = 1  # everything else (paddles, ball) just set to 1
        return image_frame.astype(np.float).ravel()

    @staticmethod
    def discount_and_normalize_rewards(r, gamma):
        discounted_rewards = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards
