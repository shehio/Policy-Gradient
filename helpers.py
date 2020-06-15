import numpy as np


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
    def discount_rewards(r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    @staticmethod
    def policy_forward(model, x):
        h = np.dot(model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(model['W2'], h)
        p = Helpers.sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    @staticmethod
    def policy_backward(model, eph, epdlogp, epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0  # backprop relu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}
