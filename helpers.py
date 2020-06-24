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
