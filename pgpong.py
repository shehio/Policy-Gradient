""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import time

from helpers import Helpers

# hyperparameters
hidden_layers_count = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

resume = True  # resume from previous checkpoint?
render = True
sleep_for_rendering_in_seconds = 0.02

# model initialization
pixels_count = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {'W1': np.random.randn(hidden_layers_count, pixels_count) / np.sqrt(pixels_count),
             'W2': np.random.randn(hidden_layers_count) / np.sqrt(hidden_layers_count)}

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    while True:
        if render:
            env.render()
            time.sleep(sleep_for_rendering_in_seconds)

        # pre-process the observation, set input to network to be difference image
        cur_x = Helpers.preprocess_frame(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(pixels_count)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = Helpers.policy_forward(model, x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x)  # observation
        hs.append(h)  # hidden state
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = Helpers.discount_rewards(epr, gamma)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            grad = Helpers.policy_backward(model, eph, epdlogp, epx)
            for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.items():
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            if episode_number % 100 == 0:
                pickle.dump(model, open('save.p', 'wb'))

            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
