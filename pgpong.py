import numpy as np
import gym
import time

from helpers import Helpers
from helpers import MLP

# hyper-parameters
hidden_layers_count = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

resume = True  # resume from previous checkpoint?
render = False
sleep_for_rendering_in_seconds = 0.02

pixels_count = 80 * 80  # input dimensionality: 80x80 grid
DOWN = 2
UP = 3


def save_agent():
    if episode_number % 100 == 0:
        policy_network.save_network()


def train_agent():
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
        policy_network.train(learning_rate, decay_rate)


def modify_gradient(states, hidden_layers, dlogps, discounted_rewards):
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    episode_states = np.vstack(states)
    episode_hidden_layers = np.vstack(hidden_layers)
    epdlogp = np.vstack(dlogps)
    episode_discounted_rewards = np.vstack(discounted_rewards)
    # compute the discounted reward backwards through time
    discounted_epr = Helpers.discount_and_normalize_rewards(episode_discounted_rewards, gamma)
    epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
    policy_network.backward_pass(episode_hidden_layers, epdlogp, episode_states)


def render_game():
    if render:
        env.render()
        time.sleep(sleep_for_rendering_in_seconds)


def get_frame_difference():
    # pre-process the observation, set input to network to be difference image
    current_frame = Helpers.preprocess_frame(observation)
    state = current_frame - previous_frame if previous_frame is not None else np.zeros(pixels_count)
    return state, current_frame


if __name__ == '__main__':
    env = gym.make("Pong-v0")
    observation = env.reset()
    previous_frame, running_reward = None, None  # used in computing the difference frame
    states, hidden_layers, dlogps, discounted_rewards = [], [], [], []
    reward_sum = 0
    episode_number = 0

    policy_network = MLP(input_count=6400, hidden_layers_count=200)

    if resume:
        policy_network.load_network()

    while True:
        render_game()
        state, previous_frame = get_frame_difference()

        # forward the policy network and sample an action from the returned probability
        action_probability_space, hidden_layer = policy_network.forward_pass(state)
        action = DOWN if np.random.uniform() < action_probability_space else UP  # roll the dice!

        # record various intermediates (needed later for backprop)
        states.append(state)
        hidden_layers.append(hidden_layer)
        y = 1 if action == 2 else 0  # a "fake label"

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        dlogps.append(y - action_probability_space)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        discounted_rewards.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

        if done:  # an episode finished
            episode_number += 1

            modify_gradient(states, hidden_layers, dlogps, discounted_rewards)
            train_agent()
            save_agent()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            observation = env.reset()
            reward_sum = 0
            previous_frame = None
            states, hidden_layers, dlogps, discounted_rewards = [], [], [], []
