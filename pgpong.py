import numpy as np
import gym
import time

from agent import Agent
from helpers import Helpers
from memory import Memory

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
    reward_sum = 0
    episode_number = 0
    agent = Agent(learning_rate, decay_rate)

    while True:
        render_game()
        state, previous_frame = get_frame_difference()
        action = agent.get_action(state)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        agent.memory.rewards.append(reward)

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

        if done:
            episode_number += 1
            agent.modify_gradient()
            agent.episode_updates(episode_number)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            observation = env.reset()
            reward_sum = 0
            previous_frame = None
