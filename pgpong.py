import numpy as np
import gym
import time

from agent import Agent
from helpers import Helpers

# hyper-parameters
hidden_layers_count = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

render = True
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
    env = gym.make("ALE/Pong-v5", render_mode="human")
    observation, _ = env.reset()
    previous_frame, running_reward = None, None  # used in computing the difference frame
    reward_sum = 0
    episode_number = 0
    points_scored = 0
    points_conceeded = 0
    agent = Agent(learning_rate, decay_rate, load_network=True)

    while True:
        render_game()
        state, previous_frame = get_frame_difference()
        action = agent.sample_and_record_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.reap_reward(reward)
        reward_sum += reward

        if reward == 1:
            points_scored += 1
            print('ep %d: point scored, score: %f: %f' % (episode_number, points_conceeded, points_scored))
        elif reward == -1:
            points_conceeded += 1
            print('ep %d: point conceeded, score: %f: %f' % (episode_number, points_conceeded, points_scored))
        # else:
        #     print(reward)
        #     print('ep %d: nothing happened, score: %f: %f' % (episode_number, points_conceeded, points_scored))

        if done:
            agent.make_episode_end_updates(episode_number)

            running_reward = reward_sum if running_reward is None else running_reward * (episode_number - 1) / episode_number + reward_sum / episode_number
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

            observation, _ = env.reset()
            reward_sum = 0
            points_scored = 0
            points_conceeded = 0
            episode_number += 1
            previous_frame = None
