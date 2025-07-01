import gym
import time
import numpy as np
from helpers import Helpers

class Game:
    def __init__(self, game_name, render, sleep_for_rendering_in_seconds, pixels_count):
        self.game_name = game_name
        self.render_flag = render
        self.sleep_for_rendering_in_seconds = sleep_for_rendering_in_seconds
        self.pixels_count = pixels_count
        self.env = self.get_environment()
        self.observation, _ = self.env.reset()
        self.previous_frame = None
        self.running_reward = 0.0
        self.reward_sum = 0.0
        self.episode_number = 0
        self.points_scored = 0
        self.points_conceeded = 0

    def get_environment(self):
        if self.render_flag:
            return gym.make(self.game_name, render_mode="human")
        else:
            return gym.make(self.game_name)

    def render(self):
        if self.render_flag:
            self.env.render()
            time.sleep(self.sleep_for_rendering_in_seconds)

    def end_episode(self):
        self.running_reward = self.running_reward * (self.episode_number - 1) / self.episode_number + self.reward_sum / self.episode_number if self.episode_number > 0 else self.reward_sum
        print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()

    def reset(self):
        self.observation, _ = self.env.reset()
        self.previous_frame = None
        self.reward_sum = 0.0
        self.points_scored = 0
        self.points_conceeded = 0
        return self.observation

    def step(self, action):
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self.observation, reward, done, info

    def get_frame_difference(self):
        current_frame = Helpers.preprocess_frame(self.observation)
        state = current_frame - self.previous_frame if self.previous_frame is not None else np.zeros(self.pixels_count)
        self.previous_frame = current_frame
        return state

    def update_episode_stats(self, reward):
        self.reward_sum += reward
        if reward == 1:
            self.points_scored += 1
        elif reward == -1:
            self.points_conceeded += 1 