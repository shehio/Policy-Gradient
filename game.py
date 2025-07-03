import gym
import time
import numpy as np
from typing import Optional

class Game:
    def __init__(self, game_name: str, render: bool, sleep_for_rendering_in_seconds: float, pixels_count: int, episode_number: int) -> None:
        self.game_name = game_name
        self.render_flag = render
        self.sleep_for_rendering_in_seconds = sleep_for_rendering_in_seconds
        self.pixels_count = pixels_count
        self.env = self.get_environment()
        self.observation, _ = self.env.reset()
        self.previous_frame: Optional[np.ndarray] = None
        self.running_reward: float = 0.0
        self.reward_sum: float = 0.0
        self.episode_number: int = 0
        self.points_scored: int = 0
        self.points_conceeded: int = 0

    def get_environment(self) -> gym.Env:
        if self.render_flag:
            return gym.make(self.game_name, render_mode="human")
        else:
            return gym.make(self.game_name)

    def render(self) -> None:
        if self.render_flag:
            self.env.render()
            time.sleep(self.sleep_for_rendering_in_seconds)

    def end_episode(self) -> None:
        self.running_reward = self.running_reward * (self.episode_number - 1) / self.episode_number + self.reward_sum / self.episode_number if self.episode_number > 0 else self.reward_sum
        print('Resetting env. Episode: %i, episode reward: %i, running mean: %f.' % (self.episode_number, self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()

    def reset(self) -> np.ndarray:
        self.observation, _ = self.env.reset()
        self.previous_frame = None
        self.reward_sum = 0.0
        self.points_scored = 0
        self.points_conceeded = 0
        return self.observation

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self.observation, reward, done, info

    def preprocess_frame(self, image_frame: np.ndarray) -> np.ndarray:
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        image_frame = image_frame[35:195]  # crop
        image_frame = image_frame[::2, ::2, 0]  # downsample by factor of 2
        image_frame[image_frame == 144] = 0  # erase background (background type 1)
        image_frame[image_frame == 109] = 0  # erase background (background type 2)
        image_frame[image_frame != 0] = 1  # everything else (paddles, ball) just set to 1
        return image_frame.astype(float).ravel()

    def get_frame_difference(self) -> np.ndarray:
        current_frame = self.preprocess_frame(self.observation)
        state = current_frame - self.previous_frame if self.previous_frame is not None else np.zeros(self.pixels_count)
        self.previous_frame = current_frame
        return state

    def update_episode_stats(self, reward: float) -> None:
        self.reward_sum += reward
        if reward == 1:
            self.points_scored += 1
        elif reward == -1:
            self.points_conceeded += 1

def discount_and_normalize_rewards(r: np.ndarray, gamma: float) -> np.ndarray:
    discounted_rewards = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards 