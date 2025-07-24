import gymnasium as gym
import ale_py  # This registers the ALE environments
import time
import numpy as np
from typing import Optional
from collections import deque

class Game:
    def __init__(self, game_name: str, render: bool, pixels_count: int, episode_number: int) -> None:
        self.game_name = game_name
        self.render_flag = render
        self.pixels_count = pixels_count
        self.env = self.get_environment()
        self.starting_episode_number = episode_number
        self.episode_number: int = episode_number
        self.recent_rewards = deque(maxlen=100)
        self.running_reward: float = 0.0
        self.reset()

    def get_environment(self) -> gym.Env:
        if self.render_flag:
            env = gym.make(self.game_name, render_mode="human")
        else:
            env = gym.make(self.game_name)
        
        # Configure Atari environment to prevent frame skipping
        # This ensures we capture every frame and don't miss any game state
        try:
            atari_env = env.unwrapped
            if hasattr(atari_env, 'frameskip'):
                setattr(atari_env, 'frameskip', 1)  # Set frame skip to 1 (no skipping)

            # Access ALE directly if available
            if hasattr(atari_env, 'ale'):
                ale = getattr(atari_env, 'ale')
                ale.setInt('frame_skip', 1)  # Ensure ALE frame skip is also 1
                # Disable action repeat to ensure every action is processed
                ale.setFloat('repeat_action_probability', 0.0)
        except Exception as e:
            print(f"Warning: Could not configure Atari environment settings: {e}")
        
        return env

    def reset(self) -> None:
        self.observation, _ = self.env.reset()
        self.previous_frame: Optional[np.ndarray] = None
        self.reward_sum: float = 0.0
        self.points_scored: int = 0
        self.points_conceeded: int = 0

    def end_episode(self) -> None:
        self.__update_running_reward()
        if self.episode_number % 10 == 0:
            print('Resetting env. Episode: %i, episode reward: %i, running mean(last 100 episodes): %f.' % (self.episode_number, self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self.observation, float(reward), done, info

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

    def __update_running_reward(self) -> None:
        # Add current episode reward to the rolling window
        self.recent_rewards.append(self.reward_sum)

        # Calculate average of last 100 episodes (or fewer if we haven't reached 100 yet)
        self.running_reward = float(np.mean(self.recent_rewards))
