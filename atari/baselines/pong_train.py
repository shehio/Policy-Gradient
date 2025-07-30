from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4) # Stack 4 frames

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000_000)

model.save(f"pong_ppo__cnn_1M")