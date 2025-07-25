from stable_baselines3 import PPO
from helpers import create_atari_environment, render_model

env_name = 'ALE/Breakout-v5'
env = create_atari_environment(name=env_name, render=True)
model = PPO.load('final_ppo_breakout')
render_model(env, model, 10 * 1000)
