from stable_baselines3 import PPO
from helpers import create_environment, render_model

env_name = 'Breakout-v4'
env = create_environment(name=env_name, render=True)
model = PPO.load('final_ppo_breakout.zip')
render_model(env, model, 10 * 1000)
