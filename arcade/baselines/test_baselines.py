from stable_baselines3 import PPO
from helpers import create_environment, render_model

env_name = 'LunarLander-v2'
env = create_environment(name=env_name, render=True)
model = PPO.load('final_ppo_lunar')
render_model(env, model, 10 * 1000)
