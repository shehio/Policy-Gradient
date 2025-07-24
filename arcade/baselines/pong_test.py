from stable_baselines3 import DQN
from helpers import create_atari_environment, render_model

env_name = 'Pong-v0'
env = create_atari_environment(name=env_name, render=True)
model = DQN.load('pong_ppo.zip')
render_model(env, model, 10 * 1000)
