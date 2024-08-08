from helpers import create_directory_if_not_exists, create_atari_environment, train
from stable_baselines3 import DQN

env_name = 'Pong-v0'
log_directory = 'logs'
model_directory = 'models/PPO'

env = create_atari_environment(env_name, render=False)
create_directory_if_not_exists(model_directory)
create_directory_if_not_exists(log_directory)

# Use 'CnnPolicy' for image-based observations like Atari Breakout
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_directory)

timesteps = 10000
epochs = 10
model = train(model, timesteps, epochs, env_name, model_directory)
model.save('pong_ppo')

