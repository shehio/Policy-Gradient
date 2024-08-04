from helpers import create_directory_if_not_exists, create_environment, train
from stable_baselines3 import PPO

env_name = 'LunarLander-v2'
log_directory = 'logs'
model_directory = 'models/PPO'

env = create_environment(env_name)
create_directory_if_not_exists(model_directory)
create_directory_if_not_exists(log_directory)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_directory)

timesteps = 10000
epochs = 100
model = train(model, timesteps, epochs, env_name,  model_directory)
model.save('final_ppo_lunar')
