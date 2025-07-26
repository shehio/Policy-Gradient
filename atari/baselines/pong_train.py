from helpers import create_directory_if_not_exists, create_atari_environment, train, SaveEveryStepCallback
from stable_baselines3 import DQN

env_name = 'ALE/Pong-v5'
log_directory = 'logs'
model_directory = 'models/PPO'

env = create_atari_environment(env_name, render=False)
create_directory_if_not_exists(model_directory)
create_directory_if_not_exists(log_directory)

# Use 'CnnPolicy' for image-based observations like Atari Breakout
model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=log_directory)

save_callback = SaveEveryStepCallback(save_freq=100_000, save_path=model_directory, game_name='Pong')

timesteps = 100_000_000
epochs = 1000
model = train(model, timesteps, epochs, env_name, model_directory, callback=save_callback)
model.save(f'final_ppo_breakout_epochs_{epochs}_timesteps_{timesteps}')

