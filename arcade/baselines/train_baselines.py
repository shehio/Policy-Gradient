import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

def create_environment(name: str):
    env = gym.make(name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def create_directory_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(model, timesteps, epochs, tensorboard_log_name, model_directory):
    for i in range(epochs):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=tensorboard_log_name)
        model.save(f"{model_directory}/{timesteps * i}")

    return model


env_name = 'LunarLander-v2'
log_directory = 'logs'
model_directory = 'models/PPO'

env = create_environment(env_name)
create_directory_if_not_exists(model_directory)
create_directory_if_not_exists(log_directory)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_directory)

timesteps = 10000
epochs = 10
model = train(model, timesteps, epochs, env_name,  model_directory)
model.save('final_ppo_lunar')
