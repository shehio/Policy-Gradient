import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

env = gym.make('LunarLander-v2')
env = Monitor(env)
env = DummyVecEnv([lambda: env])

models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

timesteps = 10000
for i in range(10):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{timesteps * i}")

model.save("final_ppo_lunar")
model = PPO.load("final_ppo_lunar")

obs = env.reset()

for i in range(3000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if i % 30 == 0:
        env.render()
env.close()
