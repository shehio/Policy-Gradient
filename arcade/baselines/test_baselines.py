import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = gym.make('LunarLander-v2', render_mode='human')
env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = PPO.load("final_ppo_lunar")
obs = env.reset()

for i in range(3000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if i % 30 == 0:
        env.render()
env.close()
