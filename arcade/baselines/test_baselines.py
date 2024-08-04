import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def create_environment(name: str):
    env = gym.make(name, render_mode='human')
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env

def render_model(env, model, timesteps):
    obs = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if i % 100 == 0:
            env.render()
    env.close()


env_name = 'LunarLander-v2'
env = create_environment(env_name)
model = PPO.load('final_ppo_lunar')
render_model(env, model, 10 * 1000)

