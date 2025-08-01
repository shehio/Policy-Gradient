import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("LunarLander-v3")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 5

env = gym.make("LunarLander-v3", render_mode="human")

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)
        # print(f"Episode {ep+1}, Reward: {rewards}")

    print(f"Episode {ep+1} finished")

env.close()
