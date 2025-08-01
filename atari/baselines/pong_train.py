from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py  # This registers the ALE environments
import os

env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4) # Stack 4 frames

# Let user choose the algorithm
print("Choose your algorithm:")
print("1. PPO (Proximal Policy Optimization)")
print("2. DQN (Deep Q-Network)")
print("3. A2C (Advantage Actor-Critic)")
print("4. A2C (with CNN policy)")

choice = input("Enter your choice (1-4): ").strip()

# Map choice to algorithm and model path
if choice == "1":
    algorithm = PPO
    model_path = "pong_ppo_cnn_1M"
    policy = "CnnPolicy"
elif choice == "2":
    algorithm = DQN
    model_path = "pong_dqn_cnn_1M"
    policy = "CnnPolicy"
elif choice == "3":
    algorithm = A2C
    model_path = "pong_a2c_mlp_1M"
    policy = "MlpPolicy"
elif choice == "4":
    algorithm = A2C
    model_path = "pong_a2c_cnn_1M"
    policy = "CnnPolicy"
else:
    print("Invalid choice. Defaulting to PPO...")
    algorithm = PPO
    model_path = "pong_ppo_cnn_1M"
    policy = "CnnPolicy"

print(f"Selected: {algorithm.__name__} with {policy}")

# Try to load existing model, create new one if it doesn't exist
if os.path.exists(f"{model_path}.zip"):
    print(f"Loading existing model from {model_path}")
    model = algorithm.load(model_path, env=env, verbose=1)
else:
    print(f"Model {model_path} not found. Creating new {algorithm.__name__} model...")
    model = algorithm(policy, env, verbose=1)

# Continue training
print("Starting training...")
model.learn(total_timesteps=5000_000)

# Save with algorithm name in filename
save_path = f"pong_{algorithm.__name__.lower()}_cnn_5M"
model.save(save_path)
print(f"Training completed and model saved as {save_path}!")