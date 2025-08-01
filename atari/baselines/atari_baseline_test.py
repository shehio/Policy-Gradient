from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
import time
import gymnasium as gym
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test trained RL agents on Atari games with rendering')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to the trained model file (without .zip extension)')
    parser.add_argument('--env', '-e', type=str, default='ALE/Pong-v5',
                       help='Environment to test on (default: ALE/Pong-v5)')
    parser.add_argument('--episodes', '-n', type=int, default=3,
                       help='Number of episodes to run (default: 3)')
    parser.add_argument('--algorithm', '-a', type=str, choices=['auto', 'ppo', 'dqn', 'a2c'], 
                       default='auto', help='Algorithm type (auto detects from model)')
    parser.add_argument('--delay', '-d', type=float, default=0.01,
                       help='Delay between frames in seconds (default: 0.01)')
    
    return parser.parse_args()

def detect_algorithm_from_model(model_path):
    """Try to detect the algorithm from the model filename"""
    if 'ppo' in model_path.lower():
        return PPO
    elif 'dqn' in model_path.lower():
        return DQN
    elif 'a2c' in model_path.lower():
        return A2C
    else:
        # Default to A2C if we can't detect
        return A2C

def render_model(model_path, env_name, algorithm_class, num_episodes=3, delay=0.01):
    """
    Render a trained model playing the game
    """
    # Create environment with rendering
    env = make_atari_env(env_name, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4) # Stack 4 frames

    # Load the model
    try:
        model = algorithm_class.load(model_path, env=env)
        print(f"Loaded {algorithm_class.__name__} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load without environment...")
        model = algorithm_class.load(model_path)
        print(f"Loaded {algorithm_class.__name__} model from {model_path}")
    
    # Run episodes
    total_rewards = []
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            step_count += 1
            
            # Add delay to make it watchable
            time.sleep(delay)
            
            if dones[0]:
                print(f"Episode {episode + 1} finished with reward: {total_reward}, steps: {step_count}")
                total_rewards.append(total_reward)
                break
    
    env.close()
    
    # Print summary
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\nSummary: {num_episodes} episodes completed")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Best episode: {max(total_rewards)}")
        print(f"Worst episode: {min(total_rewards)}")

def main():
    args = parse_arguments()
    
    # Check if model file exists in the new directory structure
    model_file = f"{args.model}.zip"
    model_paths = [
        f"../models/baselines/{model_file}",  # New organized structure
        model_file,  # Current directory (fallback)
    ]
    
    model_found = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            args.model = model_path.replace('.zip', '')
            model_found = True
            break
    
    if not model_found:
        print(f"Error: Model file {model_file} not found!")
        print("Available models in ../models/baselines/:")
        baselines_dir = "../models/baselines"
        if os.path.exists(baselines_dir):
            for file in os.listdir(baselines_dir):
                if file.endswith(".zip"):
                    print(f"  - {file.replace('.zip', '')}")
        else:
            print("  No models directory found")
        return
    
    # Determine algorithm
    if args.algorithm == 'auto':
        algorithm_class = detect_algorithm_from_model(args.model)
    else:
        algorithm_map = {
            'ppo': PPO,
            'dqn': DQN,
            'a2c': A2C
        }
        algorithm_class = algorithm_map[args.algorithm]
    
    print(f"Testing {algorithm_class.__name__} model on {args.env}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Frame delay: {args.delay}s")
    print("-" * 50)
    
    render_model(args.model, args.env, algorithm_class, args.episodes, args.delay)

if __name__ == "__main__":
    main() 