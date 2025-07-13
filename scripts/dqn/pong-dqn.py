import gymnasium as gym
import ale_py  # This registers the ALE environments
import os
import numpy as np
import torch
import sys
import time
import json
from collections import deque

# Get the project root directory (2 levels up from this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
from dqn.config.environment_config import EnvironmentConfig # type: ignore
from dqn.config.exploration_config import ExplorationConfig # type: ignore
from dqn.config.image_config import ImageConfig # type: ignore
from dqn.config.learning_config import LearningConfig # type: ignore
from dqn.config.model_config import ModelConfig # type: ignore
from dqn.config.training_config import TrainingConfig # type: ignore

from dqn.agent import Agent # type: ignore
from dqn.config.hyperparameters import HyperParameters # type: ignore


def main():

    # Create hyperparameters using the HyperParameters class
    hyperparams = HyperParameters(
        environment_config=EnvironmentConfig(
            environment="ALE/Pong-v5",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            render_game_window=True
        ),
        model_config=ModelConfig(
            save_models=True,
            model_path="./models/pong-cnn-",
            save_model_interval=10,
            train_model=True,
            load_model_from_file=True,
            load_file_episode=850
        ),
        training_config=TrainingConfig(
            batch_size=64,
            max_episode=100000,
            max_step=100000,
            max_memory_len=50000,
            min_memory_len=40000
        ),
        learning_config=LearningConfig(
            gamma=0.97,
            alpha=0.00025
        ),
        exploration_config=ExplorationConfig(
            epsilon_start=1.0,
            epsilon_decay=0.99,
            epsilon_minimum=0.05
        ),
        image_config=ImageConfig(
            target_h=80,
            target_w=64,
            crop_top=20
        )
    )
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(hyperparams.model.model_path), exist_ok=True)
    
    # Create environment
    if hyperparams.environment.render_game_window:
        env = gym.make(hyperparams.environment.environment, render_mode="human")
    else:
        env = gym.make(hyperparams.environment.environment, render_mode="rgb_array")
    
    # Create agent
    agent = Agent(env, hyperparams)
    
    # Load model if specified
    if hyperparams.model.load_model_from_file:
        model_path = f"{hyperparams.model.model_path}{hyperparams.model.load_file_episode}.pkl"
        epsilon_path = f"{hyperparams.model.model_path}{hyperparams.model.load_file_episode}.json"
        
        if os.path.exists(model_path):
            agent.online_model.load_state_dict(torch.load(model_path, map_location=hyperparams.environment.device))
            print(f"Model loaded from {model_path}")
            
            # Load epsilon value if JSON file exists
            if os.path.exists(epsilon_path):
                with open(epsilon_path, 'r') as outfile:
                    param = json.load(outfile)
                    agent.epsilon = param.get('epsilon', agent.epsilon)
                print(f"Epsilon loaded: {agent.epsilon}")
            else:
                print(f"Epsilon file {epsilon_path} not found, using default epsilon")
        else:
            print(f"Model file {model_path} not found!")
    
    # Determine starting episode
    if hyperparams.model.load_model_from_file:
        start_episode = hyperparams.model.load_file_episode + 1
    else:
        start_episode = 0
    
    # Training loop
    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
    total_step = 1  # Cumulative sum of all steps in episodes
    
    for episode in range(start_episode, hyperparams.training.max_episode):
        start_time = time.time()  # Keep time
        state, _ = env.reset()
        state = agent.preProcess(state)
        
        # Stack state: Every state contains 4 time continuous frames
        # We stack frames like 4 channel image
        state = np.stack((state, state, state, state))
        
        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode
        
        for step in range(hyperparams.training.max_step):
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = agent.preProcess(next_state)
            
            # Stack next state: Every state contains 4 time continuous frames
            # We stack frames like 4 channel image
            next_state = np.stack((next_state, state[0], state[1], state[2]))
            
            # Store experience
            agent.storeResults(state, action, reward, next_state, done)
            
            # Train if enough memory
            if hyperparams.model.train_model:
                loss, max_q = agent.train()
            else:
                loss, max_q = [0, 0]
            
            total_loss += loss
            total_max_q_val += max_q
            state = next_state
            total_reward += float(reward)
            total_step += 1
            
            # Should this be based on episode or step?
            if total_step % 1000 == 0:
                agent.adaptiveEpsilon()
            
            if done or truncated:
                break
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.target_model.load_state_dict(agent.online_model.state_dict())
        
        # Episode completed - detailed logging like original pong-dqn.py
        current_time = time.time()  # Keep current time
        time_passed = current_time - start_time  # Find episode duration
        current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
        epsilon_dict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file

        if hyperparams.model.save_models and episode % hyperparams.model.save_model_interval == 0:  # Save model as file
            weights_path = f"{hyperparams.model.model_path}{episode}.pkl"
            epsilon_path = f"{hyperparams.model.model_path}{episode}.json"

            torch.save(agent.online_model.state_dict(), weights_path)
            with open(epsilon_path, 'w') as outfile:
                json.dump(epsilon_dict, outfile)

        if hyperparams.model.train_model:
            agent.target_model.load_state_dict(agent.online_model.state_dict())  # Update target model

        last_100_ep_reward.append(total_reward)
        avg_max_q_val = total_max_q_val / step if step > 0 else 0

        out_str = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
            episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, time_passed, step, total_step
        )

        print(out_str)

        if hyperparams.model.save_models:
            output_path = f"{hyperparams.model.model_path}out.txt"  # Save outStr to file
            with open(output_path, 'a') as outfile:
                outfile.write(out_str + "\n")
    
    env.close()

if __name__ == "__main__":
    main() 