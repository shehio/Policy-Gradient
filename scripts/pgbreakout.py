import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent
from hyperparameters import HyperParameters
from mlp_torch import MLP
from game import Game

# Game params
GAME_NAME = "ALE/Breakout-v5"
render = False

# Hyperparameters
pixels_count = 80 * 80  # input dimensionality: 80x80 grid
hidden_layers_count = 200  # number of hidden layer neurons
output_count = 1
hyperparams = HyperParameters(
    learning_rate=1e-4,
    decay_rate=0.99,
    gamma=0.99,
    batch_size=10,
    save_interval=1000
)

# Load network from file
load_network = True
load_episode_number = 0
network_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch_mlp.p')

def click_fire(game: Game):
    ## After the game is initialized, we need to click the fire button to start the game
    game.step(1)

if __name__ == '__main__':
    try:
        print("Initializing game...")
        game = Game(GAME_NAME, render, pixels_count, load_episode_number)
        print("Creating policy network...")
        policy_network = MLP(pixels_count, hidden_layers_count, output_count, network_file, GAME_NAME)
        if load_network:
            print(f"Loading network from episode {load_episode_number}...")
            policy_network.load_network(load_episode_number)

        print("Creating agent...")
        agent = Agent(policy_network, hyperparams)
        print("Clicking fire...")
        click_fire(game)
        print("Starting training loop...")

    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

    try:
        # Track previous lives to detect life loss
        previous_lives = 5
        
        while True:
            state = game.get_frame_difference()

            action = agent.sample_and_record_action(state)
            observation, reward, done, info = game.step(action)
            agent.reap_reward(reward)
            game.update_episode_stats(reward)
            
            # Check if lives decreased (indicating life loss) and click the fire button
            current_lives = info.get('lives', None)
            if current_lives is not None and current_lives < previous_lives:
                click_fire(game)
                previous_lives = current_lives
            
            if done:
                # print(f"Episode {game.episode_number}: reward={game.reward_sum:.1f}, running mean={game.running_reward:.3f}")
                
                agent.make_episode_end_updates(game.episode_number)
                game.end_episode()
                game.reset()

                previous_lives = 5
                click_fire(game)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Final episode: {game.episode_number}")
        print(f"Final running reward: {game.running_reward:.3f}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
