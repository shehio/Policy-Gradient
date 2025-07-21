import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pg.agent import Agent  # type: ignore
from pg.hyperparameters import HyperParameters  # type: ignore
from pg.mlp_torch import MLP  # type: ignore
from pg.game import Game  # type: ignore

# Game params
GAME_NAME = "ALE/MsPacman-v5"
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
    save_interval=10_000
)

# Load network from file
load_network = False  # Start fresh for Pacman
load_episode_number = 0
network_file = os.path.join(os.path.dirname(__file__), '../..', 'models', 'torch_mlp_pacman.p')

def click_fire(game: Game):
    ## After the game is initialized, we need to click the fire button to start the game
    game.step(1)

if __name__ == '__main__':
    try:
        print("Initializing Pacman game...")
        game = Game(GAME_NAME, render, pixels_count, load_episode_number)
        print("Creating policy network...")
        policy_network = MLP(pixels_count, hidden_layers_count, output_count, network_file, GAME_NAME)
        if load_network:
            print(f"Loading network from episode {load_episode_number}...")
            policy_network.load_network(load_episode_number)

        print("Creating agent...")
        agent = Agent(policy_network, hyperparams)
        print("Clicking fire to start...")
        click_fire(game)
        print("Starting training loop...")

    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

    try:
        while True:
            state = game.get_frame_difference()

            action = agent.sample_and_record_action(state)
            observation, reward, done, info = game.step(action)
            agent.reap_reward(reward)
            game.update_episode_stats(reward)
            
            if done:
                agent.make_episode_end_updates(game.episode_number)
                episode_reward = game.reward_sum
                game.end_episode()
                print(f"Episode {game.episode_number}: reward={episode_reward:.1f}, running mean={game.running_reward:.3f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Final episode: {game.episode_number}")
        print(f"Final running reward: {game.running_reward:.3f}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise 