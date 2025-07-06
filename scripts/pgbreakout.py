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
        while True:
            state = game.get_frame_difference()

            # If there's no difference in the frame, click the fire button
            if ((np.sum(state)) == 0):
                click_fire(game)
            
            action = agent.sample_and_record_action(state)
            observation, reward, done, info = game.step(action)
            agent.reap_reward(reward)
            game.update_episode_stats(reward)

            # if reward == 1:
            #     print('ep %i: point scored, score: %i' % (game.episode_number, game.points_scored))
            
            if done:
                agent.make_episode_end_updates(game.episode_number)
                game.end_episode()
                game.reset()
                print("Resetting game...")
                click_fire(game)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
