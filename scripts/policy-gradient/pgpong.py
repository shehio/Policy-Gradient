import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from agent import Agent
from hyperparameters import HyperParameters
from mlp_torch import MLP
from game import Game

# Game params
GAME_NAME = "ALE/Pong-v5"
# Disable rendering on headless servers (like EC2)
render = False
sleep_for_rendering_in_seconds = 0.001

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
load_network = True
load_episode_number = 10_000
network_file = os.path.join(os.path.dirname(__file__), '../..', 'models', 'torch_mlp.p')

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

            # if reward == 1:
            #     print('ep %d: point scored, score: %i: %i' % (game.episode_number, game.points_conceeded, game.points_scored))
            # elif reward == -1:
            #     print('ep %d: point conceeded, score: %i: %i' % (game.episode_number, game.points_conceeded, game.points_scored))

            if done:
                agent.make_episode_end_updates(game.episode_number)
                game.end_episode()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
