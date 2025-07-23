import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pg.pacman.multi_action_agent import MultiActionAgent
from pg.hyperparameters import HyperParameters
from pg.pacman.cnn_torch_multiaction import CNNMultiAction
from pg.game import Game
from pg.pacman.preprocess_pacman import preprocess_pacman_frame_color_aware_difference

# Game params
GAME_NAME = "ALE/MsPacman-v5"
render = False

# Hyperparameters
input_channels = 7
hidden_layers_count = 200
hyperparams = HyperParameters(
    learning_rate=1e-4,
    decay_rate=0.99,
    gamma=0.98,
    batch_size=5,
    save_interval=1000
)

# Load network from file
load_network = False
load_episode_number = 0
network_file = os.path.join(os.path.dirname(__file__), '../..', 'models', 'torch_mlp_pacman.p')

if __name__ == '__main__':
    try:
        print("Initializing Pacman game...")
        game = Game(GAME_NAME, render, 80*80*7, load_episode_number)
        output_count = game.env.action_space.n
        print(f"Creating CNN policy network with {output_count} actions...")
        print(f"Input: {input_channels} channels (80x80x7 color features)")
        policy_network = CNNMultiAction(input_channels, hidden_layers_count, output_count, network_file, GAME_NAME)
        if load_network:
            print(f"Loading network from episode {load_episode_number}...")
            policy_network.load_network(load_episode_number)

        print("Creating multi-action agent with better exploration...")
        agent = MultiActionAgent(policy_network, hyperparams)
        print("Starting training loop...")

    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

    try:
        previous_frame = None
        while True:
            state = preprocess_pacman_frame_color_aware_difference(game.observation, previous_frame)
            previous_frame = game.observation.copy()

            action = agent.sample_and_record_action(state)
            observation, reward, done, info = game.step(action)
            agent.reap_reward(reward)
            game.update_episode_stats(reward)
            
            if done:
                agent.make_episode_end_updates(game.episode_number)
                episode_reward = game.reward_sum
                game.end_episode()
                
                if game.episode_number % 10 == 0:
                    recent_rewards = agent.total_rewards[-10:] if len(agent.total_rewards) >= 10 else agent.total_rewards
                    recent_avg = np.mean(recent_rewards) if recent_rewards else 0
                    print(f"Episode {game.episode_number}: reward={episode_reward:.1f}, running mean={game.running_reward:.3f}, recent avg={recent_avg:.3f}")
                
                previous_frame = None

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Final episode: {game.episode_number}")
        print(f"Final running reward: {game.running_reward:.3f}")
        if agent.total_rewards:
            print(f"Recent average reward: {np.mean(agent.total_rewards[-20:]):.3f}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise 