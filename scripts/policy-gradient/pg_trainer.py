#!/usr/bin/env python3
"""
Unified Policy Gradient Trainer
Handles training for Pong, Breakout, and Ms. Pacman with configurable parameters.
"""

import sys
import os
import numpy as np
import argparse
from typing import Dict, Any
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

from pg.hyperparameters import HyperParameters
from pg.game import Game

# Import game configs using absolute path
sys.path.append(os.path.dirname(__file__))
from game_configs import get_game_config, get_input_size, get_output_size, get_network_file, get_pacman_final_stats

class PolicyGradientTrainer:
    """Unified trainer for policy gradient algorithms."""
    
    def __init__(self, game_name: str, config: Dict[str, Any]):
        self.game_name = game_name
        self.config = config
        self.game_config = get_game_config(game_name)
        
        # Initialize game-specific variables
        self.previous_frame = None
        
    def setup_network(self, game: Game) -> Any:
        """Setup the appropriate network based on game type."""
        input_size = get_input_size(self.game_name)
        output_size = get_output_size(self.game_name, game.env)
        hidden_layers_count = self.config.get('hidden_layers_count', 200)
        network_file = self.config.get('network_file', get_network_file(self.game_name))
        
        if self.game_name == 'pacman':
            print(f"Creating CNN policy network with {output_size} actions...")
            print(f"Input: 7 channels (80x80x7 color features)")
        else:
            print(f"Creating MLP policy network...")
            
        return self.game_config.network_class(
            input_size, hidden_layers_count, output_size, 
            network_file, self.game_config.game_id
        )
    
    def setup_agent(self, policy_network: Any, hyperparams: HyperParameters) -> Any:
        """Setup the appropriate agent."""
        return self.game_config.agent_class(policy_network, hyperparams)
    
    def preprocess_state(self, game: Game) -> np.ndarray:
        """Preprocess the game state."""
        if self.game_config.preprocess_func:
            return self.game_config.preprocess_func(game, self.previous_frame)
        return game.get_frame_difference()
    
    def post_step(self, game: Game, info: Dict[str, Any]) -> None:
        """Handle post-step logic (e.g., Breakout fire button)."""
        if self.game_config.post_step_func:
            self.game_config.post_step_func(game, info)
    
    def post_episode(self, game: Game, agent: Any) -> None:
        """Handle post-episode logic."""
        if self.game_config.post_episode_func:
            self.game_config.post_episode_func(game, agent)
        
        # Reset frame for next episode (for games that need it)
        if self.game_name == 'pacman':
            self.previous_frame = None
    
    def train(self):
        """Main training loop."""
        try:
            print(f"Initializing {self.game_config.name} game...")
            
            input_size = get_input_size(self.game_name)
            game = Game(
                self.game_config.game_id, 
                self.config.get('render', False), 
                input_size, 
                self.config.get('load_episode_number', 0)
            )
            
            policy_network = self.setup_network(game)
            
            if self.config.get('load_network', False):
                load_episode = self.config.get('load_episode_number', 0)
                print(f"Loading network from episode {load_episode}...")
                policy_network.load_network(load_episode)
            
            hyperparams = HyperParameters(
                learning_rate=self.config.get('learning_rate', 1e-4),
                decay_rate=self.config.get('decay_rate', 0.99),
                gamma=self.config.get('gamma', 0.99),
                batch_size=self.config.get('batch_size', 10),
                save_interval=self.config.get('save_interval', 10000)
            )
            
            agent = self.setup_agent(policy_network, hyperparams)
            print("Starting training loop...")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise
        
        try:
            while True:
                state = self.preprocess_state(game)
                
                action = agent.sample_and_record_action(state)
                observation, reward, done, info = game.step(action)
                agent.reap_reward(reward)
                game.update_episode_stats(reward)
                
                self.post_step(game, info)
                
                if done:
                    agent.make_episode_end_updates(game.episode_number)
                    game.end_episode()
                    self.post_episode(game, agent)
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print(f"Final episode: {game.episode_number}")
            print(f"Final running reward: {game.running_reward:.3f}")
            
            # Game-specific final stats
            if self.game_name == 'pacman':
                final_stats = get_pacman_final_stats(agent)
                if final_stats:
                    print(final_stats)
                
        except Exception as e:
            print(f"Error during training: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Unified Policy Gradient Trainer")
    parser.add_argument("game", choices=['pong', 'breakout', 'pacman'], help="Game to train on")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--no-load-network", action="store_true", help="Don't load pre-trained network")
    parser.add_argument("--load-episode", type=int, help="Episode number to load from (defaults to latest)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--save-interval", type=int, default=10000, help="Save interval")
    parser.add_argument("--network-file", type=str, help="Network file path")
    
    args = parser.parse_args()
    
    # Set default episode numbers for each game
    default_episodes = {
        'pong': 70000,
        'breakout': 50000,
        'pacman': 0  # Start fresh for Pacman since it's newer
    }
    
    config = {
        'render': args.render,
        'load_network': not args.no_load_network,  # Default to loading
        'load_episode_number': args.load_episode if args.load_episode is not None else default_episodes[args.game],
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'save_interval': args.save_interval,
    }
    
    if args.network_file:
        config['network_file'] = args.network_file
    
    trainer = PolicyGradientTrainer(args.game, config)
    trainer.train()

if __name__ == "__main__":
    main() 