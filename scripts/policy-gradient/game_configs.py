"""
Game configurations for Policy Gradient training.
Contains all game-specific settings, agents, networks, and logic.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from pg.agent import Agent
from pg.mlp_torch import MLP
from pg.pacman.multi_action_agent import MultiActionAgent
from pg.pacman.cnn_torch_multiaction import CNNMultiAction
from pg.pacman.preprocess_pacman import preprocess_pacman_frame_color_aware_difference
from pg.game import Game

class GameConfig:
    """Configuration for different games."""
    
    def __init__(self, name: str, game_id: str, agent_class, network_class, 
                 preprocess_func=None, post_step_func=None, post_episode_func=None,
                 init_func=None, input_size=None, output_size=None, network_file=None):
        self.name = name
        self.game_id = game_id
        self.agent_class = agent_class
        self.network_class = network_class
        self.preprocess_func = preprocess_func
        self.post_step_func = post_step_func
        self.post_episode_func = post_episode_func
        self.init_func = init_func
        self.input_size = input_size
        self.output_size = output_size
        self.network_file = network_file

def handle_breakout_fire(game: Game, info: Dict[str, Any]) -> None:
    """Handle Breakout's fire button logic."""
    current_lives = info.get('lives', None)
    if hasattr(game, 'previous_lives') and current_lives is not None and current_lives < game.previous_lives:
        print(f"🔥 Life lost! Clicking fire. Lives: {game.previous_lives} -> {current_lives}")
        game.step(1)  # Click fire
    game.previous_lives = current_lives

def handle_breakout_episode_end(game: Game, agent: Any) -> None:
    """Handle Breakout episode end logic."""
    print("🔥 Episode ended, clicking fire for new episode...")
    game.previous_lives = 5
    game.step(1)  # Click fire to start new episode
    print("🔥 Fire clicked for new episode!")

def handle_breakout_init(game: Game) -> None:
    """Handle Breakout initialization - click fire to start the ball."""
    print("🔥 Clicking fire to start Breakout...")
    game.step(1)  # Click fire to start the game
    print("🔥 Fire clicked!")

def handle_pacman_episode_end(game: Game, agent: Any) -> None:
    """Handle Pacman episode end logic."""
    if game.episode_number % 10 == 0:
        recent_rewards = agent.total_rewards[-10:] if len(agent.total_rewards) >= 10 else agent.total_rewards
        recent_avg = np.mean(recent_rewards) if recent_rewards else 0
        print(f"Episode {game.episode_number}: reward={game.reward_sum:.1f}, running mean={game.running_reward:.3f}, recent avg={recent_avg:.3f}")

def get_pacman_final_stats(agent: Any) -> str:
    """Get Pacman-specific final statistics."""
    if hasattr(agent, 'total_rewards') and agent.total_rewards:
        recent_avg = np.mean(agent.total_rewards[-20:])
        return f"Recent average reward: {recent_avg:.3f}"
    return ""

# Game configurations
GAME_CONFIGS = {
    'pong': GameConfig(
        name="Pong",
        game_id="ALE/Pong-v5",
        agent_class=Agent,
        network_class=MLP,
        preprocess_func=lambda game, prev_frame: game.get_frame_difference(),
        input_size=80*80,
        output_size=1,
        network_file="torch_mlp"
    ),
    
    'breakout': GameConfig(
        name="Breakout", 
        game_id="ALE/Breakout-v5",
        agent_class=Agent,
        network_class=MLP,
        preprocess_func=lambda game, prev_frame: game.get_frame_difference(),
        post_step_func=handle_breakout_fire,
        post_episode_func=handle_breakout_episode_end,
        init_func=handle_breakout_init,
        input_size=80*80,
        output_size=1,
        network_file="torch_mlp"
    ),
    
    'pacman': GameConfig(
        name="Ms. Pacman",
        game_id="ALE/MsPacman-v5", 
        agent_class=MultiActionAgent,
        network_class=CNNMultiAction,
        preprocess_func=lambda game, prev_frame: preprocess_pacman_frame_color_aware_difference(game.observation, prev_frame),
        post_episode_func=handle_pacman_episode_end,
        input_size=80*80*7,
        output_size=None,  # Will be set dynamically from env.action_space.n
        network_file="torch_mlp_pacman"
    )
}

def get_game_config(game_name: str) -> GameConfig:
    """Get game configuration by name."""
    if game_name not in GAME_CONFIGS:
        raise ValueError(f"Unknown game: {game_name}. Available games: {list(GAME_CONFIGS.keys())}")
    return GAME_CONFIGS[game_name]

def get_input_size(game_name: str) -> int:
    """Get input size for a game."""
    config = get_game_config(game_name)
    return config.input_size

def get_output_size(game_name: str, game_env=None) -> int:
    """Get output size for a game."""
    config = get_game_config(game_name)
    if config.output_size is None and game_env is not None:
        return game_env.action_space.n
    return config.output_size or 1

def get_network_file(game_name: str) -> str:
    """Get default network file for a game by finding the most recent model."""
    import os
    import re
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        # Return base path if no models directory exists
        config = get_game_config(game_name)
        return config.network_file
    
    # Pattern mapping for each game
    pattern_map = {
        'pong': r'torch_mlp_ALE_Pong_v5.*_(\d+)$',
        'breakout': r'torch_mlp_ALE_Breakout_v5.*_(\d+)$',
        'pacman': r'torch_mlp_pacman_ALE_MsPacman_v5_cnn.*_(\d+)$',  # Prioritize CNN models
    }
    
    pattern = pattern_map.get(game_name)
    if not pattern:
        config = get_game_config(game_name)
        return config.network_file
    
    max_episode = 0
    latest_file = None
    
    for fname in os.listdir(model_dir):
        match = re.search(pattern, fname)
        if match:
            try:
                ep = int(match.group(1))
                if ep > max_episode:
                    max_episode = ep
                    latest_file = fname
            except Exception:
                continue
    
    if latest_file:
        # For MLP class, we need to return just the base name without the full path
        # The MLP class will construct the full filename itself
        config = get_game_config(game_name)
        return config.network_file
    else:
        # Return base path if no existing models found
        config = get_game_config(game_name)
        return config.network_file 