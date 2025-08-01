#!/usr/bin/env python3
"""
Game Model Manager for Policy Gradient Training
Helps manage models for different games with unique naming.
"""

import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from mlp_torch import MLP


def list_models_for_game(game_name):
    """List all models for a specific game."""
    models_dir = os.path.join(os.path.dirname(__file__), "policy-gradient", "models")
    base_name = "torch_mlp"
    sanitized_game = game_name.replace("/", "_").replace("-", "_")

    pattern = f"{base_name}_{sanitized_game}_*"
    matching_files = glob.glob(os.path.join(models_dir, pattern))

    print(f"\nModels for {game_name}:")
    if matching_files:
        episodes = []
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            try:
                # Extract episode number from filename
                parts = filename.split("_")
                episode_num = int(parts[-1])
                episodes.append((episode_num, file_path))
            except (ValueError, IndexError):
                continue

        episodes.sort()
        for episode_num, file_path in episodes:
            file_size = os.path.getsize(file_path) / 1e6  # Size in MB
            print(
                f"  Episode {episode_num}: {os.path.basename(file_path)} ({file_size:.1f} MB)"
            )
    else:
        print("  No models found for this game.")


def main():
    print("Game Model Manager")
    print("=" * 30)

    # Current architecture (matching scripts)
    pixels_count = 80 * 80
    hidden_layers_count = 200
    output_count = 1
    network_file = os.path.join(
        os.path.dirname(__file__), "policy-gradient", "models", "torch_mlp.p"
    )

    print(f"Architecture: {pixels_count}->{hidden_layers_count}->{output_count}")
    print(f"Network file base: {network_file}")

    # List models for different games
    games = ["ALE/Pong-v5", "ALE/Breakout-v5"]

    for game in games:
        list_models_for_game(game)

    print("\nUsage:")
    print("1. To continue training Pong from latest model:")
    print("   Update load_episode_number in pgpong.py")
    print("2. To continue training Breakout from latest model:")
    print("   Update load_episode_number in pgbreakout.py")
    print("3. To start fresh:")
    print("   Set load_network = False in the script")


if __name__ == "__main__":
    main()
