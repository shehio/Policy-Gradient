#!/usr/bin/env python3
"""
Script to automatically record gameplay for a given game using the most recent model.
Usage: python record_gameplay.py <game_name>
Example: python record_gameplay.py mspacman
"""

import os
import sys
import glob
import re
import subprocess
import argparse
from pathlib import Path


def find_most_recent_model(game_name):
    """
    Find the most recent model for a given game.
    Returns the model path without .zip extension.
    """
    model_dirs = [f"../models/baselines/{game_name.lower()}"]

    all_models = []

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            pattern = os.path.join(model_dir, "*.zip")
            models = glob.glob(pattern)
            all_models.extend(models)

    if not all_models:
        raise FileNotFoundError(f"No models found for game '{game_name}'")

    model_timesteps = []
    for model_path in all_models:
        filename = os.path.basename(model_path)
        match = re.search(r"_(\d+)\.zip$", filename)
        if match:
            timesteps = int(match.group(1))
            model_timesteps.append((model_path, timesteps))

    if not model_timesteps:
        raise ValueError(
            f"Could not extract timesteps from model filenames for game '{game_name}'"
        )

    # Sort by timesteps (descending) and return the most recent
    model_timesteps.sort(key=lambda x: x[1], reverse=True)
    most_recent_model = model_timesteps[0][0]

    # Remove .zip extension
    model_path_without_zip = most_recent_model[:-4]

    print(f"Found most recent model: {model_path_without_zip}")
    print(f"Timesteps: {model_timesteps[0][1]:,}")

    return model_path_without_zip


def get_env_name(game_name):
    """
    Convert game name to environment name.
    """
    game_mapping = {
        "mspacman": "ALE/MsPacman-v5",
        "pacman": "ALE/MsPacman-v5",
        "pong": "ALE/Pong-v5",
        "breakout": "ALE/Breakout-v5",
        "spaceinvaders": "ALE/SpaceInvaders-v5",
        "asteroids": "ALE/Asteroids-v5",
        "berzerk": "ALE/Berzerk-v5",
        "bowling": "ALE/Bowling-v5",
        "boxing": "ALE/Boxing-v5",
        "centipede": "ALE/Centipede-v5",
        "chopper": "ALE/ChopperCommand-v5",
        "crazy": "ALE/CrazyClimber-v5",
        "defender": "ALE/Defender-v5",
        "demon": "ALE/DemonAttack-v5",
        "double": "ALE/DoubleDunk-v5",
        "enduro": "ALE/Enduro-v5",
        "fishing": "ALE/FishingDerby-v5",
        "freeway": "ALE/Freeway-v5",
        "frostbite": "ALE/Frostbite-v5",
        "gopher": "ALE/Gopher-v5",
        "gravitar": "ALE/Gravitar-v5",
        "hero": "ALE/Hero-v5",
        "ice": "ALE/IceHockey-v5",
        "jamesbond": "ALE/Jamesbond-v5",
        "kangaroo": "ALE/Kangaroo-v5",
        "krull": "ALE/Krull-v5",
        "kung": "ALE/KungFuMaster-v5",
        "montezuma": "ALE/MontezumaRevenge-v5",
        "name": "ALE/NameThisGame-v5",
        "phoenix": "ALE/Phoenix-v5",
        "pitfall": "ALE/Pitfall-v5",
        "pong": "ALE/Pong-v5",
        "private": "ALE/PrivateEye-v5",
        "qbert": "ALE/Qbert-v5",
        "riverraid": "ALE/Riverraid-v5",
        "roadrunner": "ALE/RoadRunner-v5",
        "robotank": "ALE/Robotank-v5",
        "seaquest": "ALE/Seaquest-v5",
        "skiing": "ALE/Skiing-v5",
        "solaris": "ALE/Solaris-v5",
        "star": "ALE/StarGunner-v5",
        "tennis": "ALE/Tennis-v5",
        "time": "ALE/TimePilot-v5",
        "tutankham": "ALE/Tutankham-v5",
        "upndown": "ALE/UpNDown-v5",
        "venture": "ALE/Venture-v5",
        "videopinball": "ALE/VideoPinball-v5",
        "wizard": "ALE/WizardOfWor-v5",
        "yars": "ALE/YarsRevenge-v5",
        "zaxxon": "ALE/Zaxxon-v5",
    }

    # Try exact match
    if game_name.lower() in game_mapping:
        return game_mapping[game_name.lower()]

    # Try partial match
    for key, value in game_mapping.items():
        if key in game_name.lower() or game_name.lower() in key:
            return value

    # Default to ALE format
    return f"ALE/{game_name.capitalize()}-v5"


def extract_timesteps_from_model(model_path):
    """Extract timesteps from model path for naming the output files."""
    filename = os.path.basename(model_path)
    match = re.search(r"_(\d+)", filename)
    if match:
        timesteps = int(match.group(1))
        # Format timesteps for display
        if timesteps >= 1000000:
            return f"{timesteps // 1000000}M"
        elif timesteps >= 1000:
            return f"{timesteps // 1000}k"
        else:
            return str(timesteps)
    return "unknown"


def run_recording(model_path, env_name, game_name):
    """
    Run the model with recording enabled.
    """
    # Create assets/videos directory if it doesn't exist
    videos_dir = Path("../../assets/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    timesteps = extract_timesteps_from_model(model_path)
    video_filename = f"{game_name}_dqn_cnn_{timesteps}_gameplay.mp4"

    video_path = videos_dir / video_filename

    print(f"Recording video to: {video_path}")

    # Run the test script with recording
    cmd = [
        "python",
        "atari_baseline_test.py",
        "--model",
        model_path,
        "--env",
        env_name,
        "--episodes",
        "1",
        "--record",
        "--video_path",
        str(video_path),
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running recording: {result.stderr}")
        return None

    print("Recording completed successfully!")
    return str(video_path)


def convert_to_gif(video_path, game_name):
    """
    Convert the recorded video to GIF using ffmpeg.
    """
    if not video_path or not os.path.exists(video_path):
        print("Video file not found, skipping GIF conversion")
        return None

    # Generate GIF filename
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.basename(video_path)
    gif_filename = video_basename.replace(".mp4", ".gif")
    gif_path = os.path.join(video_dir, gif_filename)

    print(f"Converting to GIF: {gif_path}")

    # ffmpeg command to convert to GIF
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=10,scale=320:-1:flags=lanczos",
        gif_path,
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error converting to GIF: {result.stderr}")
        return None

    print("GIF conversion completed successfully!")
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description="Record gameplay for a given game using the most recent model"
    )
    parser.add_argument(
        "game_name", type=str, help="Name of the game (e.g., mspacman, pong, breakout)"
    )

    args = parser.parse_args()
    game_name = args.game_name

    try:
        # Find the most recent model
        model_path = find_most_recent_model(game_name)

        # Get environment name
        env_name = get_env_name(game_name)
        print(f"Using environment: {env_name}")

        # Run recording
        video_path = run_recording(model_path, env_name, game_name)

        if video_path:
            # Convert to GIF
            gif_path = convert_to_gif(video_path, game_name)

            if gif_path:
                print("\nSuccessfully created gameplay recording!")
                print(f"Video: {video_path}")
                print(f"GIF: {gif_path}")
            else:
                print("Failed to convert video to GIF")
        else:
            print("Failed to record gameplay")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
