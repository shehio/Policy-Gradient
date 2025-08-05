import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_game_with_recording(
    game_name, max_episodes=None, output_dir="recordings", **kwargs
):
    """Run a game using the unified pg_trainer.py with recording capabilities."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    log_file = output_path / f"{game_name}_training.log"
    video_dir = output_path / f"{game_name}_videos"
    video_dir.mkdir(exist_ok=True)

    print(f"Starting recording for: {game_name}")
    print(f"Output: {output_path}")
    print(f"Videos: {video_dir}")

    # Build command for pg_trainer.py
    cmd = [
        sys.executable,
        "atari/scripts/policy-gradient/pg_trainer.py",
        game_name,
        "--render",
    ]

    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if key == "learning_rate":
                cmd.extend(["--learning-rate", str(value)])
            elif key == "batch_size":
                cmd.extend(["--batch-size", str(value)])
            elif key == "save_interval":
                cmd.extend(["--save-interval", str(value)])
            elif key == "no_load_network" and value:
                cmd.append("--no-load-network")

    # Set environment variables for recording
    env = os.environ.copy()
    env["RECORD_VIDEOS"] = "1"
    env["VIDEO_DIR"] = str(video_dir)

    try:
        # Run the trainer
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
        )

        # Capture and display output
        with open(log_file, "w") as log_f:
            for line in process.stdout:
                print(line.rstrip())
                log_f.write(line)
                log_f.flush()

        process.wait()

        print(f"\nTraining completed!")
        print(f"Log saved to: {log_file}")
        print(f"Videos saved to: {video_dir}")

        return True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Policy Gradient training with recording"
    )
    parser.add_argument(
        "game", choices=["pong", "breakout", "pacman"], help="Game to train on"
    )
    parser.add_argument(
        "--max-episodes", type=int, help="Maximum episodes to run (not yet implemented)"
    )
    parser.add_argument("--output-dir", default="recordings", help="Output directory")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--save-interval", type=int, help="Save interval")
    parser.add_argument(
        "--no-load-network", action="store_true", help="Don't load pre-trained network"
    )

    args = parser.parse_args()

    # Convert args to kwargs
    kwargs = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "save_interval": args.save_interval,
        "no_load_network": args.no_load_network,
    }

    success = run_game_with_recording(
        args.game, max_episodes=args.max_episodes, output_dir=args.output_dir, **kwargs
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
