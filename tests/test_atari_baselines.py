"""
Tests for Atari baselines functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the atari directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atari", "baselines"))

try:
    from atari_baseline_train import (
        parse_arguments,
        get_env_name,
        ensure_model_dir,
        find_latest_model,
    )
    from atari_baseline_test import (
        parse_arguments as test_parse_arguments,
        detect_algorithm_from_model,
    )
except ImportError as e:
    pytest.skip(f"Could not import atari baselines: {e}", allow_module_level=True)


def test_test_parse_arguments():
    """Test that argument parsing works with required arguments."""
    with patch("sys.argv", ["atari_baseline_test.py", "--model", "test_model"]):
        args = test_parse_arguments()
        assert args.model == "test_model"
        assert args.env == "ALE/Pong-v5"
        assert args.episodes == 3
        assert args.algorithm == "auto"
        assert args.delay == 0.01


class TestAtariBaselineTrain:
    """Test the atari_baseline_train module."""

    def test_parse_arguments_defaults(self):
        """Test argument parsing with default values."""
        with patch("sys.argv", ["atari_baseline_train.py"]):
            args = parse_arguments()
            assert args.algorithm == "ppo"
            assert args.timesteps == "100000"
            assert args.env == "ALE/Pong-v5"
            assert args.n_envs == 1
            assert args.seed == 0

    def test_parse_arguments_custom(self):
        """Test argument parsing with custom values."""
        with patch(
            "sys.argv",
            [
                "atari_baseline_train.py",
                "--algorithm",
                "dqn",
                "--timesteps",
                "500000",
                "--env",
                "ALE/Breakout-v5",
                "--n_envs",
                "4",
                "--seed",
                "42",
            ],
        ):
            args = parse_arguments()
            assert args.algorithm == "dqn"
            assert args.timesteps == "500000"
            assert args.env == "ALE/Breakout-v5"
            assert args.n_envs == 4
            assert args.seed == 42

    def test_get_env_name(self):
        """Test environment name extraction."""
        assert get_env_name("ALE/Pong-v5") == "pong"
        assert get_env_name("ALE/Breakout-v5") == "breakout"
        assert get_env_name("ALE/MsPacman-v5") == "mspacman"
        assert get_env_name("ALE/SpaceInvaders-v4") == "spaceinvaders"

    @patch("os.makedirs")
    def test_ensure_model_dir(self, mock_makedirs):
        """Test model directory creation."""
        ensure_model_dir("test_models")
        mock_makedirs.assert_called_once_with("test_models", exist_ok=True)

    @patch("glob.glob")
    def test_find_latest_model(self, mock_glob):
        """Test finding the latest model."""
        # Mock glob to return some model files
        mock_glob.return_value = [
            "test_models/model_100000.zip",
            "test_models/model_500000.zip",
            "test_models/model_1000000.zip",
        ]

        latest_model, timesteps = find_latest_model("test_models/model")
        assert latest_model == "test_models/model_1000000"
        assert timesteps == 1000000

    @patch("glob.glob")
    def test_find_latest_model_no_models(self, mock_glob):
        """Test finding latest model when none exist."""
        mock_glob.return_value = []

        latest_model, timesteps = find_latest_model("test_models/model")
        assert latest_model is None
        assert timesteps == 0


class TestAtariBaselineTest:
    """Test the atari_baseline_test module."""

    def test_test_parse_arguments_defaults(self):
        """Test argument parsing with default values."""
        with patch("sys.argv", ["atari_baseline_test.py", "--model", "test_model"]):
            args = test_parse_arguments()
            assert args.model == "test_model"
            assert args.env == "ALE/Pong-v5"
            assert args.episodes == 3
            assert args.algorithm == "auto"
            assert args.delay == 0.01

    def test_test_parse_arguments_custom(self):
        """Test argument parsing with custom values."""
        with patch(
            "sys.argv",
            [
                "atari_baseline_test.py",
                "--model",
                "test_model",
                "--env",
                "ALE/Breakout-v5",
                "--episodes",
                "5",
                "--algorithm",
                "ppo",
                "--delay",
                "0.05",
            ],
        ):
            args = test_parse_arguments()
            assert args.model == "test_model"
            assert args.env == "ALE/Breakout-v5"
            assert args.episodes == 5
            assert args.algorithm == "ppo"
            assert args.delay == 0.05

    def test_detect_algorithm_from_model(self):
        """Test algorithm detection from model filename."""
        # Mock the algorithm classes
        mock_ppo = MagicMock()
        mock_dqn = MagicMock()
        mock_a2c = MagicMock()

        with patch("atari_baseline_test.PPO", mock_ppo), patch(
            "atari_baseline_test.DQN", mock_dqn
        ), patch("atari_baseline_test.A2C", mock_a2c):
            assert detect_algorithm_from_model("pong_ppo_cnn_100000") == mock_ppo
            assert detect_algorithm_from_model("breakout_dqn_cnn_500000") == mock_dqn
            assert detect_algorithm_from_model("pacman_a2c_cnn_200000") == mock_a2c
            # Default to A2C if no algorithm detected
            assert detect_algorithm_from_model("unknown_model") == mock_a2c
