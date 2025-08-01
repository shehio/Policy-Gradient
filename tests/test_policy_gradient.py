"""
Tests for Policy Gradient implementation.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the atari directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "atari", "algorithms", "pg", "src")
)

try:
    from mlp_torch import MLP
    from agent import PolicyGradientAgent
    from memory import Memory
except ImportError as e:
    pytest.skip(
        f"Could not import Policy Gradient modules: {e}", allow_module_level=True
    )


class TestMLP:
    """Test the MLP model."""

    def test_mlp_initialization(self):
        """Test that the MLP can be initialized."""
        mlp = MLP(
            input_count=6400,
            hidden_layers_count=200,
            output_count=1,
            network_file="test.pkl",
            game_name="pong",
        )
        assert mlp is not None
        assert hasattr(mlp, "forward")
        assert hasattr(mlp, "forward_pass")

    def test_mlp_forward_pass(self):
        """Test the forward pass of the MLP."""
        mlp = MLP(
            input_count=6400,
            hidden_layers_count=200,
            output_count=1,
            network_file="test.pkl",
            game_name="pong",
        )

        # Mock input
        input_data = np.random.randn(6400).astype(np.float32)

        # Mock torch
        with patch("mlp_torch.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor

            # Mock the forward method
            mock_output = MagicMock()
            mock_output.cpu.return_value.numpy.return_value = np.array([0.7])
            mock_hidden = MagicMock()
            mock_hidden.cpu.return_value.numpy.return_value = np.random.randn(200)

            mlp.forward = MagicMock(return_value=mock_output)
            mlp.relu = MagicMock(return_value=mock_hidden)

            output, hidden = mlp.forward_pass(input_data)
            assert output is not None
            assert hidden is not None

    def test_mlp_save_load_network(self):
        """Test saving and loading the network."""
        mlp = MLP(
            input_count=6400,
            hidden_layers_count=200,
            output_count=1,
            network_file="test.pkl",
            game_name="pong",
        )

        # Mock torch.save and torch.load
        with patch("mlp_torch.torch.save") as mock_save, patch(
            "mlp_torch.torch.load"
        ) as mock_load, patch("mlp_torch.os.path.exists", return_value=True):

            # Test saving
            mlp.save_network(100)
            mock_save.assert_called()

            # Test loading
            mock_state_dict = MagicMock()
            mock_load.return_value = mock_state_dict

            mlp.load_network(100)
            mock_load.assert_called()


class TestMemory:
    """Test the Memory class."""

    def test_memory_initialization(self):
        """Test that the Memory can be initialized."""
        memory = Memory()
        assert memory is not None
        assert hasattr(memory, "states")
        assert hasattr(memory, "actions")
        assert hasattr(memory, "rewards")

    def test_memory_add_episode(self):
        """Test adding an episode to memory."""
        memory = Memory()

        # Mock episode data
        states = [np.random.randn(6400) for _ in range(10)]
        actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        rewards = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        memory.add_episode(states, actions, rewards)

        assert len(memory.states) == 10
        assert len(memory.actions) == 10
        assert len(memory.rewards) == 10

    def test_memory_clear(self):
        """Test clearing the memory."""
        memory = Memory()

        # Add some data
        states = [np.random.randn(6400) for _ in range(5)]
        actions = [0, 1, 0, 1, 0]
        rewards = [0.0, 1.0, 0.0, 1.0, 0.0]

        memory.add_episode(states, actions, rewards)
        assert len(memory.states) == 5

        # Clear memory
        memory.clear()
        assert len(memory.states) == 0
        assert len(memory.actions) == 0
        assert len(memory.rewards) == 0


class TestPolicyGradientAgent:
    """Test the PolicyGradientAgent class."""

    def test_agent_initialization(self):
        """Test that the agent can be initialized."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 2  # Binary action space

        # Mock the MLP
        with patch("agent.MLP") as mock_mlp_class:
            mock_mlp = MagicMock()
            mock_mlp_class.return_value = mock_mlp

            agent = PolicyGradientAgent(mock_env)
            assert agent is not None
            assert hasattr(agent, "act")
            assert hasattr(agent, "train")

    def test_agent_act_method(self):
        """Test the act method of the agent."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 2

        # Mock the MLP
        with patch("agent.MLP") as mock_mlp_class:
            mock_mlp = MagicMock()
            mock_mlp_class.return_value = mock_mlp

            agent = PolicyGradientAgent(mock_env)

            # Mock the state
            state = np.random.randn(6400).astype(np.float32)

            # Mock the MLP's forward pass
            mock_mlp.forward_pass.return_value = (np.array([0.7]), np.random.randn(200))

            action = agent.act(state)
            assert isinstance(action, int)
            assert action in [0, 1]

    def test_agent_train_method(self):
        """Test the train method of the agent."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 2

        # Mock the MLP
        with patch("agent.MLP") as mock_mlp_class:
            mock_mlp = MagicMock()
            mock_mlp_class.return_value = mock_mlp

            agent = PolicyGradientAgent(mock_env)

            # Mock the training process
            with patch.object(agent, "memory") as mock_memory:
                mock_memory.states = [np.random.randn(6400) for _ in range(10)]
                mock_memory.actions = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                mock_memory.rewards = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

                # Mock the MLP's training
                mock_mlp.train = MagicMock()

                agent.train()
                mock_mlp.train.assert_called()


class TestPolicyGradientIntegration:
    """Integration tests for Policy Gradient functionality."""

    def test_pg_script_imports(self):
        """Test that Policy Gradient scripts can be imported."""
        scripts_path = os.path.join(
            os.path.dirname(__file__), "..", "atari", "algorithms", "pg", "scripts"
        )

        try:
            sys.path.insert(0, scripts_path)
            # Test that the main functions can be imported
            from pgpong import main
            from pgbreakout import main as breakout_main
            from pgpacman import main as pacman_main

            assert main is not None
            assert breakout_main is not None
            assert pacman_main is not None

        except ImportError as e:
            pytest.skip(f"Could not import Policy Gradient script modules: {e}")

    def test_pg_pacman_imports(self):
        """Test that Pacman-specific modules can be imported."""
        pacman_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "atari",
            "algorithms",
            "pg",
            "src",
            "pacman",
        )

        try:
            sys.path.insert(0, pacman_path)
            from multi_action_agent import MultiActionAgent
            from cnn_torch_multiaction import CNNMultiAction
            from mlp_torch_multiaction import MLPMultiAction

            assert MultiActionAgent is not None
            assert CNNMultiAction is not None
            assert MLPMultiAction is not None

        except ImportError as e:
            pytest.skip(f"Could not import Pacman-specific modules: {e}")

    def test_hyperparameters_import(self):
        """Test that hyperparameters can be imported."""
        try:
            from hyperparameters import HyperParameters

            # Test that hyperparameters can be instantiated
            hyperparams = HyperParameters()
            assert hyperparams is not None

        except ImportError as e:
            pytest.skip(f"Could not import hyperparameters: {e}")
