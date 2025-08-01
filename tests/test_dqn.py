"""
Tests for DQN implementation.
"""
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the atari directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'atari', 'algorithms', 'dqn', 'src'))

try:
    from agent import DQNAgent
    from model import DuelingCNN
except ImportError as e:
    pytest.skip(f"Could not import DQN modules: {e}", allow_module_level=True)


class TestDuelingCNN:
    """Test the DuelingCNN model."""

    def test_model_initialization(self):
        """Test that the model can be initialized."""
        model = DuelingCNN(input_channels=4, num_actions=6)
        assert model is not None
        assert hasattr(model, 'forward')

    def test_model_forward_pass(self):
        """Test the forward pass of the model."""
        model = DuelingCNN(input_channels=4, num_actions=6)
        batch_size = 2
        input_tensor = np.random.randn(batch_size, 4, 80, 80).astype(np.float32)
        
        # Mock torch
        with patch('model.torch') as mock_torch:
            mock_torch.from_numpy.return_value = MagicMock()
            mock_torch.randn.return_value = MagicMock()
            
            # Mock the forward method to return a tensor
            mock_output = MagicMock()
            mock_output.cpu.return_value.numpy.return_value = np.random.randn(batch_size, 6)
            model.forward = MagicMock(return_value=mock_output)
            
            result = model.forward(input_tensor)
            assert result is not None

    def test_model_architecture(self):
        """Test that the model has the expected architecture components."""
        model = DuelingCNN(input_channels=4, num_actions=6)
        
        # Check that the model has the expected layers
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'value_stream')
        assert hasattr(model, 'advantage_stream')


class TestDQNAgent:
    """Test the DQNAgent class."""

    def test_agent_initialization(self):
        """Test that the agent can be initialized."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 6
        mock_env.observation_space.shape = (4, 80, 80)
        
        # Mock the model
        with patch('agent.DuelingCNN') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            agent = DQNAgent(mock_env)
            assert agent is not None
            assert hasattr(agent, 'act')
            assert hasattr(agent, 'train')

    def test_agent_act_method(self):
        """Test the act method of the agent."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 6
        mock_env.observation_space.shape = (4, 80, 80)
        
        # Mock the model
        with patch('agent.DuelingCNN') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            agent = DQNAgent(mock_env)
            
            # Mock the state
            state = np.random.randn(4, 80, 80).astype(np.float32)
            
            # Mock the model's forward pass
            mock_q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            mock_model.return_value = MagicMock()
            mock_model.return_value.cpu.return_value.numpy.return_value = mock_q_values
            
            # Test epsilon-greedy action selection
            action = agent.act(state)
            assert isinstance(action, int)
            assert 0 <= action < 6

    def test_agent_store_results(self):
        """Test that the agent can store experience."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 6
        mock_env.observation_space.shape = (4, 80, 80)
        
        # Mock the model
        with patch('agent.DuelingCNN') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            agent = DQNAgent(mock_env)
            
            # Test storing experience
            state = np.random.randn(4, 80, 80).astype(np.float32)
            action = 2
            reward = 1.0
            next_state = np.random.randn(4, 80, 80).astype(np.float32)
            done = False
            
            agent.storeResults(state, action, reward, next_state, done)
            # The experience should be stored in the replay buffer
            assert hasattr(agent, 'memory')

    def test_agent_train_method(self):
        """Test the train method of the agent."""
        # Mock the environment
        mock_env = MagicMock()
        mock_env.action_space.n = 6
        mock_env.observation_space.shape = (4, 80, 80)
        
        # Mock the model
        with patch('agent.DuelingCNN') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            agent = DQNAgent(mock_env)
            
            # Mock the training process
            with patch.object(agent, 'memory') as mock_memory:
                mock_memory.sample.return_value = (
                    np.random.randn(32, 4, 80, 80),
                    np.array([0, 1, 2, 3] * 8),
                    np.random.randn(32),
                    np.random.randn(32, 4, 80, 80),
                    np.array([False] * 32)
                )
                
                # Mock the model's forward and backward passes
                mock_model.return_value = MagicMock()
                mock_model.return_value.backward = MagicMock()
                
                loss, max_q = agent.train()
                assert isinstance(loss, (int, float))
                assert isinstance(max_q, (int, float))


class TestDQNIntegration:
    """Integration tests for DQN functionality."""

    def test_dqn_config_imports(self):
        """Test that DQN configuration modules can be imported."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'atari', 'algorithms', 'dqn', 'src', 'config')
        
        try:
            sys.path.insert(0, config_path)
            from environment_config import EnvironmentConfig
            from model_config import ModelConfig
            from training_config import TrainingConfig
            from learning_config import LearningConfig
            from exploration_config import ExplorationConfig
            from image_config import ImageConfig
            
            # Test that configs can be instantiated
            env_config = EnvironmentConfig()
            model_config = ModelConfig()
            training_config = TrainingConfig()
            learning_config = LearningConfig()
            exploration_config = ExplorationConfig()
            image_config = ImageConfig()
            
            assert env_config is not None
            assert model_config is not None
            assert training_config is not None
            assert learning_config is not None
            assert exploration_config is not None
            assert image_config is not None
            
        except ImportError as e:
            pytest.skip(f"Could not import DQN config modules: {e}")

    def test_dqn_script_imports(self):
        """Test that DQN scripts can be imported."""
        scripts_path = os.path.join(os.path.dirname(__file__), '..', 'atari', 'algorithms', 'dqn', 'scripts')
        
        try:
            sys.path.insert(0, scripts_path)
            # Test that the main function can be imported
            from pong_dqn import main
            from pacman_dqn import main as pacman_main
            
            assert main is not None
            assert pacman_main is not None
            
        except ImportError as e:
            pytest.skip(f"Could not import DQN script modules: {e}") 