# DQN (Deep Q-Network) Implementation

This directory contains a custom implementation of Deep Q-Network (DQN) for Atari games.

## Structure

```
dqn/
├── scripts/           # Training scripts
│   ├── pong-dqn.py    # Train DQN on Pong
│   ├── pacman-dqn.py  # Train DQN on Ms. Pacman
│   ├── dqn_trainer.py # Generic DQN trainer
│   └── dqn_tester.py  # Test trained DQN models
└── src/               # Source code
    ├── agent.py       # DQN agent implementation
    ├── model.py       # Neural network architecture
    └── config/        # Configuration classes
        ├── environment_config.py
        ├── model_config.py
        ├── training_config.py
        ├── learning_config.py
        ├── exploration_config.py
        └── image_config.py
```

## Features

- **Dueling CNN Architecture**: Separate value and advantage streams
- **Experience Replay**: Stores and samples past experiences
- **Target Networks**: Stabilizes training with separate target network
- **Frame Stacking**: Uses 4 consecutive frames as state representation
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation

## Usage

### Training

```bash
# Train on Pong
python scripts/pong-dqn.py

# Train on Ms. Pacman
python scripts/pacman-dqn.py

# Use generic trainer
python scripts/dqn_trainer.py --game pong --episodes 1000
```

### Testing

```bash
# Test trained model
python scripts/dqn_tester.py --model path/to/model --game pong
```

## Configuration

The DQN implementation uses a modular configuration system:

- **Environment Config**: Game-specific settings
- **Model Config**: Neural network architecture
- **Training Config**: Learning parameters
- **Exploration Config**: Epsilon-greedy settings
- **Image Config**: Frame preprocessing

## Model Storage

Trained models are saved in:
- `../../models/dqn/pong/` for Pong models
- `../../models/dqn/pacman/` for Pacman models

## Dependencies

```bash
pip install torch numpy gymnasium[atari] ale-py
```

## Architecture

The DQN uses a Dueling CNN architecture:
1. **Convolutional Layers**: Feature extraction from game pixels
2. **Dueling Streams**: Separate value and advantage computation
3. **Fully Connected**: Final Q-value computation
4. **Output**: Q-values for each action

## Performance

- **Pong**: Typically achieves 20+ average reward after 1M timesteps
- **Pacman**: Complex maze navigation with ghost avoidance
- **Training Time**: 1-2 hours for 1M timesteps on GPU 