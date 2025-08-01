# Policy Gradient (REINFORCE) Implementation

This directory contains a custom implementation of Policy Gradient (REINFORCE) algorithm for Atari games.

## Structure

```
pg/
├── scripts/               # Training scripts
│   ├── pgpong.py         # Train PG on Pong
│   ├── pgbreakout.py     # Train PG on Breakout
│   ├── pgpacman.py       # Train PG on Ms. Pacman
│   ├── pg_trainer.py     # Generic PG trainer
│   └── pg_tester.py      # Test trained PG models
└── src/                   # Source code
    ├── agent.py          # PG agent implementation
    ├── game.py           # Game environment wrapper
    ├── memory.py         # Episode memory buffer
    ├── mlp_torch.py      # PyTorch MLP implementation
    ├── mlp.py            # NumPy MLP (legacy)
    ├── hyperparameters.py # Training hyperparameters
    └── pacman/           # Pacman-specific implementations
        ├── multi_action_agent.py
        ├── cnn_torch_multiaction.py
        ├── mlp_torch_multiaction.py
        └── preprocess_pacman.py
```

## Features

- **REINFORCE Algorithm**: Direct policy optimization
- **PyTorch Implementation**: Modern MLP with GPU support
- **CNN Support**: Convolutional networks for Pacman
- **Episode Memory**: Stores complete episodes for training
- **Multi-Action Support**: 9-action space for Pacman
- **Frame Preprocessing**: Optimized image processing

## Usage

### Training

```bash
# Train on Pong (binary actions)
python scripts/pgpong.py

# Train on Breakout (binary actions)
python scripts/pgbreakout.py

# Train on Ms. Pacman (9 actions)
python scripts/pgpacman.py

# Use generic trainer
python scripts/pg_trainer.py pong --render
```

### Testing

```bash
# Test trained model
python scripts/pg_tester.py --model path/to/model --game pong
```

## Game-Specific Implementations

### Pong & Breakout
- **Action Space**: Binary (UP/DOWN for Pong, LEFT/RIGHT for Breakout)
- **Network**: MLP with 6400 input units, 200 hidden units
- **Policy**: Sigmoid output for binary action selection

### Ms. Pacman
- **Action Space**: 9 discrete actions (8 directions + NOOP)
- **Network**: CNN with 7 input channels (color masks + grayscale)
- **Policy**: Softmax output for multi-action selection
- **Preprocessing**: Color-aware preprocessing for ghost detection

## Configuration

Key hyperparameters in `src/hyperparameters.py`:
- Learning rate
- Batch size
- Episode length
- Network architecture
- Training duration

## Model Storage

Trained models are saved in:
- `../../models/pg/pong/` for Pong models
- `../../models/pg/breakout/` for Breakout models
- `../../models/pg/pacman/` for Pacman models

## Dependencies

```bash
pip install torch numpy gymnasium[atari] ale-py
```

## Architecture

### MLP Architecture (Pong/Breakout)
- **Input**: 6400 units (80x80 preprocessed frames)
- **Hidden**: 200 units with ReLU activation
- **Output**: 1 unit with Sigmoid activation

### CNN Architecture (Pacman)
- **Input**: 7 channels at 80x80 resolution
- **Conv1**: 32 filters, 8x8 kernel, stride 4
- **Conv2**: 64 filters, 4x4 kernel, stride 2
- **Conv3**: 64 filters, 3x3 kernel, stride 1
- **Fully Connected**: 200 hidden units
- **Output**: 9 action probabilities

## Performance

- **Pong**: Achieves 20+ average reward after 70k episodes
- **Breakout**: Good paddle control and ball tracking
- **Pacman**: Complex maze navigation with ghost avoidance
- **Training Time**: 2-4 hours for 100k episodes on CPU

## Key Differences from DQN

- **Policy-based**: Learns π(s) → a directly
- **On-policy**: Uses current policy for acting and learning
- **Stochastic**: Outputs action probabilities
- **Episode-based**: Trains on complete episodes
- **Higher variance**: Less stable but can handle continuous actions 