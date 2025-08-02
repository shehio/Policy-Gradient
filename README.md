# Game AI & Reinforcement Learning

[![Build Status](https://github.com/shehio/rl/actions/workflows/test.yml/badge.svg)](https://github.com/shehio/rl/actions/workflows/test.yml)
[![Tests](https://github.com/shehio/rl/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/shehio/rl/actions/workflows/test.yml)


A comprehensive repository for implementing and experimenting with various reinforcement learning algorithms and classical game AI techniques. The objective of this repo is not to create the best playing agent but to experiment and explore different agents playing board games and video games.

## Project Overview

This repository contains implementations of:
- **Reinforcement Learning Algorithms**: Policy Gradient (REINFORCE), Deep Q-Network (DQN), PPO, A2C
- **Classical Game AI**: Chess engines, graph search algorithms
- **Game Environments**: Atari games, chess, puzzle games
- **Infrastructure**: Cloud deployment, model management

## Directory

```
rl/
├── atari/                 # Atari game implementations
│   ├── algorithms/        # Custom RL implementations
│   │   ├── dqn/          # Deep Q-Network
│   │   └── pg/           # Policy Gradient
│   ├── baselines/        # Stable Baselines3 implementations
│   ├── models/           # All trained models
│   └── common/           # Shared utilities
├── chess/                # Chess engine integration
├── graph-search/         # Classical search algorithms
├── terraform/            # Cloud infrastructure
└── assets/               # Images and resources
```

## Quick Start

### Prerequisites
```bash
./init.sh
```

### Atari Games (Reinforcement Learning)
```bash
# For detailed instructions, see atari/README.md
cd atari/baselines
python atari_baseline_train.py --algorithm ppo --env ALE/Pong-v5
```

### Chess Engines
```bash
# For detailed instructions, see chess/README.md
cd chess
./build_chess_engines.sh
python stockfish_vs_leela.py
```

### Graph Search
```bash
# For detailed instructions, see graph-search/README.md
cd graph-search
python river-crossing-puzzle.py
```

## Detailed Documentation

### Theory & Algorithms
- **[THEORY.md](THEORY.md)** - Comprehensive guide to RL algorithms and theory

### Atari Games & RL
- **[atari/README.md](atari/README.md)** - Complete guide to Atari implementations
- **[atari/algorithms/dqn/README.md](atari/algorithms/dqn/README.md)** - DQN implementation details
- **[atari/algorithms/pg/README.md](atari/algorithms/pg/README.md)** - Policy Gradient implementation details
- **[atari/baselines/README.md](atari/baselines/README.md)** - Stable Baselines3 usage guide

### Chess & Classical AI
- **[chess/README.md](chess/README.md)** - Chess engine integration guide

### Infrastructure
- **[terraform/README.md](terraform/README.md)** - Cloud deployment setup

## 🎮 Supported Games & Algorithms

### Atari Games
- **Pong**: Binary actions (UP/DOWN)
- **Breakout**: Binary actions (LEFT/RIGHT)
- **Ms. Pacman**: 9 discrete actions
- **Any Atari game**: Via Stable Baselines3

### Algorithms
- **Policy Gradient (REINFORCE)**: Custom PyTorch implementation
- **Deep Q-Network (DQN)**: Custom implementation with Dueling CNN
- **Stable Baselines3**: PPO, DQN, A2C with automatic checkpointing

### Chess
- **Stockfish**: World's strongest chess engine
- **Leela Chess Zero**: Neural network-based engine
- **Game Analysis**: Move analysis and evaluation

## Key Features

- **Algorithm-Centric Organization**: Each algorithm has its own space
- **Model Management**: Organized model storage by algorithm and game
- **Cloud Ready**: Terraform configurations for AWS deployment
- **Comprehensive Testing**: Test scripts for all implementations
- **Documentation**: Detailed guides for each component

## Performance

### Demo: Pong DQN Agent

Watch our trained Deep Q-Network (DQN) agent playing Pong after 6 million training steps:

![Pong DQN Gameplay](assets/videos/pong_dqn_cnn_6M_gameplay.gif)

**Model Details:**
- **Algorithm**: Deep Q-Network (DQN) with Dueling CNN architecture
- **Environment**: ALE/Pong-v5 (Atari Learning Environment)
- **Training Steps**: 6,000,000
- **Architecture**: Convolutional Neural Network with dueling streams
- **Performance**: Achieves consistent high scores through strategic paddle positioning

**Technical Specifications:**
- **Input**: 4 stacked grayscale frames (84x84 pixels)
- **Actions**: 2 discrete actions (UP/DOWN)
- **Network**: 3 convolutional layers + dueling streams (value + advantage)
- **Training**: Experience replay, target networks, epsilon-greedy exploration

The agent demonstrates strong gameplay by learning to predict ball trajectories and position the paddle optimally. The dueling architecture separates value and advantage estimation for more stable learning.

### Demo: Breakout DQN Agent

Watch our trained Deep Q-Network (DQN) agent playing Breakout after 5 million training steps:

![Breakout DQN Gameplay](assets/videos/breakout_dqn_cnn_5M_gameplay.gif)

**Model Details:**
- **Algorithm**: Deep Q-Network (DQN) with Dueling CNN architecture
- **Environment**: ALE/Breakout-v5 (Atari Learning Environment)
- **Training Steps**: 5,000,000
- **Architecture**: Convolutional Neural Network with dueling streams
- **Performance**: Demonstrates advanced brick-breaking strategy and paddle control

**Technical Specifications:**
- **Input**: 4 stacked grayscale frames (84x84 pixels)
- **Actions**: 3 discrete actions (LEFT/RIGHT/FIRE)
- **Network**: 3 convolutional layers + dueling streams (value + advantage)
- **Training**: Experience replay, target networks, epsilon-greedy exploration

The agent shows improved strategic gameplay by learning to break bricks efficiently while maintaining paddle control. The enhanced life tracking ensures episodes run until all lives are lost, providing longer gameplay demonstrations.

## Contributing

1. Choose the appropriate subdirectory for your contribution
2. Follow the existing code structure and documentation
3. Update relevant README files
4. Test your changes thoroughly

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [Stockfish](https://stockfishchess.org/) - Chess engine
- [Leela Chess Zero](https://lczero.org/) - Neural chess engine
