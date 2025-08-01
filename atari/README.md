# Atari Game AI Implementation

This directory contains implementations of various reinforcement learning algorithms for Atari games, organized by algorithm type for better maintainability and scalability.

## Directory Structure

```
atari/
├── algorithms/                # Custom algorithm implementations
│   ├── dqn/                   # Deep Q-Network implementation
│   │   ├── scripts/           # DQN training scripts
│   │   └── src/               # DQN source code
│   ├── pg/                    # Policy Gradient implementation
│   │   ├── scripts/           # PG training scripts
│   │   └── src/               # PG source code
├── baselines/                 # Stable Baselines3 implementations
│   ├── atari_baseline_train.py
│   ├── atari_baseline_test.py
│   └── README.md
├── models/                    # All trained models
│   ├── dqn/                   # DQN models by game
│   ├── pg/                    # PG models by game
│   └── baselines/             # Stable Baselines3 models
└── common/                    # Shared utilities
    ├── environments/
    ├── utils/
    └── configs/
```

## Quick Start

### Using Stable Baselines3 (Recommended)
```bash
cd baselines

# Train on any Atari game
python atari_baseline_train.py --algorithm ppo --env ALE/Pong-v5 --timesteps 1000000

# Test trained model
python atari_baseline_test.py --model pong_ppo_cnn_1000000
```

### Using Custom DQN Implementation
```bash
cd algorithms/dqn/scripts

# Train DQN on Pong
python pong-dqn.py

# Train DQN on Pacman
python pacman-dqn.py
```

### Using Custom Policy Gradient Implementation
```bash
cd algorithms/pg/scripts

# Train PG on Pong
python pgpong.py

# Train PG on Breakout
python pgbreakout.py

# Train PG on Pacman
python pgpacman.py
```

## Detailed Documentation

### Algorithm-Specific Guides
- **[algorithms/dqn/README.md](algorithms/dqn/README.md)** - Complete DQN implementation guide
- **[algorithms/pg/README.md](algorithms/pg/README.md)** - Complete Policy Gradient implementation guide
- **[baselines/README.md](baselines/README.md)** - Stable Baselines3 usage guide

## 🎮 Algorithm Implementations

### 1. DQN (Deep Q-Network)
- **Location**: `algorithms/dqn/`
- **Features**: Dueling CNN architecture, experience replay, target networks
- **Best for**: Discrete action spaces, value-based learning
- **Games**: Pong, Pacman, Breakout

### 2. Policy Gradient (REINFORCE)
- **Location**: `algorithms/pg/`
- **Features**: PyTorch implementation, CNN support for Pacman
- **Best for**: Policy-based learning, continuous optimization
- **Games**: Pong, Breakout, Pacman

### 3. Stable Baselines3
- **Location**: `baselines/`
- **Features**: PPO, DQN, A2C algorithms, automatic checkpointing
- **Best for**: Production-ready training, multiple algorithms
- **Games**: All Atari games

## Model Storage

Models are organized by algorithm and game:
- `models/dqn/pong/` - DQN models for Pong
- `models/pg/breakout/` - PG models for Breakout
- `models/baselines/` - Stable Baselines3 models

### Model File Types
- **Stable Baselines3**: `.zip` files (e.g., `pong_ppo_cnn_1000000.zip`)
- **DQN**: `.pkl` files + `.json` (epsilon) + `.txt` (output logs)
- **Policy Gradient**: Custom naming with episode numbers



## Getting Started

1. **Choose your approach**:
   - **Stable Baselines3**: For quick, reliable training
   - **Custom DQN**: For learning and experimentation
   - **Custom PG**: For policy-based approaches

2. **Navigate to the appropriate directory**:
   ```bash
   cd atari/baselines          # For Stable Baselines3
   cd atari/algorithms/dqn     # For custom DQN
   cd atari/algorithms/pg      # For custom Policy Gradient
   ```

3. **Follow the specific README** in each directory for detailed instructions.

## 📦 Dependencies

```bash
pip install stable_baselines3 gymnasium[atari] ale-py torch numpy
```

## 🔄 Algorithm Comparison

| Aspect | Policy Gradient | DQN | Stable Baselines3 |
|--------|----------------|-----|-------------------|
| **Learning Type** | On-policy | Off-policy | Both |
| **Action Spaces** | Continuous & Discrete | Discrete | Both |
| **Sample Efficiency** | Lower | Higher | High |
| **Training Stability** | Less stable | More stable | Very stable |
| **Implementation** | Custom | Custom | Pre-built |
| **Best For** | Learning, experimentation | Learning, experimentation | Production |

## 🤝 Contributing

When adding new algorithms:
1. Create a new directory under `algorithms/`
2. Follow the existing structure (scripts/ and src/)
3. Create a detailed README.md
4. Update this README with documentation
5. Add appropriate model storage directories

## 🔗 Related Documentation

- **[Main README](../README.md)** - Project overview and quick start
- **[Chess README](../chess/README.md)** - Chess engine integration
- **[Graph Search README](../graph-search/README.md)** - Classical search algorithms 