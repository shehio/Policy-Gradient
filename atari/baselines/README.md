# Atari Training Script Guide

This directory contains training scripts for Atari games using Stable Baselines3. The main script `atari_baseline_train.py` is a flexible, command-line driven trainer that supports multiple algorithms and environments.

## Quick Start

### Basic Usage
```bash
# Train PPO on Pong (default settings)
python atari_baseline_train.py

# Train DQN on Breakout for 500k timesteps
python atari_baseline_train.py --algorithm dqn --env ALE/Breakout-v5 --timesteps 500000

# Train A2C with CNN policy on Ms. Pacman infinitely
python atari_baseline_train.py -a a2c_cnn -e ALE/MsPacman-v5 -t infinite
```

## Command Line Arguments

### Required Arguments
None - all arguments have sensible defaults.

### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--algorithm` | `-a` | `ppo` | Algorithm to use: `ppo`, `dqn`, `a2c`, `a2c_cnn` |
| `--timesteps` | `-t` | `100000` | Training timesteps (use "infinite" for continuous) |
| `--env` | `-e` | `ALE/Pong-v5` | Environment to train on |
| `--n_envs` | `-n` | `1` | Number of parallel environments |
| `--seed` | `-s` | `0` | Random seed for reproducibility |

## Algorithms

### 1. PPO (Proximal Policy Optimization)
- **Best for**: Most Atari games, good balance of performance and stability
- **Policy**: CNN (Convolutional Neural Network)
- **Use case**: General purpose training

```bash
python atari_baseline_train.py --algorithm ppo
```

### 2. DQN (Deep Q-Network)
- **Best for**: Games with discrete actions, value-based learning
- **Policy**: CNN
- **Use case**: When you want to learn Q-values

```bash
python atari_baseline_train.py --algorithm dqn
```

### 3. A2C (Advantage Actor-Critic) - MLP
- **Best for**: Simpler environments, faster training
- **Policy**: MLP (Multi-Layer Perceptron)
- **Use case**: When CNN is not needed

```bash
python atari_baseline_train.py --algorithm a2c
```

### 4. A2C (Advantage Actor-Critic) - CNN
- **Best for**: Complex visual environments
- **Policy**: CNN
- **Use case**: When you need spatial reasoning

```bash
python atari_baseline_train.py --algorithm a2c_cnn
```

## Training Modes

### Finite Training
Train for a specific number of timesteps:

```bash
# Train for 100k timesteps (default)
python atari_baseline_train.py

# Train for 1M timesteps
python atari_baseline_train.py --timesteps 1000000

# Train for 500k timesteps
python atari_baseline_train.py -t 500000
```

### Infinite Training
Train continuously until manually stopped:

```bash
# Infinite training (press Ctrl+C to stop)
python atari_baseline_train.py --timesteps infinite

# Or use the short form
python atari_baseline_train.py -t inf
```

## Environment Options

### Supported Atari Games
- `ALE/Pong-v5` (default)
- `ALE/Breakout-v5`
- `ALE/MsPacman-v5`
- `ALE/SpaceInvaders-v5`
- `ALE/Asteroids-v5`
- `ALE/BeamRider-v5`
- And many more...

### Example Environment Usage
```bash
# Train on Breakout
python atari_baseline_train.py --env ALE/Breakout-v5

# Train on Ms. Pacman
python atari_baseline_train.py -e ALE/MsPacman-v5

# Train on Space Invaders
python atari_baseline_train.py --env ALE/SpaceInvaders-v5
```

## Parallel Environments

### Single Environment (Default)
```bash
python atari_baseline_train.py --n_envs 1
```
- Good for testing and debugging
- Lower memory usage
- Slower training

### Multiple Environments
```bash
# 4 parallel environments (recommended)
python atari_baseline_train.py --n_envs 4

# 8 parallel environments (faster)
python atari_baseline_train.py -n 8

# 16 parallel environments (high performance)
python atari_baseline_train.py --n_envs 16
```
- Faster training
- Better sample efficiency
- Higher memory usage

## Model Management

### Automatic Checkpoint Loading
The script automatically finds and loads the most recent model for the chosen algorithm and environment:

```bash
# If pong_ppo_cnn_100000.zip exists, it will be loaded
python atari_baseline_train.py --algorithm ppo

# If breakout_dqn_cnn_500000.zip exists, it will be loaded
python atari_baseline_train.py -a dqn -e ALE/Breakout-v5
```

### Model Naming Convention
Models are saved with the pattern: `{env_name}_{algorithm}_{policy}_{timesteps}.zip`

Examples:
- `pong_ppo_cnn_100000.zip`
- `breakout_dqn_cnn_500000.zip`
- `mspacman_a2c_cnn_1000000.zip`

### Checkpoint Frequency
- **Finite training**: Model saved once at the end
- **Infinite training**: Model saved every 100,000 timesteps

## Complete Examples

### Example 1: Quick PPO Training
```bash
python atari_baseline_train.py --algorithm ppo --timesteps 100000 --n_envs 4
```
- Trains PPO on Pong for 100k timesteps
- Uses 4 parallel environments
- Saves as `pong_ppo_cnn_100000.zip`

### Example 2: Long DQN Training
```bash
python atari_baseline_train.py -a dqn -e ALE/Breakout-v5 -t 2000000 -n 8
```
- Trains DQN on Breakout for 2M timesteps
- Uses 8 parallel environments
- Saves as `breakout_dqn_cnn_2000000.zip`

### Example 3: Infinite A2C Training
```bash
python atari_baseline_train.py -a a2c_cnn -e ALE/MsPacman-v5 -t infinite -n 4
```
- Trains A2C with CNN on Ms. Pacman infinitely
- Uses 4 parallel environments
- Saves checkpoints every 100k timesteps
- Press Ctrl+C to stop

### Example 4: Reproducible Training
```bash
python atari_baseline_train.py --algorithm ppo --seed 42 --n_envs 1
```
- Trains PPO with fixed random seed
- Uses single environment for debugging
- Results will be reproducible

## Performance Tips

### For Faster Training
```bash
# Use more parallel environments
python atari_baseline_train.py --n_envs 8

# Use infinite training for long sessions
python atari_baseline_train.py -t infinite

# Use PPO (generally fastest to converge)
python atari_baseline_train.py -a ppo
```

### For Memory Efficiency
```bash
# Use fewer parallel environments
python atari_baseline_train.py --n_envs 1

# Use MLP policy instead of CNN
python atari_baseline_train.py -a a2c
```

### For Best Results
```bash
# Use 4-8 parallel environments
python atari_baseline_train.py --n_envs 4

# Train for at least 1M timesteps
python atari_baseline_train.py -t 1000000

# Use CNN policy for visual games
python atari_baseline_train.py -a ppo  # or a2c_cnn
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--n_envs` to 1 or 2
2. **Slow Training**: Increase `--n_envs` to 4 or 8
3. **Poor Performance**: Train for longer (1M+ timesteps)
4. **Model Not Loading**: Check if model files exist in current directory

### Getting Help
```bash
# Show all available options
python atari_baseline_train.py --help
```

## Related Files

- `pong_test.py`: Test trained models with rendering
- `breakout_train.py`: Specific training script for Breakout
- `pacman_train.py`: Specific training script for Ms. Pacman
- `helpers.py`: Utility functions for training

## Dependencies

Make sure you have the required packages installed:
```bash
pip install stable_baselines3 gymnasium[atari] ale-py
``` 