# Atari Training Script Guide

This directory contains training and testing scripts for Atari games using Stable Baselines3. The main scripts are:

- `atari_baseline_train.py`: Flexible, command-line driven trainer that supports multiple algorithms and environments
- `atari_baseline_test.py`: Test trained models with rendering and performance analysis

## Quick Start

### Training
```bash
# Train PPO on Pong (default settings)
python atari_baseline_train.py

# Train DQN on Breakout for 500k timesteps
python atari_baseline_train.py --algorithm dqn --env ALE/Breakout-v5 --timesteps 500000

# Train A2C with CNN policy on Ms. Pacman infinitely
python atari_baseline_train.py -a a2c_cnn -e ALE/MsPacman-v5 -t infinite
```

### Testing
```bash
# Test a PPO model on Pong
python atari_baseline_test.py --model pong_ppo_cnn_100000

# Test a DQN model on Breakout
python atari_baseline_test.py -m breakout_dqn_cnn_500000 -e ALE/Breakout-v5

# Test with custom settings
python atari_baseline_test.py -m pong_a2c_cnn_1M -n 5 -d 0.02
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

- `atari_baseline_test.py`: Test trained models with rendering


## Dependencies

Make sure you have the required packages installed:
```bash
pip install stable_baselines3 gymnasium[atari] ale-py
``` 

## Testing Trained Models

The `atari_baseline_test.py` script allows you to test trained models with visual rendering and performance analysis.

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | **Required** | Path to trained model (without .zip extension) |
| `--env` | `-e` | `ALE/Pong-v5` | Environment to test on |
| `--episodes` | `-n` | `3` | Number of episodes to run |
| `--algorithm` | `-a` | `auto` | Algorithm type: `auto`, `ppo`, `dqn`, `a2c` |
| `--delay` | `-d` | `0.01` | Delay between frames in seconds |

### Basic Testing

```bash
# Test a PPO model (auto-detects algorithm)
python atari_baseline_test.py --model pong_ppo_cnn_100000

# Test a DQN model on Breakout
python atari_baseline_test.py -m breakout_dqn_cnn_500000 -e ALE/Breakout-v5

# Test with more episodes
python atari_baseline_test.py -m pong_a2c_cnn_1M -n 10
```

### Advanced Testing

```bash
# Test with custom frame delay (slower playback)
python atari_baseline_test.py -m pong_ppo_cnn_1M -d 0.05

# Test with specific algorithm (if auto-detection fails)
python atari_baseline_test.py -m my_model -a ppo

# Test on different environment
python atari_baseline_test.py -m pong_ppo_cnn_1M -e ALE/Breakout-v5
```

### Output

The test script provides:
- **Visual rendering**: Watch the agent play in real-time
- **Episode details**: Reward and step count for each episode
- **Performance summary**: Average, best, and worst episode rewards
- **Error handling**: Lists available models if specified model not found

### Example Output
```
Testing PPO model on ALE/Pong-v5
Model: pong_ppo_cnn_100000
Episodes: 3
Frame delay: 0.01s
--------------------------------------------------
Loaded PPO model from pong_ppo_cnn_100000
Starting episode 1
Episode 1 finished with reward: 21.0, steps: 1500
Starting episode 2
Episode 2 finished with reward: 18.0, steps: 1200
Starting episode 3
Episode 3 finished with reward: 24.0, steps: 1800

Summary: 3 episodes completed
Average reward: 21.00
Best episode: 24.0
Worst episode: 18.0
```

### Getting Help
```bash
# Show all available options for testing
python atari_baseline_test.py --help
```

### Troubleshooting Testing

1. **Model not found**: The script will list all available models in the directory
2. **Wrong algorithm**: Use `--algorithm` to specify the correct algorithm type
3. **Slow rendering**: Increase `--delay` for slower playback
4. **No visual output**: Make sure you're running in an environment that supports GUI

## Complete Workflow Example

Here's a complete example of training and testing a model:

```bash
# 1. Train a PPO model on Pong
python atari_baseline_train.py -a ppo -t 500000 -n 4

# 2. Test the trained model
python atari_baseline_test.py -m pong_ppo_cnn_500000 -n 5

# 3. Continue training for more timesteps
python atari_baseline_train.py -a ppo -t 1000000 -n 4

# 4. Test the improved model
python atari_baseline_test.py -m pong_ppo_cnn_1500000 -n 5
``` 