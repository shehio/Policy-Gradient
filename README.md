# Game AI: Policy Gradient Methods, Deep Q-Learning & Classical Approaches

## Project Overview
This repository studies AI in games through multiple approaches: classical search methods (graph-based and tree-based), classical Reinforcement Learning with tabular Q-functions, and modern deep learning approaches using policy networks and temporal difference methods. The project implements both Policy Gradient (REINFORCE) and Deep Q-Network (DQN) reinforcement learning algorithms to train agents to play Atari games including Pong, Breakout, and Ms. Pacman. The code is modular, easy to follow, and inspired by classic deep RL tutorials.

## Repo's Objectives
1. Learn about the theory behind AI in games and the taxonomy of algorithms
2. Run pre-built models
3. Learn about training such models using different algorithms
4. Implement and compare different RL approaches

## Why Game AI?
- **AI benchmarking**: Compare different algorithms and approaches
- **Challenge Human Players**: Create agents that can compete with humans
- **Simulation-Based Testing**: Test if games are playable and balanced
- **NPC Creation**: Avoid highly efficient, yet predictable and boring play styles. Academics have argued for Turing test for NPCs

## Features
- **Dual Algorithm Support**: Both Policy Gradient (REINFORCE) and Deep Q-Network (DQN) implementations
- **Multi-Game Support**: Train on Pong, Breakout, Ms. Pacman, and other Atari games with proper game-specific handling
- **Modern Gymnasium**: Upgraded from deprecated `gym` to modern `gymnasium` library
- **Python 3.13 Support**: Updated to work with the latest Python versions
- **PyTorch Optimization**: Enhanced MLP implementation with batch processing and GPU support
- **CNN Support**: Convolutional Neural Networks for spatial feature learning in games like Ms. Pacman
- **AWS GPU Hosting**: Complete cloud infrastructure with Terraform for GPU training
- **Advanced Model Management**: Sophisticated save/load system with episode tracking and versioning
- **Performance Tracking**: Real-time training metrics and episode statistics
- **Performance Optimizations**: GPU acceleration and batch processing improvements
- **Game-Specific Logic**: Proper handling of different game mechanics (e.g., FIRE action for Breakout)
- **Modular Architecture**: Clean separation of agent, memory, hyperparameters, and game environment
- **Frame Preprocessing**: Optimized image processing for neural network input
- **Type Annotations**: Full type safety for robust development
- **Chess Engine Integration**: Stockfish and Leela Chess Zero for classical game analysis
- **Periodic Model Saving**: Automatic model checkpointing every N steps during training
- **Stable Baselines3 Integration**: Pre-built PPO implementations with custom callbacks

## Games Classification
Games can be classified into many different dimensions based on:
- **Observability**: Full vs partial information
- **Stochasticity**: Deterministic vs random elements
- **Time Granularity**: Turn-based vs real-time

<img width="458" alt="Games I" src="https://github.com/user-attachments/assets/6d388b66-1e0b-4657-9d17-e4603e21968a">

## Gameplay Algorithms
There are three main approaches to gameplay: Planning-based approaches, Reinforcement Learning (RL), and Supervised Learning. Most agents employ one or a composite of these approaches.

### Planning Approaches
- **Tree Search**: Both classical (minimax, alpha-beta) and stochastic (Monte Carlo Tree Search)
- **Evolutionary Planning**: Plans as sequences with mutation and crossover
- **Symbolic Representation**: Rule-based systems

### Reinforcement Learning
RL algorithms can be categorized based on:
- **Model-based vs Model-free**: Whether they learn a model of the environment
- **Value-based vs Policy-based**: Whether they learn Q-values or direct policies
- **On-policy vs Off-policy**: Whether they use the same policy for acting and learning

The most successful RL algorithms in gameplay are generally model-free policy-based, although they're sample inefficient and unstable.

### Supervised Learning
- **Behavioral Cloning**: Learning actions from experienced players
- **Function Approximators**: Neural networks that behave like skilled players

## Policy Gradients vs DQN: A Comparison

| Aspect | Policy Gradients (REINFORCE) | Deep Q-Network (DQN) |
|--------|------------------------------|----------------------|
| **Approach** | Policy-based (learns π(s) → a) | Value-based (learns Q(s,a)) |
| **Learning Type** | On-policy | Off-policy |
| **Action Spaces** | Continuous & Discrete | Discrete only |
| **Policy Type** | Stochastic | Deterministic |
| **Sample Efficiency** | Lower (requires more samples) | Higher (experience replay) |
| **Training Stability** | Less stable (high variance) | More stable (target networks) |
| **Implementation** | Simpler | More complex |
| **Best For** | Continuous actions, stochastic policies | Discrete actions, large state spaces |
| **Key Features** | Direct policy optimization | Experience replay, target networks |

## Algorithm Selection Factors
The choice of an algorithm or category of algorithms usually depends on:
- **Game's Nature**: Observability, stochasticity, time granularity
- **Action Space**: Branching factor and game state representation
- **Time Constraints**: How much time an agent has before making a decision
- **Source Code Availability**: Whether we have access to the game's internals

### Branching Factor
```mermaid
xychart-beta
    title "Chess"
    x-axis "turn" [1, 10, 20, 30, 40, 50, 60, 70]
    y-axis "Branching Factor" 1 --> 50
    line [20, 20, 35,  35, 35, 35, 15, 15]
```

| Game       | Average Branching Factor |
|------------|-----------------------------
| Chess      | 33                       |
| Go         | 250                      |
| Pac-Man    | 4                        |
| Checkers   | 8                        |

Searching a game tree of depth `d` and branching factor `b` is $O(b^d)$.

### Action Space Considerations
The bigger the action space, the longer it takes for a policy (approximated using a neural network) to stabilize. To counter the time requirement and sample inefficiency, `macro-actions` and different/sampled `game-state representation` could be used.

### Game State Representation and Source Code
Sometimes, the source code is present so an API exists (or could exist) to provide a richer state representation while other times, when source code is not present, raw pixels are used for example.

### Additional Considerations
- **Training Time**: How long it takes to train the agent
- **Transferability**: Whether the trained policy can transfer from one game to another

## Algorithm Comparison

| Algorithm Family       | Pros | Cons |
|------------------------|------|------|
| Tree Search            | Strong for games that require adversarial planning | Sometimes can be very expensive to go deeper depending on branching factor. Also, don't work well for hidden information games |
| Evolutionary Planning  | Good for complex optimization problems | Can be computationally expensive |
| RL                     | Strong with games that require perception, motor skills, and continuous movement | Game state representation dictates how cheap or expensive training and evaluation is. Also, subject to the pitfalls of function approximators |
| Supervised Learning    | Fast to implement, good for imitation learning | Requires expert demonstrations, may not generalize well |

## Directory Structure
```
Policy-Gradient/
├── atari/                 # Atari game implementations
│   ├── src/               # Source code modules
│   │   ├── pg/            # Policy Gradient implementation
│   │   │   ├── agent.py   # PG agent logic
│   │   │   ├── game.py    # Game environment
│   │   │   ├── hyperparameters.py # PG hyperparameters
│   │   │   ├── memory.py  # Episode memory buffer
│   │   │   ├── mlp.py     # Legacy MLP (NumPy)
│   │   │   └── mlp_torch.py # Modern MLP (PyTorch)
│   │   └── dqn/           # DQN implementation
│   │       ├── agent.py   # DQN agent logic
│   │       ├── model.py   # Dueling CNN model
│   │       └── config/    # DQN configuration classes
│   ├── scripts/           # Executable scripts
│   │   ├── policy-gradient/ # PG training scripts
│   │   ├── dqn/           # DQN training scripts
│   │   └── game_model_manager.py # Model management utilities
│   └── baselines/         # Stable Baselines3 implementations
│       ├── helpers.py     # Training utilities and callbacks
│       ├── breakout_train.py # PPO training for Breakout
│       ├── pacman_train.py # PPO training for Ms. Pacman
│       └── breakout_test.py # Test trained models
├── chess/                 # Chess engine integration
│   ├── engine-scripts/    # Build scripts for Stockfish and Leela
│   └── game-analysis.py   # Chess game analysis tools
├── graph-search/          # Classical search algorithms
├── models/                # Trained model files
├── terraform/             # Infrastructure as Code
├── assets/                # Images and diagrams
└── requirements.txt       # Python dependencies
```

## Quick Start
1. **Install Python 3.11+** (recommended: 3.13)
2. **Setup environment:**
   ```sh
   python3.13 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Policy Gradient Training (3 Scripts)

**1. Train on Pong (Binary Actions):**
```sh
python atari/scripts/policy-gradient/pgpong.py
```
- Uses MLP with binary action space (UP/DOWN)
- Optimized for Pong's simple mechanics
- Fast training with 2-action policy

**2. Train on Breakout (Binary Actions):**
```sh
python atari/scripts/policy-gradient/pgbreakout.py
```
- Uses MLP with binary action space (LEFT/RIGHT)
- Handles Breakout's paddle movement and ball physics
- Includes FIRE action for ball release

**3. Train on Ms. Pacman (Multi-Action CNN):**
```sh
python atari/scripts/policy-gradient/pgpacman.py
```
- Uses CNN with 9-action space (all directions + NOOP)
- Color-aware preprocessing for ghost detection
- Advanced exploration with temperature scheduling
- Best for complex spatial reasoning

### Unified Policy Gradient Trainer

**Single script for all games:**
```sh
# Train Pong (loads latest model by default)
python atari/scripts/policy-gradient/pg_trainer.py pong --render

# Train Breakout with custom parameters
python atari/scripts/policy-gradient/pg_trainer.py breakout --learning-rate 2e-4 --batch-size 5

# Train Pacman from scratch (no pre-trained model)
python atari/scripts/policy-gradient/pg_trainer.py pacman --no-load-network
```

### DQN Training

**Train on Pong (DQN):**
```sh
python atari/scripts/dqn/pong-dqn.py
```
- Uses Dueling CNN architecture
- Experience replay and target networks
- Optimized hyperparameters for Pong

**Train on Ms. Pacman (DQN):**
```sh
python atari/scripts/dqn/pacman-dqn.py
```
- Uses Dueling CNN with frame stacking
- Advanced exploration strategies
- Optimized for complex maze navigation

### Stable Baselines3 (Atari)

**Train Breakout with PPO (with periodic saving):**
```sh
cd atari/baselines
python breakout_train.py
```
- Saves model every 100,000 steps
- Uses PPO algorithm with CNN policy
- Automatic checkpointing with game name in filename

**Train Ms. Pacman with PPO:**
```sh
python pacman_train.py
```
- Saves model every 100,000 steps
- Optimized for complex maze navigation
- Uses CNN policy for spatial reasoning

**Test trained model:**
```sh
python breakout_test.py
```

### Chess Engine Integration

**Build chess engines:**
```sh
cd chess
./build_chess_engines.sh
```

**Test Stockfish integration:**
```sh
python test_stockfish.py
```

## Policy Gradient Implementation

### REINFORCE Algorithm

![REINFORCE Algorithm](assets/reinforce.png)

### Network Architecture
The PyTorch policy network features:
- **Input Layer**: 6400 units (80x80 preprocessed frames)
- **Hidden Layer**: 200 units with ReLU activation
- **Output Layer**: 1 unit with Sigmoid activation
- **GPU Acceleration**: Automatic CUDA support when available
- **Batch Processing**: Optimized for efficient training

### Ms. Pacman Action Space

In the Gym environment for Pac-Man (specifically Ms. Pacman on Atari 2600), there are 9 discrete actions available by default instead of just 4 because the action space is designed to allow for both the four primary cardinal movements—up, down, left, and right—and the four diagonals (up-right, up-left, down-right, down-left), as well as a NOOP (no operation) action.

Here's a table summarizing the 9 actions:

| Value | Meaning |
|-------|---------|
| 0 | NOOP |
| 1 | UP |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | DOWN |
| 5 | UPRIGHT |
| 6 | UPLEFT |
| 7 | DOWNRIGHT |
| 8 | DOWNLEFT |

### CNN Architecture for Ms. Pacman
For Ms. Pacman, we use a Convolutional Neural Network (CNN) to better process the spatial information:

- **Input**: 7 channels (6 color masks + grayscale) at 80×80 resolution
- **Conv1**: 32 filters, 8×8 kernel, stride 4 → Learns large spatial patterns
- **Conv2**: 64 filters, 4×4 kernel, stride 2 → Learns medium patterns
- **Conv3**: 64 filters, 3×3 kernel, stride 1 → Learns fine details
- **Fully Connected**: 200 hidden units → Final action probabilities
- **Output**: 9 action probabilities (one for each possible move)

## DQN Implementation

### Network Architecture
The DQN uses a Dueling CNN architecture:
- **Input Layer**: Preprocessed game frames (80x64 grayscale)
- **Convolutional Layers**: Feature extraction from game pixels
- **Dueling Architecture**: Separate value and advantage streams
- **Output Layer**: Q-values for each action
- **Experience Replay**: Stores and samples past experiences
- **Target Network**: Stabilizes training with separate target network

### Key Features
- **Experience Replay Buffer**: Stores (state, action, reward, next_state, done) tuples
- **Target Network**: Separate network for computing target Q-values
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Frame Stacking**: Uses 4 consecutive frames as state representation

## Stable Baselines3 Implementation

### PPO Training with Periodic Saving
The baselines implementation uses Stable Baselines3's PPO algorithm with custom callbacks for periodic model saving:

- **SaveEveryStepCallback**: Automatically saves models every N steps (configurable)
- **Game-specific naming**: Models are saved with game name prefix (e.g., `Breakout_step_100000`)
- **CNN Policy**: Uses convolutional neural networks for image-based observations
- **Tensorboard logging**: Real-time training metrics and visualization

### Training Configuration
```python
# Example configuration from breakout_train.py
env_name = 'ALE/Breakout-v5'
timesteps = 100_000_000
epochs = 1000
save_freq = 100_000  # Save every 100k steps
```

## Classical Games Support
- **Atari 2600**: Through Arcade Learning Environment (ALE)
- **Nintendo NES**: Through various emulators
- **Commodore 64**: Through emulation
- **ZX Spectrum**: Through emulation

## Regret and Card Games
Card games feature hidden information. When playing poker and similar games, Counterfactual Regret Minimization algorithms are used. Agents learn by self-play similar to how RL learns by playing checkers and backgammon. Regret is the difference between the action that was taken and the action that could have been taken.

```python
def get_action(self):
    strategy = self.get_strategy()
    return np.random.choice(self.num_actions, p=strategy)

def train(self):
    for _ in range(self.num_iterations):
        action = self.get_action()
        payoff = np.random.rand()  # Simulating the payoff randomly
        self.update_regret(action, payoff)
```

## Game Classification
- **Board Games**: Chess, Go, Checkers
- **Card Games**: Poker, Bridge
- **Classic Arcade Games**: Pac-Man, Pong, Breakout
- **Strategy Games**: Civilization, StarCraft
- **Racing Games**: Need for Speed, Mario Kart
- **First Person Games**: Doom, Quake
- **Interactive Games**: Minecraft, Roblox

## Cloud GPU Training
1. **Deploy to AWS:**
   ```sh
   cd terraform
   terraform init
   terraform apply
   terraform destroy
   ```

2. **Monitor training:**
   ```sh
   ./check_status.sh
   ```

3. **Download results:**
   ```sh
   scp -i your-key.pem ubuntu@your-instance:~/Policy-Gradient/models/ ./models/
   ```

## Glossary
- **[NPC](https://en.wikipedia.org/wiki/Non-player_character)**: A non-player character (NPC) is a character in a game that is not controlled by a player
- **[Game Tree](https://en.wikipedia.org/wiki/Game_tree)**: A tree representing all possible game states
- **[Minimax](https://en.wikipedia.org/wiki/Minimax)**: Assumes no collusion from multiple players for more than two players
- **[Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)**: Optimization technique for minimax algorithm
- **[Monte Carlo Tree Search](https://github.com/shehio/monte-carlo-tree-search)**: Stochastic tree search algorithm
- **[Branching Factor](https://en.wikipedia.org/wiki/Branching_factor)**: Average number of possible moves at each position

## Resources
- [Policy Gradients: Pong from Pixels](https://youtu.be/tqrcjHuNdmQ?si=XElMeYhPr7vCBb1b)
- [REINFORCE Algorithm](https://youtu.be/5eSh5F8gjWU?si=b1lRf6Ks_q_0dekA)
- [Karpathy's "Pong from Pixels"](http://karpathy.github.io/2016/05/31/rl/)
- [Gymnasium](https://gymnasium.farama.org/) | [PyTorch](https://pytorch.org/docs/)
- [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [Microsoft vs Ms. Pac-Man](https://blogs.microsoft.com/ai/divide-conquer-microsoft-researchers-used-ai-master-ms-pac-man/)
- [Berkeley's Pac-Man Projects](http://ai.berkeley.edu/project_overview)
- [Artificial Intelligence and Games](https://gameaibook.org/book.pdf)

## Contributing
This project is designed for educational purposes. Feel free to:
- Add more Atari games
- Implement additional RL algorithms
- Improve cloud infrastructure
- Add monitoring tools
- Implement classical search algorithms
- Add support for more game types

## License
This project is open source and available under the MIT License.
