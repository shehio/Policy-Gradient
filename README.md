# Game AI
This is a repository for studying AI in games. It investigates classical search methods, such as graph-based and tree-based methods, classical Reinforcement Learning methods with tabular Q-function, and more modern approaches relying on a policy network and temporal difference. The plan is to significantly expand this repository, covering more challenging game states and action spaces and focusing on sample efficiency and computational resources, offering exciting possibilities for AI in gaming. We start with having the two top chess engines play against each other.

## Repo's Objectives
1. Learn about the theory behind AI in games and the taxanomy of algorithms
2. Run pre-built models
3. Learn about training such models using different algorithms

## Why?
- AI benchmarking
- Challenge Human Players
- Simulation-Based Testing: Test if the game is playable.
- NPC Creation: Avoid highly efficient, yet predictable and boring play style. Acadamics have argued for Turing test for NPCs.

## Games Classification
Games can be classified into many different dimensions based on
 - Observability
 - Stochasticity
 - Time Granularity
 
<img width="458" alt="Games I" src="https://github.com/user-attachments/assets/6d388b66-1e0b-4657-9d17-e4603e21968a">

## Gameplay Algorithms
There are three main ways for gameplay: Planning-based approaches, Reinforcement Learning (RL), and Surpervised Learning. Most agents employ one or a composite of these approaches.

Planning approaches include tree-search (both classical and stochastic), evolutionary planning, and symbolic representation. RL algorithms can have different categorization based on employing a model and optimizing a value function or a policy. The most successful RL algorithms in game-play are generally model-free policy-based although they're sample inefficient and unstable. Supervised learning algorithms include behavioral cloning, which is learning actions from experienced players.

The choice of an algorithm or a category of algorithms usually rely on
- [Game's Nature](#Games-Classification)
- Action Space: Branching factor and Game State Representation
- Time an agent has before making a decision
- Presence of source code

### Branching Factor
```mermaid
xychart-beta
    title "Chess"
    x-axis "turn" [1, 10, 20, 30, 40, 50, 60, 70]
    y-axis "Branching Factor" 1 --> 50
    line [20, 20, 35,  35, 35, 35, 15, 15]
```
This graph is based on averages I found online. A better estimate is to analyze most famous games in the pgn format and get more concrete averages or simulate games between computer players and see the number of valid moves at every turn so the graph is more realistic. The branching factor of chess dwarfs when compared to Go!

| Game       | Average Branching Factor |
|------------|-----------------------------
| Chess      | 33                       |
| Go         | 250                      |
| Pac-Man    | 4                        |
| Checkers   | 8                        |

Searching a game tree of depth `d` and branching factor `b` is $O(b^d)$.

### Action Space
The bigger the action space, the longer it takes for a policy (approximated using a neural network) to stabilize. To counter the time requirement and sample inefficiency, `macro-actions` and different/sampled `game-state representation` could be used.

### Game State Represenatation and Source Code
Sometimes, the source code is present so an API exists (or could exist) to provide a richer state reprentation while other times, when source code is not present, raw pixels are used for example.

### More Considerations
- Training Time
- Transferability of the trained policy from one game to another

### Classical/Stochastic Tree Search
Usually applied to games that feature full observability.

### Evolutionary Planning
A plan is a sequence on which mutation and crossover are applied.

### Reinforcement Learning (Classical, Deep, Evolutionary)
Applicable to games when there's sufficient learning time. Since Q-values (state, action) couldn't be stored for billions of states for video games, academics used funcion approximators, thus neural networks were introduced.

#### Catastrophic Forgetting
Catastrophic forgetting, also known as catastrophic interference, is a phenomenon observed in artificial neural networks where the model abruptly and drastically forgets previously learned information upon learning new information.

#### Experience Replay
Experience replay is a technique used to mitigate the problem of catastrophic forgetting in reinforcement learning. It involves storing the agent's past experiences in a replay buffer and then randomly sampling from this buffer to train the model.


### Suprevised Learning
Function approximators that behave like a skilled player.

### Comparison

| Algorithm Family       | Pros | Cons |
|------------------------|------|------|
| Tree Search            |   Strong for games that require adversarial planning. | Sometimes can be very expensive to go deeper depending on branching factor. Also, don't work well for hidden information games. |
| Evolutionary Planning  |      |      |
| RL                     | Strong with games that require perception, motor skills, and continuous movement. | Game state representation dictates how cheap or expensive training and evaluation is. Also, subject to the pifalls of function approximators. |
| Supervised Learning    |      |      |

### Classical Games
- Atari 2600
- Nintendo NES
- Commodore 64
- ZX Spectrum


### Video Game Gameplay Conception
- [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [Microsoft vs Ms. Pac-Man](https://blogs.microsoft.com/ai/divide-conquer-microsoft-researchers-used-ai-master-ms-pac-man/)
- [Berkeley's Pac-Man Projects](http://ai.berkeley.edu/project_overview)


### Regret and Card Games
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

## Glossary
- [NPC](https://en.wikipedia.org/wiki/Non-player_character): A non-player character (NPC) is a character in a game that is not controlled by a player. 
- [Game Tree](https://en.wikipedia.org/wiki/Game_tree)
- [Minimax](https://en.wikipedia.org/wiki/Minimax): Assumes no collusion from multiple players for more than two players.
- [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Monte Carlo Tree Search](https://github.com/shehio/monte-carlo-tree-search)
- [Branching Factor](https://en.wikipedia.org/wiki/Branching_factor)

## References
- [Artificial Intelligence and Games](https://gameaibook.org/book.pdf)
