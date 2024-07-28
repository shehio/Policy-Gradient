# Game AI
This is a repository for studying AI in games. It investigates classical search methods, such as graph-based and tree-based methods, classical Reinforcement Learning methods with tabular Q-function, and more modern approaches relying on a policy network and temporal difference. The plan is to significantly expand this repository, covering more challenging game states and action spaces and focusing on sample efficiency and computational resources, offering exciting possibilities for AI in gaming. We start with having the two top chess engines play against each other.

## Why?
- AI benchmarking
- Challenge Human Players
- Simulation-Based Testing: Test if the game is playable.
- NPC Creation: Avoid highly efficient, yet predictable and boring play style.

## Games Classification
Games can be classified into many different dimensions based on
 - Observability
 - Stochasticity
 - Time Granularity
<img width="458" alt="Games I" src="https://github.com/user-attachments/assets/6d388b66-1e0b-4657-9d17-e4603e21968a">

## Turing Test
Turing test for NPCs.


### Gameplay Algorithms
There are three main ways for gameplay: Planning-based approaches, Reinforcement Learning (RL), and Surpervised learning. Most agents employ one or a composite of these approaches.
Planning approaches include tree-search (both classical and stochastic), evolutionary planning, and symbolic representation. RL algorithms can have different categorization based on employing a model and optimizing a value function or a policy. The most successful RL algorithms in game-play are generally model-free policy-based although they're sample inefficient and unstable.


### Glossary
- [NPC](https://en.wikipedia.org/wiki/Non-player_character): A non-player character (NPC) is a character in a game that is not controlled by a player. 
- [Game Tree](https://en.wikipedia.org/wiki/Game_tree)
- [Minimax](https://en.wikipedia.org/wiki/Minimax): Assumes no collusion from multiple players for more than two players.
- [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Monte Carlo Tree Search](https://github.com/shehio/monte-carlo-tree-search)

### References
- [Artificial Intelligence and Games](https://gameaibook.org/book.pdf)
