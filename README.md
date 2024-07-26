# Game AI
This is a repository for studying AI in games.
It investigates classical search methods, such as graph-based and tree-based methods, classical Reinforcement Learning methods with tabular Q-function, and more modern approaches relying on a policy network and temporal difference.
The plan is to significantly expand this repository, covering more challenging game states and action spaces and focusing on sample efficiency and computational resources, offering exciting possibilities for AI in gaming.
We start with having the two top chess engines play against each other.

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

### Glossary
- [NPC](https://en.wikipedia.org/wiki/Non-player_character): A non-player character (NPC) is a character in a game that is not controlled by a player. 

### References
- [Artificial Intelligence and Games](https://gameaibook.org/book.pdf)

## chessanalysis
### Why

### How
- Clone and build Stockfish and lc0: `./build_engines.sh`
- Activate Python virtual environment: `python3 -m venv venv && source venv/bin/activate`
- Install Pip requirements: `pip3 install -r requirements.txt`
- Test Stockfish integration `python3 test_stockfish.py`

### Further Reading
- [Fen Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation): A standard notation for describing a particular board position of a chess game.
- [Algebraic Notation](https://en.wikipedia.org/wiki/Algebraic_notation_(chess)): The standard method for recording and describing the moves in a game of chess. 
- [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface): An open communication protocol that enables chess engines to communicate with user interfaces.
- [Chess Engine](https://en.wikipedia.org/wiki/Chess_engine)
- [Stockfish](https://en.wikipedia.org/wiki/Stockfish_(chess)): A free and open-source chess engine, available for various desktop and mobile platforms. 
- [Leela Chess Zero](https://en.wikipedia.org/wiki/Leela_Chess_Zero): A free, open-source, and deep neural network–based chess engine and volunteer computing project.
- [Top Chess Engine Championship](https://en.wikipedia.org/wiki/Top_Chess_Engine_Championship): A computer chess tournament that has been run since 2010.
### Gameplay Algorithms
- [Game Tree](https://en.wikipedia.org/wiki/Game_tree)
- [Minimax](https://en.wikipedia.org/wiki/Minimax): Assumes no collusion from multiple players for more than two players.
- [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Monte Carlo Tree Search](https://github.com/shehio/monte-carlo-tree-search)

### Todos
