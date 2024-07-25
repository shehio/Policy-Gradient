# Game AI
This is a repository to study AI in games. It investigates classical search methods, like graph-based and tree-based and classical Reinforcement Learning methods with tabular Q function and more modern approaches relying on a policy network.
The plan is to extend this repo to cover more of challenging game states and action spaces, focusing on sample efficiency and computation resources.
We start with having the two top chess engines play against each other.

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
- [Minimax](https://en.wikipedia.org/wiki/Minimax)
- [Alpha Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Monte Carlo Tree Search](https://github.com/shehio/monte-carlo-tree-search)

### Todos