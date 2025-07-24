import chess.pgn
from stockfish import Stockfish

stockfish = Stockfish(path="./Stockfish/src/stockfish")
stockfish.set_depth(20)

pgn = open("famous-games/byrne_fischer_1956.pgn")
game = chess.pgn.read_game(pgn)

board = game.board()
for move in game.mainline_moves():
    board.push(move)
    stockfish.set_fen_position(board.fen())
    print(board.fen())
    print(f"Move: {move}, Evaluation: {stockfish.get_evaluation()}")

pgn.close()
