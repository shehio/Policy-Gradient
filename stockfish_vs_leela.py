from stockfish import Stockfish
from lc0_client import Lc0Client
import chess

stockfish = Stockfish(path="./Stockfish/src/stockfish")
stockfish.set_depth(5)

lc0 = Lc0Client(lc0_path='./lc0/build/lc0', weights_path='./lc0/build/t1-512x15x8h-distilled-swa-3395000.pb.gz')
lc0.start_engine()

import chess

def check_game_status(board):
    if board.is_checkmate():
        print("Game over: Checkmate")
        if board.turn == chess.WHITE:
            print("Black won!")
        else:
            print("White won!")
        return True
    if board.is_stalemate():
        print("Game over: Stalemate")
        return True
    if board.is_insufficient_material():
        print("Game over: Insufficient material")
        return True
    if board.is_seventyfive_moves():
        print("Game over: Seventy-five move rule")
        return True
    if board.is_fivefold_repetition():
        print("Game over: Fivefold repetition")
        return True
    if board.can_claim_fifty_moves():
        print("Game over: Fifty-move rule")
        return True
    if board.can_claim_threefold_repetition():
        print("Game over: Threefold repetition")
        return True
    return False


if __name__ == "__main__":
    moves = []

    board = chess.Board()

    while True:
        stockfish.set_fen_position(board.fen())
        white_move = stockfish.get_best_move()
        board.push_san(white_move)
        moves.append(white_move)
        
        if check_game_status(board):
            break

        lc0.set_fen_position(board.fen())
        black_move = lc0.get_best_response(nodes=5)
        board.push_san(black_move)
        moves.append(black_move)

        print(board.fen())

        if check_game_status(board):
            break
    
    print(board)
    print(board.move_stack)
    
lc0.stop_engine()
exit(0)