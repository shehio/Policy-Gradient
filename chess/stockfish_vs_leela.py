from collections import defaultdict
import chess
from stockfish import Stockfish
from chess.lc0_client import Lc0Client


stockfish = Stockfish(path="./Stockfish/src/stockfish")
stockfish.set_depth(10)

lc0 = Lc0Client(lc0_path='./lc0/build/lc0', weights_path='./lc0/build/t1-512x15x8h-distilled-swa-3395000.pb.gz')
lc0.start_engine()


def check_game_status(board):
    if board.is_checkmate():
        print()
        if board.turn == chess.WHITE:
            print("Game over: Checkmate: Black won!")
        else:
            print("Game over: Checkmate: White won!")
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

def play_move(agent, board, moves):
    agent.set_fen_position(board.fen())
    move = agent.get_best_move()
    board.push_san(move)
    moves.append(move)

def get_agent(turn, agent0, agent1):
    if turn == 0:
        return agent0
    return agent1

if __name__ == "__main__":
    moves = []
    wins = defaultdict(int)

    agent0, agent1 = stockfish, lc0
    games = 10
    verbose = False

    for i in range(games):
        board = chess.Board()
        turn = 0

        while True:
            agent = get_agent(turn, agent0, agent1)
            play_move(agent, board, moves)

            if verbose:
                print(board.fen())

            if check_game_status(board):
                break

            turn = 1 - turn
        
        print(board)
        print('\n')

        if board.is_checkmate():
            if board.turn:
                wins[agent1] += 1
            else:
                wins[agent0] += 1
        else:
            wins['neither'] += 1
    
    print(wins)

lc0.stop_engine()
exit(0)