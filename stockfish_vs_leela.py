from stockfish import Stockfish
from lc0_client import Lc0Client

stockfish = Stockfish(path="./Stockfish/src/stockfish")
stockfish.set_depth(5)

lc0 = Lc0Client(lc0_path='./lc0/build/lc0', weights_path='./lc0/build/t1-512x15x8h-distilled-swa-3395000.pb.gz')
lc0.start_engine()

if __name__ == "__main__":
    initial_fen_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stockfish.set_fen_position(initial_fen_position)
    moves = []

    while True:
        stockfish.set_position(moves)
        print(stockfish.get_board_visual())
        white_move = stockfish.get_best_move()

        if not white_move:
            break
        moves.append(white_move)

        lc0.set_position(moves)
        black_move = lc0.get_best_response(nodes=5)
        moves.append(black_move)

    print(stockfish.get_board_visual())    

lc0.stop_engine()
exit(0)