from stockfish import Stockfish


positions = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "expected_evaluation": {"type": "cp", "value": 107},
    },
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "expected_evaluation": {"type": "cp", "value": 31},
    },
    {
        "fen": "r1bqkbnr/pppppppp/n7/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
        "expected_evaluation": {"type": "cp", "value": 16},
    },
]


stockfish = Stockfish(path="./Stockfish/src/stockfish")
print(
    f"The current major version of stockfish is: {stockfish.get_stockfish_major_version()}\n"
)

depth = 30
stockfish.set_depth(depth)

if __name__ == "__main__":
    for position in positions:
        fen = position["fen"]

        stockfish.set_fen_position(fen)
        actual_evaluation = stockfish.get_evaluation()
        best_move = stockfish.get_best_move()

        print(stockfish.get_board_visual())
        print(f"The current best move is {best_move}.\n")

        expected_evaluation = position["expected_evaluation"]
        if actual_evaluation == expected_evaluation:
            print("The evaluation matches the known benchmark.\n")
        else:
            print("The evaluation does not match the known benchmark.\n")
