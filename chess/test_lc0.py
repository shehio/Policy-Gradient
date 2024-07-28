from chess.lc0_client import Lc0Client

if __name__ == "__main__":
    lc0_path = './lc0/build/lc0'
    weights_path = './lc0/build/t1-512x15x8h-distilled-swa-3395000.pb.gz'

    client = Lc0Client(lc0_path, weights_path)
    client.start_engine()

    moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
    client.set_position(moves)

    response = client.go(nodes=100)
    print(f"Engine response: {response}\n")

    client.set_fen_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    response = client.get_best_move(nodes=100)
    print(f"Engine response: {response}\n")

    client.stop_engine()
