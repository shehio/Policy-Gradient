import subprocess
import time

class Lc0Client:
    def __init__(self, lc0_path, weights_path):
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.process = None

    def start_engine(self):
        print(f"Starting lc0 Engine!")
        self.process = subprocess.Popen(
            [self.lc0_path, f'--weights={self.weights_path}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        time.sleep(1)  # Give the engine time to start

    def set_position(self, moves):
        self.__send_command(f'position startpos moves {" ".join(moves)}')

    def go(self, nodes=100):
        self.__send_command(f'go nodes {nodes}')
        return self.__read_response()

    def stop_engine(self):
        self.__send_command('quit')
        self.process.terminate()

    def __send_command(self, command):
        print(f"Sending command: {command}")
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

    def __read_response(self):
        response = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == 'readyok' or line.startswith('bestmove'):
                response.append(line)
                break
            response.append(line)
        return response

if __name__ == "__main__":
    lc0_path = './lc0/build/lc0'
    weights_path = './lc0/build/t1-512x15x8h-distilled-swa-3395000.pb.gz'

    client = Lc0Client(lc0_path, weights_path)
    client.start_engine()

    moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
    client.set_position(moves)

    response = client.go(nodes=100)
    print(f"Engine response: {response}\n")

    client.stop_engine()