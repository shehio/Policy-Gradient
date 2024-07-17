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

    def set_fen_position(self, fen):
        self.__send_command(f'position fen {fen}')

    def go(self, nodes=100):
        self.__send_command(f'go nodes {nodes}', verbose=False)
        return self.__read_response()
    
    def get_best_move(self, nodes=10):
        response = self.go(nodes)
        for line in response:
            if line.startswith('bestmove'):
                lc0_move = line.split()[1]
                break
        return lc0_move

    def stop_engine(self):
        self.__send_command('quit')
        self.process.terminate()

    def __send_command(self, command, verbose=False):
        if verbose:
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
