class State:
    def __init__(self, source: set, destination: set) -> None:
        self.source = frozenset(source)
        self.destination = frozenset(destination)
    
    def is_terminal_state(self) -> bool:
        return not self.source
