class State:
    def __init__(self, source: set, destination: set) -> None:
        self.source = frozenset(source)
        self.destination = frozenset(destination)
    
    def is_terminal_state(self) -> bool:
        return not self.source

    def is_valid(self) -> bool:
        return self.__check_side(self.source) and self.__check_side(self.destination)
