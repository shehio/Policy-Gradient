class State:
    def __init__(self, source: set, destination: set) -> None:
        self.source = frozenset(source)
        self.destination = frozenset(destination)
    
    def is_terminal_state(self) -> bool:
        return not self.source

    def is_valid(self) -> bool:
        return self.__check_side(self.source) and self.__check_side(self.destination)

    def __check_side(self, side):
        husbands = {person[0] for person in side if person.endswith('H')}
        wives = {person[0] for person in side if person.endswith('W')}
        for wife in wives:
            # wife is without her husband and there are other husbands
            if wife not in husbands and husbands:
                return False
        return True
    
    def __str__(self) -> str:
        return '\nsource: ' + ' '.join(self.source) +  '\ndestination: ' + ' '.join(self.destination)
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.source == other.source and self.destination == other.destination

    def __hash__(self):
        return hash((self.source, self.destination))
    
