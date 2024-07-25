class State:
    def __init__(self, source: set, destination: set) -> None:
        self.source = frozenset(source)
        self.destination = frozenset(destination)
    
    def is_terminal_state(self) -> bool:
        return not self.source

    def is_valid(self) -> bool:
        return self.__check_side(self.source) and self.__check_side(self.destination)
    

    def move_entity(self, source, destination, entities):
        new_source = set(source)
        new_destination = set(destination)
        for entity in entities:
            new_source.remove(entity)
            new_destination.add(entity)
        return State(new_source, new_destination)

    def next_states(self) -> list:
        result = []

        if 'boat' in self.destination:
            source, destination = self.destination, self.source
        else:
            source, destination = self.source, self.destination

        # Move boat with one person
        for person in source:
            if person != 'boat':
                result.append(self.move_entity(source, destination, ['boat', person]))

        # Move boat with two people
        for first_person in source:
            if first_person != 'boat':
                for second_person in source:
                    if second_person != 'boat' and first_person != second_person:
                        result.append(self.move_entity(source, destination, ['boat', first_person, second_person]))

        return result
    
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
