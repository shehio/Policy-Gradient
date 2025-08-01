class ExplorationConfig:
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_minimum: float = 0.05,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minimum
