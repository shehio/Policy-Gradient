class HyperParameters:
    def __init__(self, learning_rate: float, decay_rate: float, gamma: float, batch_size: int, save_interval: int) -> None:
        self.learning_rate: float = learning_rate
        self.decay_rate: float = decay_rate
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.save_interval: int = save_interval 