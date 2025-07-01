class HyperParameters:
    def __init__(self, learning_rate, decay_rate, gamma=0.99, batch_size=5):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.batch_size = batch_size 