class TrainingConfig:
    def __init__(self, batch_size: int = 64,
                 max_episode: int = 100000,
                 max_step: int = 100000,
                 max_memory_len: int = 50000,
                 min_memory_len: int = 40000):
        self.batch_size = batch_size
        self.max_episode = max_episode
        self.max_step = max_step
        self.max_memory_len = max_memory_len
        self.min_memory_len = min_memory_len 