class ModelConfig:
    def __init__(self, save_models: bool = True,
                 model_path: str = "./arcade/scripts/dqn/models/pong-cnn-",
                 save_model_interval: int = 10,
                 train_model: bool = True,
                 load_model_from_file: bool = True,
                 load_file_episode: int = 0):
        self.save_models = save_models
        self.model_path = model_path
        self.save_model_interval = save_model_interval
        self.train_model = train_model
        self.load_model_from_file = load_model_from_file
        self.load_file_episode = load_file_episode 