import torch
from .environment_config import EnvironmentConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .learning_config import LearningConfig
from .exploration_config import ExplorationConfig
from .image_config import ImageConfig


class HyperParameters:
    def __init__(self):
        self.environment = EnvironmentConfig(
            environment="ALE/Pong-v5",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            render_game_window=False
        )
        
        self.model = ModelConfig(
            save_models=True,
            model_path="./models/pong-cnn-",
            save_model_interval=10,
            train_model=True,
            load_model_from_file=False,
            load_file_episode=0
        )
        
        self.training = TrainingConfig(
            batch_size=64,
            max_episode=100000,
            max_step=100000,
            max_memory_len=50000,
            min_memory_len=40000
        )
        
        self.learning = LearningConfig(
            gamma=0.97,
            alpha=0.00025
        )
        
        self.exploration = ExplorationConfig(
            epsilon_start=1.0,
            epsilon_decay=0.999,
            epsilon_minimum=0.05
        )
        
        self.image = ImageConfig(
            target_h=80,
            target_w=64,
            crop_top=20
        ) 