import torch
from .environment_config import EnvironmentConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .learning_config import LearningConfig
from .exploration_config import ExplorationConfig
from .image_config import ImageConfig


class HyperParameters:
    def __init__(self, environment_config: EnvironmentConfig, model_config: ModelConfig, training_config: TrainingConfig, learning_config: LearningConfig, exploration_config: ExplorationConfig, image_config: ImageConfig):
        self.environment = environment_config
        self.model = model_config
        self.training = training_config
        self.learning = learning_config
        self.exploration = exploration_config
        self.image = image_config