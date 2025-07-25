import torch
from typing import Optional


class EnvironmentConfig:
    def __init__(self, environment: str = "ALE/Pong-v5", 
                 device: Optional[torch.device] = None,
                 render_game_window: bool = False):
        self.environment = environment
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_game_window = render_game_window 