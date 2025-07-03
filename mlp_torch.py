import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, input_count: int, hidden_layers_count: int, output_count: int, network_file: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_count, hidden_layers_count)
        self.fc2 = nn.Linear(hidden_layers_count, output_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4, alpha=0.99)
        self.gradient_buffer = []
        self.saved_log_probs = []
        self.rewards = []
        self.network_file = network_file

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward_pass(self, input: np.ndarray) -> Tuple[float, np.ndarray]:
        input_tensor = torch.from_numpy(input).float()
        output = self.forward(input_tensor)
        hidden_layer = self.relu(self.fc1(input_tensor)).detach().numpy()
        return output.item(), hidden_layer

    def backward_pass(self, eph: np.ndarray, epdlogp: np.ndarray, epx: np.ndarray) -> None:
        # Store for batch update in train()
        self.gradient_buffer.append((epx, eph, epdlogp))

    def train(self, learning_rate: float, decay_rate: float) -> None:
        self.optimizer.param_groups[0]['lr'] = learning_rate
        self.optimizer.param_groups[0]['alpha'] = decay_rate
        self.optimizer.zero_grad()
        for epx, eph, epdlogp in self.gradient_buffer:
            inputs = torch.from_numpy(epx).float()
            hidden = torch.from_numpy(eph).float()
            advantages = torch.from_numpy(epdlogp).float()
            output = self.forward(inputs)
            loss = -torch.sum(torch.log(output) * advantages)
            loss.backward()
        self.optimizer.step()
        self.gradient_buffer = []

    def load_network(self, episode_number: int) -> None:
        if episode_number > 0:
            file_name = self.network_file + str(episode_number)
            print("Loading network from file: ", file_name)
            state_dict = torch.load(file_name, map_location='cpu')
            self.load_state_dict(state_dict)

    def save_network(self, episode_number: int) -> None:
        file_name = self.network_file + str(episode_number)
        print("Saving network to file: ", file_name)
        torch.save(self.state_dict(), file_name)
