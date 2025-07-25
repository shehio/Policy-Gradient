import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, input_count: int, hidden_layers_count: int, output_count: int, network_file: str, game_name: str) -> None:
        super().__init__()
        
        # Store model parameters for unique file naming
        self.input_count = input_count
        self.hidden_layers_count = hidden_layers_count
        self.output_count = output_count
        self.game_name = game_name.replace("/", "_").replace("-", "_")  # todo(shehio): This shouldn't live here.
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"MLP initialized on device: {self.device}")
        
        # Network architecture
        self.fc1 = nn.Linear(input_count, hidden_layers_count)
        self.fc2 = nn.Linear(hidden_layers_count, output_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4, alpha=0.99)
        self.gradient_buffer = []
        self.network_file = network_file
        self.to(self.device)
        
        print(f"Model: {input_count}->{hidden_layers_count}->{output_count}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward_pass(self, input: np.ndarray) -> Tuple[float, np.ndarray]:
        input_tensor = torch.from_numpy(input).float().to(self.device)
        with torch.no_grad():  # No gradients needed for inference
            output = self.forward(input_tensor)
            hidden_layer = self.relu(self.fc1(input_tensor))
        
        # Move back to CPU for environment interaction
        output_cpu = output.cpu().numpy()
        hidden_cpu = hidden_layer.cpu().numpy()
        
        return output_cpu.item(), hidden_cpu

    def backward_pass(self, eph: np.ndarray, epdlogp: np.ndarray, epx: np.ndarray) -> None:
        # Store for batch update in train()
        self.gradient_buffer.append((epx, eph, epdlogp))

    def train(self, learning_rate: float, decay_rate: float) -> None:
        if not self.gradient_buffer:
            return
            
        # Update optimizer parameters
        self.optimizer.param_groups[0]['lr'] = learning_rate
        self.optimizer.param_groups[0]['alpha'] = decay_rate
        self.optimizer.zero_grad()
        
        # Batch all episodes together for efficient GPU processing
        all_inputs = []
        all_hidden = []
        all_advantages = []
        
        for epx, eph, epdlogp in self.gradient_buffer:
            all_inputs.append(epx)
            all_hidden.append(eph)
            all_advantages.append(epdlogp)
        
        # Convert to tensors and move to GPU in one batch
        inputs = torch.from_numpy(np.vstack(all_inputs)).float().to(self.device)
        hidden = torch.from_numpy(np.vstack(all_hidden)).float().to(self.device)
        advantages = torch.from_numpy(np.concatenate(all_advantages)).float().to(self.device)
        
        # Check for NaN or infinite values before training
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("🚨 CRITICAL: NaN or infinite inputs detected! Skipping training.")
            self.gradient_buffer = []
            return
            
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("🚨 CRITICAL: NaN or infinite advantages detected! Skipping training.")
            self.gradient_buffer = []
            return
        
        output = self.forward(inputs)
        
        # Clamp output to avoid log(0) or log(1)
        output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
        
        loss = -torch.sum(torch.log(output) * advantages)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 CRITICAL: NaN or infinite loss detected! Skipping training.")
            self.gradient_buffer = []
            return
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.gradient_buffer = []

    def load_network(self, episode_number: int) -> None:
        if episode_number > 0:
            base_name = os.path.splitext(self.network_file)[0]
            file_name = f"{base_name}_{self.game_name}_i{self.input_count}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"
            
            # Look in models directory
            models_dir = "atari/scripts/policy-gradient/models"
            file_path = os.path.join(models_dir, file_name)
            
            if os.path.exists(file_path):
                state_dict = torch.load(file_path, map_location=self.device)
                self.load_state_dict(state_dict)
                print(f"Loaded model: {file_path}")
            else:
                print(f"Model file not found: {file_path}")

    def save_network(self, episode_number: int) -> None:
        base_name = os.path.splitext(self.network_file)[0] 
        file_name = f"{base_name}_{self.game_name}_i{self.input_count}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"
        
        # Ensure the models directory exists
        models_dir = "atari/scripts/policy-gradient/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        # Save in models directory
        file_path = os.path.join(models_dir, file_name)
        torch.save(self.state_dict(), file_path)

    