import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import time
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, input_count: int, hidden_layers_count: int, output_count: int, network_file: str) -> None:
        super().__init__()
        
        # Store model parameters for unique file naming
        self.input_count = input_count
        self.hidden_layers_count = hidden_layers_count
        self.output_count = output_count
        
        # Device selection with detailed logging
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"MLP initialized on device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Network architecture
        self.fc1 = nn.Linear(input_count, hidden_layers_count)
        self.fc2 = nn.Linear(hidden_layers_count, output_count)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4, alpha=0.99)
        self.gradient_buffer = []
        self.saved_log_probs = []
        self.rewards = []
        self.network_file = network_file
        self.to(self.device)
        
        # Performance tracking
        self.forward_times = []
        self.backward_times = []
        self.episode_count = 0
        
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Model size: {sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6:.2f} MB")
        print(f"Architecture: {input_count}->{hidden_layers_count}->{output_count}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def forward_pass(self, input: np.ndarray) -> Tuple[float, np.ndarray]:
        start_time = time.time()
        

        input_tensor = torch.from_numpy(input).float().to(self.device)
        with torch.no_grad():  # No gradients needed for inference
            output = self.forward(input_tensor)
            hidden_layer = self.relu(self.fc1(input_tensor))
        
        # Move back to CPU for environment interaction
        output_cpu = output.cpu().numpy()
        hidden_cpu = hidden_layer.cpu().numpy()
        
        # Track performance
        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        
        return output_cpu.item(), hidden_cpu

    def backward_pass(self, eph: np.ndarray, epdlogp: np.ndarray, epx: np.ndarray) -> None:
        # Store for batch update in train()
        self.gradient_buffer.append((epx, eph, epdlogp))

    def train(self, learning_rate: float, decay_rate: float) -> None:
        if not self.gradient_buffer:
            return
            
        start_time = time.time()
        
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
        
        output = self.forward(inputs)
        loss = -torch.sum(torch.log(output) * advantages)
        loss.backward()
        self.optimizer.step()
    

        self.gradient_buffer = []
        train_time = time.time() - start_time
        self.backward_times.append(train_time)
        
        # Increment episode count
        self.episode_count += 1
        
        # Log performance every 1000 episodes
        if self.episode_count % 1000 == 0:
            avg_time = np.mean(self.backward_times[-100:]) if self.backward_times else 0
            print(f"Episode {self.episode_count}: Average training step time: {avg_time*1000:.2f} ms")
            print(f"Episode {self.episode_count}: Loss: {loss.item():.6f}")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                print(f"Episode {self.episode_count}: GPU memory used: {gpu_memory:.1f} MB")

    def load_network(self, episode_number: int) -> None:
        if episode_number > 0:
            base_name = os.path.splitext(self.network_file)[0]
            file_name = f"{base_name}_i{self.input_count}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"
            print("Loading network from file: ", file_name)
            
            if os.path.exists(file_name):
                state_dict = torch.load(file_name, map_location=self.device)
                self.load_state_dict(state_dict)
                print(f"Successfully loaded model with architecture {self.input_count}->{self.hidden_layers_count}->{self.output_count}")
            else:
                print(f"Warning: Network file {file_name} not found. Starting with random weights.")
                print(f"Expected architecture: {self.input_count}->{self.hidden_layers_count}->{self.output_count}")

    def save_network(self, episode_number: int) -> None:
        base_name = os.path.splitext(self.network_file)[0] 
        file_name = f"{base_name}_i{self.input_count}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"
        

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        print("Saving network to file: ", file_name)
        print(f"Architecture: {self.input_count}->{self.hidden_layers_count}->{self.output_count}")
        torch.save(self.state_dict(), file_name)

    