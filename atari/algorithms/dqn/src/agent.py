import random
import numpy as np
import torch
import torch.optim as optim
import cv2
from collections import deque
from .model import DuelCNN


class Agent:
    def __init__(self, environment, hyperparams):
        """
        Agent initialization using hyperparameters passed from training script
        """
        self.hyperparams = hyperparams

        # State size from environment
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Action size from environment
        self.action_size = environment.action_space.n

        # Image preprocessing parameters from hyperparams
        self.target_h = hyperparams.image.target_h  # Height after process
        self.target_w = hyperparams.image.target_w  # Width after process
        self.crop_top = hyperparams.image.crop_top  # Pixels to crop from top

        self.crop_dim = [self.crop_top, self.state_size_h, 0, self.state_size_w]

        # Learning parameters from hyperparams
        self.gamma = hyperparams.learning.gamma  # Discount coefficient
        self.alpha = hyperparams.learning.alpha  # Learning rate

        # Exploration parameters from hyperparams
        self.epsilon = hyperparams.exploration.epsilon_start  # Initial epsilon
        self.epsilon_decay = hyperparams.exploration.epsilon_decay  # Epsilon decay rate
        self.epsilon_minimum = (
            hyperparams.exploration.epsilon_minimum
        )  # Minimum epsilon

        # Memory parameters from hyperparams
        self.memory = deque(maxlen=hyperparams.training.max_memory_len)

        # Create models using hyperparams
        self.online_model = DuelCNN(
            h=self.target_h, w=self.target_w, output_size=self.action_size
        ).to(hyperparams.environment.device)
        self.target_model = DuelCNN(
            h=self.target_h, w=self.target_w, output_size=self.action_size
        ).to(hyperparams.environment.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # Optimizer using hyperparams
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        # Convert to numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB to grayscale
            elif image.shape[2] == 1:  # Already grayscale
                frame = image.squeeze()
            else:
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR to grayscale
        else:
            # Already 2D grayscale
            frame = image

        frame = frame[
            self.crop_dim[0] : self.crop_dim[1], self.crop_dim[2] : self.crop_dim[3]
        ]  # Cut from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def act(self, state):
        """
        Get state and do action
        Two options: explore (random) or exploit (neural network)
        """
        act_protocol = "Explore" if random.uniform(0, 1) <= self.epsilon else "Exploit"

        if act_protocol == "Explore":
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, dtype=torch.float, device=self.hyperparams.environment.device
                ).unsqueeze(0)
                q_values = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(
                    q_values
                ).item()  # Returns the indices of the maximum value

        return action

    def train(self):
        """
        Train neural nets with replay memory
        returns loss and max_q val predicted from online_net
        """
        if len(self.memory) < self.hyperparams.training.min_memory_len:
            loss, max_q = [0, 0]
            return loss, max_q

        # Sample minibatch from memory
        state, action, reward, next_state, done = zip(
            *random.sample(self.memory, self.hyperparams.training.batch_size)
        )

        # Concat batches in one array
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(
            state, dtype=torch.float, device=self.hyperparams.environment.device
        )
        next_state = torch.tensor(
            next_state, dtype=torch.float, device=self.hyperparams.environment.device
        )
        action = torch.tensor(
            action, dtype=torch.long, device=self.hyperparams.environment.device
        )
        reward = torch.tensor(
            reward, dtype=torch.float, device=self.hyperparams.environment.device
        )
        done = torch.tensor(
            done, dtype=torch.float, device=self.hyperparams.environment.device
        )

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Get index of max value from next_states_q_values
        # Use that index to get q_value from next_states_target_q_values
        next_states_target_q_value = next_states_target_q_values.gather(
            1, next_states_q_values.max(1)[1].unsqueeze(1)
        ).squeeze(1)

        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calculate loss
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon: decrease epsilon to do less exploration over time
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
