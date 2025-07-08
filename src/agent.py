from typing import Any
import numpy as np
from hyperparameters import HyperParameters
from memory import Memory

DOWN = 2
UP = 3


class Agent:
    def __init__(self, policy_network: Any, hyperparams: HyperParameters) -> None:
        self.memory = Memory()
        self.hyperparams = hyperparams
        self.policy_network = policy_network

    def sample_action(self, state: np.ndarray) -> int:
        action_probability_space, _ = self.policy_network.forward_pass(state)
        action = 2 if np.random.uniform() < action_probability_space else 3  # DOWN or UP
        return action

    def sample_and_record_action(self, state: np.ndarray) -> int:
        action_probability_space, hidden_layer = self.policy_network.forward_pass(state)
        action = self.sample_action(state)
        y = 1 if action == 2 else 0
        self.memory.states.append(state)
        self.memory.hidden_layers.append(hidden_layer)

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        self.memory.dlogps.append(y - action_probability_space)

        return action

    def reap_reward(self, reward: float) -> None:
        self.memory.rewards.append(reward)

    def make_episode_end_updates(self, episode_number: int) -> None:
        print(self.memory)
        self.__accumalate_gradient()
        self.__train_policy_network(episode_number)
        self.__save_policy_network(episode_number)

        self.memory = Memory()

    def __accumalate_gradient(self) -> None:
        # Check if there are any rewards - if not, skip training entirely
        if np.sum(self.memory.rewards) == 0:
            print("⚠️  WARNING: No rewards received - skipping training (no learning signal)")
            return

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        episode_states = np.vstack(self.memory.states)
        episode_hidden_layers = np.vstack(self.memory.hidden_layers)
        episode_dlogps = np.vstack(self.memory.dlogps)
        episode_rewards = np.vstack(self.memory.rewards)

        # compute the discounted reward backwards through time
        episode_discounted_rewards = self.__discount_and_normalize_rewards(episode_rewards, self.hyperparams.gamma)
        episode_dlogps *= episode_discounted_rewards  # modulate the gradient with advantage (PG magic happens right here.)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(episode_dlogps)) or np.any(np.isinf(episode_dlogps)):
            print("🚨 CRITICAL: NaN or infinite gradients detected! Skipping update.")
            return
        
        # Check for vanishing gradient
        gradient_magnitude = np.mean(np.abs(episode_dlogps))
        if gradient_magnitude < 1e-6:
            print("⚠️  WARNING: Very small gradients detected!")
        
        self.policy_network.backward_pass(episode_hidden_layers, episode_dlogps, episode_states)

    def __train_policy_network(self, episode_number: int) -> None:
        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.hyperparams.batch_size == 0:
            self.policy_network.train(self.hyperparams.learning_rate, self.hyperparams.decay_rate)

    def __save_policy_network(self, episode_number: int) -> None:
        if episode_number % self.hyperparams.save_interval == 0:
            self.policy_network.save_network(episode_number)

    def __discount_and_normalize_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        if np.std(discounted_rewards) > 1e-8:  # Avoid division by zero
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
        
        return discounted_rewards
