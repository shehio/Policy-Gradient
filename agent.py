from typing import Any
import numpy as np
from hyperparameters import HyperParameters
from memory import Memory

DOWN = 2
UP = 3


class Agent:
    def __init__(self, hyperparams: HyperParameters, policy_network: Any, load_network: bool = True, network_file: str = 'save.p') -> None:
        self.memory = Memory()
        self.hyperparams = hyperparams
        self.policy_network = policy_network

        if load_network:
            print("Loading Network")
            self.policy_network.load_network(network_file)

    def sample_and_record_action(self, state: np.ndarray) -> int:
        # forward the policy network and sample an action from the returned probability
        action_probability_space, hidden_layer = self.policy_network.forward_pass(state)

        if np.random.uniform() < action_probability_space: # roll the dice!
            action = DOWN
            y = 1 # a "fake label"
        else:
            action = UP
            y = 0

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
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        episode_states = np.vstack(self.memory.states)
        episode_hidden_layers = np.vstack(self.memory.hidden_layers)
        episode_dlogps = np.vstack(self.memory.dlogps)
        episode_rewards = np.vstack(self.memory.rewards)

        # compute the discounted reward backwards through time
        episode_discounted_rewards = self.__discount_and_normalize_rewards(episode_rewards, self.hyperparams.gamma)
        episode_dlogps *= episode_discounted_rewards  # modulate the gradient with advantage (PG magic happens right here.)
        self.policy_network.backward_pass(episode_hidden_layers, episode_dlogps, episode_states)

    def __train_policy_network(self, episode_number: int) -> None:
        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.hyperparams.batch_size == 0:
            self.policy_network.train(self.hyperparams.learning_rate, self.hyperparams.decay_rate)

    def __save_policy_network(self, episode_number: int) -> None:
        if episode_number % self.hyperparams.save_interval == 0:
            self.policy_network.save_network()

    def __discount_and_normalize_rewards(self, r, gamma):
        discounted_rewards = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_rewards[t] = running_add

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards
