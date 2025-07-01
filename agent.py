import numpy as np
from helpers import Helpers
from memory import Memory
from hyperparameters import HyperParameters

DOWN = 2
UP = 3


class Agent:
    def __init__(self, hyperparams, policy_network, load_network=True, network_file='save.p'):
        self.memory = Memory()
        self.hyperparams = hyperparams
        self.policy_network = policy_network

        if load_network:
            print("Loading Network")
            self.policy_network.load_network(network_file)

    def sample_and_record_action(self, state):
        # forward the policy network and sample an action from the returned probability
        action_probability_space, hidden_layer = self.policy_network.forward_pass(state)
        action = DOWN if np.random.uniform() < action_probability_space else UP  # roll the dice!
        y = 1 if action == DOWN else 0  # a "fake label"

        self.memory.states.append(state)
        self.memory.hidden_layers.append(hidden_layer)

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        self.memory.dlogps.append(y - action_probability_space)

        return action

    def reap_reward(self, reward):
        self.memory.rewards.append(reward)

    def make_episode_end_updates(self, episode_number):
        print(self.memory)
        self.__accumalate_gradient()
        self.__train_policy_network(episode_number)
        self.__save_policy_network(episode_number)

        memory = Memory()

    def __accumalate_gradient(self):
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        episode_states = np.vstack(self.memory.states)
        episode_hidden_layers = np.vstack(self.memory.hidden_layers)
        epdlogp = np.vstack(self.memory.dlogps)
        episode_rewards = np.vstack(self.memory.rewards)

        # compute the discounted reward backwards through time
        episode_discounted_rewards = Helpers.discount_and_normalize_rewards(episode_rewards, self.hyperparams.gamma)
        epdlogp *= episode_discounted_rewards  # modulate the gradient with advantage (PG magic happens right here.)
        self.policy_network.backward_pass(episode_hidden_layers, epdlogp, episode_states)

    def __train_policy_network(self, episode_number):
        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.hyperparams.batch_size == 0:
            self.policy_network.train(self.hyperparams.learning_rate, self.hyperparams.decay_rate)

    def __save_policy_network(self, episode_number):
        if episode_number % self.hyperparams.save_interval == 0:
            self.policy_network.save_network()
