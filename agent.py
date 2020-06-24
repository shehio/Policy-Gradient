import numpy as np
from helpers import Helpers
from helpers import MLP
from memory import Memory

DOWN = 2
UP = 3


class Agent:
    # TODO: Dep-Inject the Network to the Agent.
    def __init__(self, learning_rate, decay_rate, gamma=0.99, batch_size=5, load_network=True, network_file='save.p'):
        self.memory = Memory()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_network = MLP(input_count=6400, hidden_layers_count=200)

        if load_network:
            self.policy_network.load_network(network_file)

    def get_action(self, state):
        # forward the policy network and sample an action from the returned probability
        action_probability_space, hidden_layer = self.policy_network.forward_pass(state)
        action = DOWN if np.random.uniform() < action_probability_space else UP  # roll the dice!
        # record various intermediates (needed later for back-prop)
        self.memory.states.append(state)
        self.memory.hidden_layers.append(hidden_layer)
        y = 1 if action == 2 else 0  # a "fake label"

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        self.memory.dlogps.append(y - action_probability_space)

        return action

    def episode_updates(self, episode_number):
        memory = Memory()

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.batch_size == 0:
            self.policy_network.train(self.learning_rate, self.decay_rate)

        if episode_number % 100 == 0:
            self.policy_network.save_network()

    def modify_gradient(self):
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        episode_states = np.vstack(self.memory.states)
        episode_hidden_layers = np.vstack(self.memory.hidden_layers)
        epdlogp = np.vstack(self.memory.dlogps)
        episode_rewards = np.vstack(self.memory.rewards)

        # compute the discounted reward backwards through time
        episode_discounted_rewards = Helpers.discount_and_normalize_rewards(episode_rewards, self.gamma)
        epdlogp *= episode_discounted_rewards  # modulate the gradient with advantage (PG magic happens right here.)
        self.policy_network.backward_pass(episode_hidden_layers, epdlogp, episode_states)