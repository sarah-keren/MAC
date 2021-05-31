import numpy as np


class PolicyGradient(object):

    def __init__(self, num_actions, theta, num_agents=1, alpha=0.00025, gamma=0.9, mapping_fn=None):
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

        """ Record keeping / episode """
        self.grads = []
        self.rewards = []

        self.mapping_fn = mapping_fn
        self.num_actions = num_actions
        self.num_agents = num_agents

    def softmax(self, state):
        z = state.dot(self.theta)
        exp = np.exp(z)
        return exp / np.sum(exp)

    def policy(self, state):
        probs = self.softmax(state)
        return probs

    def softmax_gradient(self, softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def compute_gradient(self, probs, state, action):
        dsoftmax_comp = self.softmax_gradient(probs)
        dsoftmax = dsoftmax_comp[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None, :])

        self.grads.append(grad)
        return

    def update_weights(self):
        for i in range(len(self.grads)):
            present_val_of_rewards = sum([r * (self.gamma ** (t - i)) \
                                          for t, r in enumerate(self.rewards[i:])])
            self.theta += self.alpha * self.grads[i] * present_val_of_rewards * (self.gamma ** i)
        return

    """ Training Callbacks """

    def get_train_action(self, state):
        if self.mapping_fn:
            state = self.mapping_fn(state)

        if self.mapping_fn:
            state = self.mapping_fn(state)

        state = state[None, :]
        probs = self.policy(state)
        action = np.random.choice(self.num_actions, p=probs[0])
        self.compute_gradient(probs, state, action)
        return action

    def update_step(self, obs, action, new_obs, reward, done):
        self.rewards.append(reward)

    def update_episode(self,):
        """ Update weights, reset records for new episodes"""
        self.update_weights()

        self.grads = []
        self.rewards = []

    """ Evaluation callbacks """

    def get_action(self, state):
        """ Take an action according to policy."""
        if self.mapping_fn:
            state = self.mapping_fn(state)

        state = state[None, :]
        probs = self.policy(state)
        action = np.random.choice(self.num_actions, p=probs[0])
        return action
