import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd


class DeepPolicyGradient(object):
    def __init__(self, input_dims, num_actions, is_conv=False, gamma=0.99, learning_rate=0.01, mapping_fn=None):
        self.input_dims = input_dims[::-1]
        self.gamma = gamma
        self.reward_mem = []
        self.action_mem = []
        self.num_actions = num_actions

        if not is_conv:
            self.policy = PolicyNetwork(self.input_dims, num_actions, mapping_fn)
        else:
            self.policy = ConvPolicyNetwork(self.input_dims, num_actions)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def learn(self):
        self.optimizer.zero_grad()
        G = np.zeros_like(self.reward_mem, dtype=np.float64)

        for t in range(len(self.reward_mem)):
            G_sum = 0
            discount = 1

            for k in range(t, len(self.reward_mem)):
                G_sum += self.reward_mem[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std
        G = torch.tensor(G, dtype=torch.float).to(self.policy.device)
        loss = 0

        for g, logprob in zip(G, self.action_mem):
            loss += -g * logprob

        loss.backward()
        self.optimizer.step()

    """ Training Callbacks """

    def get_train_action(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)
        returned_action = action.item()
        return returned_action

    def update_step(self, obs, action, new_obs, reward, done):
        self.reward_mem.append(reward)

    def update_episode(self):
        self.learn()
        self.reward_mem = []
        self.action_mem = []

    """ Evaluation Callbacks """

    def get_action(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        return action.item()


class PolicyNetwork(nn.Module):

    def __init__(self, input_dims, output_dims, mapping_fn=None):
        super(PolicyNetwork, self).__init__()

        self.input_dim = input_dims
        self.output_dim = output_dims
        self.mapping_fn = mapping_fn
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, observation):
        if self.mapping_fn:
            observation = self.mapping_fn(observation)

        state = torch.Tensor(observation).to(self.device)
        x = self.fc(state)
        return x


class ConvPolicyNetwork(nn.Module):

    def __init__(self, input_dims, output_dims):
        super(ConvPolicyNetwork, self).__init__()
        self.input_dim = input_dims
        self.output_dim = output_dims
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, observation):
        observation = torch.Tensor(observation.copy()).unsqueeze(0).permute(0,3,1,2).to(self.device)
        features = self.conv(observation)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


