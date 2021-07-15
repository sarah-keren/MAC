from collections import deque
import random
import numpy as np
import torch as torch
import torch.nn as nn
from torch import autograd
from MAC.agents.agent import DecisionMaker



"""A Deep Q-Network algorithm for RL
"""
class DQN(DecisionMaker):

    def __init__(self, input_dims, num_actions, is_conv=False, learning_rate=3e-4, gamma=0.99,
                 buffer_size=10000, mapping_fn=None):
        super().__init__()
        self.input_dims = input_dims[::-1]
        self.num_actions = num_actions
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if is_conv:
            self.model = ConvDQNetwork(self.input_dims, self.num_actions).to(self.device)
        else:
            self.model = DQNetwork(self.input_dims, self.num_actions, mapping_fn).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = nn.MSELoss()

    def get_action(self, observation):
        """Get an action according the network"""
        observation = autograd.Variable(torch.from_numpy(observation.copy()).float().unsqueeze(0))
        qvals = self.model.forward(observation)
        action = np.argmax(qvals.cpu().detach().numpy())

        return action

    def get_train_action(self, observation, eps=0.1):
        """Get action for when training"""
        observation = autograd.Variable(torch.from_numpy(observation.copy()).float().unsqueeze(0))
        qvals = self.model.forward(observation)
        action = np.argmax(qvals.cpu().detach().numpy())

        if np.random.randn() < eps:
            return random.randint(0, self.num_actions - 1)

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q.detach())
        return loss

    def update_step(self, obs, action,new_obs, reward, done):
        """Add the step to the replay buffer"""
        self.replay_buffer.push(obs, action, new_obs, reward, done)

    def update_episode(self, batch_size):
        """Updating the Network"""
        if len(self.replay_buffer) > batch_size:
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


"""Class for the fully-connected Q-Network"""
class DQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, mapping_fn=None):
        super(DQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mapping_fn = mapping_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        qvals = self.fc(observation)
        return qvals


"""Class for the convolution Q-Network"""
class ConvDQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        observation = observation.permute(0, 3, 1, 2).to(self.device)
        features = self.conv(observation)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

"""Class for the Replay Buffer"""
class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)