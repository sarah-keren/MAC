import numpy as np
from control import controller
from agents.RL-agents.deep_q import DQN_Solver
import control.utils
#from pandas.core.common import flatten as flatten_list_of_list

class centralized_dqn(controller):

    # rename decode, variables
    def __init__(self, agents, observations, env, learning_rate = 0.01, gamma= 0.95, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, model=None, batch_size=64, layer1_size=64, layer2_size=64):

        # initialize super class
        super(env, agents, observations)

        # initialize the DQN that combines all agents together
        input_dim = np.sum([agent.input_dims for agent in self.agents])
        action_dim = (self.env.num_actions)**(len(self.agents))
        self.central_agent = DQN_Solver(learning_rate, gamma, epsilon, epsilon_decay, epsilon_min, input_dim, action_dim, model, batch_size, layer1_size, layer2_size)

    def get_joint_action(self):
        cur_observations = []
        for index, agent in enumerate(self.agents):
            cur_observation = agent.get_observation(self.environment.state, index)
            cur_observations.append(cur_observation)

        state = control.utils._decode_state(cur_observations)
        joint_action = self.train_action(state)
        decoded_joint_action = control.utils._decode_action(joint_action, self.environment.num_actions,
                                                            self.environment.num_agents)
        return decoded_joint_action

    def perform_joint_action(self, joint_action):
        return self.environment.step(joint_action)

    def is_done(self):
        return False

    def train_action(self, state):
        state = self._decode_state(self.observations(state))
        action = self.central_agent.train_action(state)
        joint_action = self._decode_action(action)
        return joint_action

    def eval_action(self, state):
        state = self._decode_state(self.observations(state))
        action = self.central_agent.eval_action(state)
        joint_action = self._decode_action(action)
        return joint_action

    def remember(self, state, actions, rewards, next_state, done):
        state = self._decode_state(self.observations(state))
        next_state = self._decode_state(self.observations(next_state))
        print(state, next_state)
        reward = np.sum(rewards)
        action = self._encode_action(actions)
        self.central_agent.remember(state, action, reward, next_state, done)

    def experience_replay(self):
        self.central_agent.experience_replay()

    def save(self):
        pass

    def _encode_action(self, actions):
        action = 0
        for ind, act in enumerate(actions):
            action += (self.env.num_actions ** ind) * act
        return action

    def _decode_action(self, action):
        out = []
        for ind in range(self.env.num_agents):
            out.append(action % self.env.num_actions)
            action = action // self.env.num_actions
        return list(reversed(out))

    def _decode_state(self, state):
        return np.reshape(np.vstack(state), (1, -1))
