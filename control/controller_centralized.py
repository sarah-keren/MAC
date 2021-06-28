from MAC.control.controller import Controller
import numpy as np

class Centralized(Controller):

    def __init__(self, env, agents, decision_maker):
        # initialize super class
        super().__init__(env, agents, decision_maker)

    def get_joint_action(self, observation):

        observations = []
        for agent_name in self.agents.keys():
            observations.append(observation[agent_name])

        state = self.decode_state(observations, self.environment.get_needs_conv())
        joint_act = self.decision_maker.get_action(state)
        joint_act = self.decode_action(joint_act, self.environment.get_num_actions(),
                                       len(self.environment.get_env_agents()))
        joint_action = {}
        for i, agent_name in enumerate(self.environment.get_env_agents()):
            action = joint_act[i]
            joint_action[agent_name] = action

        return joint_action

    def decode_state(self, obs, needs_conv):
        if needs_conv:
            return np.vstack(obs)
        else:
            return np.hstack(obs)

    def decode_action(self, action, num_actions, num_agents):
        out = []
        for ind in range(num_agents):
            out.append(action % num_actions)
            action = action // num_actions
        return list(reversed(out))























































