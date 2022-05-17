from .controller import Controller
import numpy as np


"""Abstract parent class for centralized controller 
"""
class CentralizedController(Controller):

    def __init__(self, env, central_agent):
        # initialize super class
        super().__init__(env)

        self.central_agent = central_agent

    def get_joint_action(self, observation):
        """Returns the joint actions of all the agents

        Args:
            observation ([dict]): The agents observations

        Returns:
            dict: dict of all the actions
        """
        # observations = {}
        # # Dict to list:
        # for agent_name in self.agents:
        #     observations[agent_name] = observation[agent_name]
        #
        # state = self.decode_state(observations)
        # # centerlized decision making
        # joint_act = self.central_agent.decision_maker.get_action(state)
        # joint_act = self.decode_action(joint_act, len(self.env.get_env_agents()))
        # joint_action = {}
        # for i, agent_name in enumerate(self.env.get_env_agents()):
        #     action = joint_act[i]
        #     joint_action[agent_name] = action
        #
        # return joint_action

        return self.central_agent.get_decision_maker().get_action(observation)

    # temp implementation
    def decode_state(self, obs):
        return obs

    def decode_action(self, action, num_agents):
        """Decodes the action from the model to RL env friendly format

        Args:
            action (int): The action from the model
            num_agents (int): number of agents

        Returns:
            list: list of individual actions
        """
        out = {}
        for ind in range(num_agents):
            out.append(action % num_actions)
            action = action // num_actions
        return list(reversed(out))