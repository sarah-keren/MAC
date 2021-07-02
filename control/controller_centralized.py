from abc import abstractmethod
from MAC.control.controller import Controller
import numpy as np

"""Abstract parent class for centralized controller 
"""
class Centralized(Controller):

    def __init__(self, env, agents, central_agent):
        # initialize super class
        super().__init__(env, agents, central_agent)
        self.decision_maker = self.central_agent.get_decision_maker()

    @abstractmethod
    def get_joint_action(self, observation):
        """Returns the joint action from all the agent

        Args:
            observation (dict): All the observations
        """
        pass

    @abstractmethod
    def decode_state(self, obs):
        """Abstract method - translate from the env observations
        into an ndarray for the model process
        """
        pass

    @abstractmethod
    def decode_action(self, action, num_actions, num_agents):
        """Abstract methods - translate from the model action
        into individual actions for every agent
        """
        pass