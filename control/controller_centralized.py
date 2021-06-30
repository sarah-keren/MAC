from MAC.control.controller import Controller
import numpy as np


class Centralized(Controller):

    def __init__(self, env, agents, central_agent):
        # initialize super class
        super().__init__(env, agents, central_agent)
        self.decision_maker = self.central_agent.get_decision_maker()

    def get_joint_action(self, observation):
        pass

    def decode_state(self, obs, needs_conv):
        pass

    def decode_action(self, action, num_actions, num_agents):
        pass























































