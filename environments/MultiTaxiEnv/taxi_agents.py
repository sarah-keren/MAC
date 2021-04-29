import numpy as np

class RandomNTaxisAgent:
    def __init__(self, agent_name_list):
        self.agent_name_list = agent_name_list

    def get_action(self, observation):
        action = {}
        for agent_name in self.agent_name_list:
            action[agent_name] = np.random.choice(np.arange(6))
        return action