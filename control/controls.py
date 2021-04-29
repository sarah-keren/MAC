import numpy as np

from MultiTaxiWrapper import MultiTaxiWrapper

class CentralizedControl:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self):
        i = 1
        while(not self.env.is_done()):
            observation = self.env.get_total_observation()
            print(f"Step {i}:")
            i += 1
            action = self.agent.get_action(observation)
            self.env.step(action)
        print(f"Finished")
        self.env.render()

class DecentralizedControl:
    def __init__(self, env, agent_list):
        self.env = env
        self.agents = agent_list

    def run(self):
        i = 1
        observation = None
        while(not self.env.is_done()):
            print(f"Step {i}:")
            i += 1
            actions = []
            for agent in self.agents:
                actions.append(agent.get_action(observation))
            joint_action = MultiTaxiWrapper.join_actions(actions)
            self.env.step(joint_action)
        print(f"Finished")
        self.env.render()