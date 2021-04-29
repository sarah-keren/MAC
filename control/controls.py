import numpy as np
import time

class CentralizedControl:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self):
        observation = self.env.reset()
        i = 1
        done = [False]
        while(not all(done)):
            print(f"Step {i}:")
            i += 1
            action = self.agent.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            print(done)
            self.env.render()
        print(f"Finished")

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
            # joint_action = MultiTaxiWrapper.join_actions(actions)
            self.env.step(joint_action)
        print(f"Finished")
        self.env.render()