import numpy as np
import time

class CentralizedControl:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, max_episode_lenth=np.inf):
        observation = self.env.reset()
        i = 1
        done = [False]
        while(not all(done) and i < max_episode_lenth):
            print(f"Step {i}:")
            i += 1
            action = self.agent.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            self.env.render()
        print(f"Finished")

class DecentralizedControl:
    def __init__(self, env, agent_list):
        self.env = env
        self.agents = agent_list

    def run(self, max_episode_lenth=np.inf):
        i = 1
        observation = None
        done = [False]
        while(not all(done) and i < max_episode_lenth):
            print(f"Step {i}:")
            i += 1
            # TODO: Needs needs to be changed when adding observation filters:
            actions = [agent.get_action(observation) for agent in self.agents]
            observation, reward, done, info = self.env.step(actions)
            self.env.render()
        print(f"Finished")
        self.env.render()