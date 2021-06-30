from abc import ABC, abstractmethod


class Controller(ABC):

    # init agents and their observations
    def __init__(self, environment, agents, central_agent=None):
        self.environment = environment
        self.agents = agents
        self.central_agent = central_agent

    def run(self, render=False, max_iteration=None):
        done = False
        index = 0
        observation = self.environment.get_env().reset()
        while done is not True:
            index += 1
            if max_iteration is not None and index > max_iteration:
                break

            # display environment
            if render:
                self.environment.get_env().render()

            # get actions for each agent to perform

            joint_action = self.get_joint_action(observation)
            observation, reward, done, info = self.perform_joint_action(joint_action)
            done = all(value == True for value in done.values())
            if done:
                break

        if render:
            self.environment.get_env().render()

    def perform_joint_action(self, joint_action):
        return self.environment.get_env().step(joint_action)

    def get_joint_action(self, observation):
        pass


