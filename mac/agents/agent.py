from abc import ABC, abstractmethod


class Agent:

    def __init__(self, decision_maker, sensor_function=None, message_filter=None):
        self.decision_maker = decision_maker
        self.sensor_function = sensor_function or (lambda x: x)  # default to identity function
        self.message_filter = message_filter

    def get_decision_maker(self):
        return self.decision_maker

    def get_observation(self, state):
        return self.sensor_function(state)


class DecisionMaker(ABC):
    """
    An abstract class for choosing an action, part of an agent.
    (An agent can have one or several of these)
    """

    @abstractmethod
    def get_action(self, observation):
        pass


class RandomDecisionMaker:
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):

        #TODO this assumes that the action space is a gym space with the `sample` funcition
        if isinstance(self.space, dict):
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()
