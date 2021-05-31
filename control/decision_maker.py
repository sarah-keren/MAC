from abc import ABC, abstractmethod


class DecisionMaker(ABC):

    @abstractmethod
    def get_decision(self, observation):
        pass


class DecisionMakerFunc(DecisionMaker):

    # init agents and their observations
    def __init__(self, decision_func):
        self.decision_func = decision_func

    def get_decision(self, environment, observation):

        decision = self.decision_func(environment, observation)
        return decision
