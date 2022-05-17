class Agent:


    def __init__(self, decision_maker, sensor_function =None, message_filter = None):
        self.decision_maker = decision_maker
        self.sensor_function = sensor_function
        self.message_filter = message_filter

    def get_decision_maker(self):
        return self.decision_maker

    def get_observation(self, state):

        return self.sensor_function(state)


"""
An abstract class for choosing an action, part of an agent.
(An agent can have one or several of these)
"""
class DecisionMaker:

    def __init__(self):
        pass

    def get_action(self, observation):
        pass

    """
    Functions for training:
    """
    def get_train_action(self, observation):
        pass

    def update_step(self, obs, action,new_obs, reward, done):
        pass

    def update_episode(self, batch_size=0):
        pass

class RandomDecisionMaker:
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):
        if type(self.space) == dict:
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()


