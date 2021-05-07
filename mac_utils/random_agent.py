class RandomAgent:
    """action_space can be either an action space (Discrete, Box etc)
       or a dictionary (key per agent) of action spaces
    """
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):
        if type(self.space) == dict:
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()