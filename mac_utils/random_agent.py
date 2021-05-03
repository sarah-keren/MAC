class RandomAgent:
    """observation_space can be either an observation space (Discrete, Box etc)
       or a list of observation_spaces
    """
    def __init__(self, observation_space):
        self.space = observation_space

    def get_action(self, observation):
        if type(self.space) == list:
            return [space.sample() for space in self.space]
        else:
            return self.space.sample()  