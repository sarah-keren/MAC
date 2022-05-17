class EnvWrappper:

    def __init__(self, env, env_agents, num_observation_spaces=1, num_actions=1):
        self.env = env
        self.env_agents = env_agents
        self.num_observation_spaces = num_observation_spaces
        self.num_actions = num_actions

    def get_env(self):
        return self.env

    def get_num_obs(self):
        return self.num_observation_spaces

    def get_num_actions(self):
        return self.num_actions

    def get_env_agents(self):
        return self.env_agents

    def step(self, joint_action):
        return self.env.step(joint_action)



class EnvWrappperGym:

    def __init__(self, env, needs_conv=False):
        super(EnvWrappperGym, self).__init__(env, self.env.possible_agents, self.env.observation_spaces[env.possible_agents[0]].shape, self.env.action_spaces[env.possible_agents[0]].n)
        self.needs_conv = needs_conv

    def get_needs_conv(self):
        return self.needs_conv
