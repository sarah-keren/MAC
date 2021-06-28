class EnvWrappper:

    def __init__(self, env, needs_conv=False):
        self.env = env
        self.num_obs = self.env.observation_spaces[env.possible_agents[0]].shape
        self.num_actions = self.env.action_spaces[env.possible_agents[0]].n
        self.needs_conv = needs_conv
        self.env_agents = self.env.possible_agents

    def get_env(self):
        return self.env

    def get_num_obs(self):
        return self.num_obs

    def get_num_actions(self):
        return self.num_actions

    def get_needs_conv(self):
        return self.needs_conv

    def get_env_agents(self):
        return self.env_agents
