import abc


class EnvWrappper:

    def __init__(self, env, env_agent_ids, observation_spaces, action_spaces):
        self.env = env
        self.env_agent_ids = env_agent_ids
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_env(self):
        return self.env

    def get_env_agents(self):
        return self.env_agent_ids

    def step(self, joint_action):
        return self.env.step(joint_action)


class EnvWrappperMultiTaxi(EnvWrappper):

    def __init__(self, env):

        # get action space of each agent
        action_spaces = {
            agent_id: env.action_space for agent_id in env.taxis_names
        }

        # get observation space for each agent
        observation_spaces = {
            agent_id: env.observation_space for agent_id in env.taxis_names
        }

        super().__init__(env, env.taxis_names, observation_spaces, action_spaces)


class EnvWrappperPZ:

    def __init__(self, env):

        # get action space of each agent
        action_spaces = {
            agent_id: env.action_space for agent_id in env.agents
        }

        # get observation space for each agent
        observation_spaces = {
            agent_id: env.observation_space for agent_id in env.agents
        }

        super(EnvWrappper, self).__init__(env, env.agents, observation_spaces, action_spaces)



