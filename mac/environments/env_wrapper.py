from abc import ABC, abstractmethod


class EnvWrappper(ABC):

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

    @abstractmethod
    def step(self, joint_action):
        pass

    @abstractmethod
    def observation_to_dict(self, obs):
        pass

    def render(self):
        pass


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

    def step(self, joint_action):
        return self.env.step(joint_action)

    def observation_to_dict(self, obs):
        return obs

    def render(self):
        return self.env.render()


class EnvWrappperPZ(EnvWrappper):

    def __init__(self, env):

        # get action space of each agent
        action_spaces = {
            agent_id: env.action_space(agent_id) for agent_id in env.agent_ids
        }

        # get observation space for each agent
        observation_spaces = {
            agent_id: env.observation_space(agent_id) for agent_id in env.agent_ids
        }

        super().__init__(env, env.agent_ids, observation_spaces, action_spaces)



