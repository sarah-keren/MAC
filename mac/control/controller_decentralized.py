from .controller import Controller


class DecentralizedController(Controller):

    def __init__(self, env, agents):
        # initialize super class
        super().__init__(env)

        # safely accept agents a dict or as a list of agents matching the agent_ids list order
        assert len(agents) == len(self.agent_ids)
        if isinstance(agents, dict):
            assert all(agent in self.agent_ids for agent in self.agent_ids)
            self.agents = agents
        elif isinstance(agents, list):
            self.agents = {id_: agent for id_, agent in zip(self.agent_ids, agents)}

    def get_joint_action(self, observation):
        """Returns the joint action

        Args:
            observation (dict): the current observatins

        Returns:
            dict: the actions for the agents
        """
        joint_action = {}
        for agent_name in self.agent_ids:
            action = self.agents[agent_name].get_decision_maker().get_action(observation[agent_name])
            joint_action[agent_name] = action

        return joint_action
