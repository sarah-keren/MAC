from MAC.control.controller import Controller


class Decentralized(Controller):

    def __init__(self, env, agents):
        # initialize super class
        super().__init__(env, agents)

    def get_joint_action(self, observation):

        joint_action = {}
        for agent_name in self.agents.keys():
            action = self.agents[agent_name].get_action(observation[agent_name])
            joint_action[agent_name] = action

        return joint_action






























































